#!/usr/bin/env python3
import mne
import pandas as pd
import numpy as np
import os
import glob
import logging
import argparse
import gc
import re
import sys
from joblib import Parallel, delayed
from itertools import zip_longest
import typing

try:
    from tqdm import tqdm
except ImportError:
    print("tqdm not found, progress bars will not be shown for file processing.")
    def tqdm(iterable, *args, **kwargs):
        desc = kwargs.get("desc", "Processing items")
        if hasattr(iterable, '__len__'):
            print(f"{desc}: {len(iterable)} items...")
        else:
            print(f"{desc}...")
        return iterable

try:
    import neurokit2 as nk
except ImportError:
    print("NeuroKit2 not found. ECG feature extraction will be skipped. "
          "Please install it: pip install neurokit2")
    nk = None

try:
    from autoreject import AutoReject
except ImportError:
    print("autoreject not found. Fallback rejection will be used. "
          "Install with: pip install autoreject")
    AutoReject = None


parser = argparse.ArgumentParser(
    description="ECG preprocessing with categorization alignment and epoch ID."
)
parser.add_argument(
    '--base_path',
    required=True,
    help='Root folder containing sub-*/ses-*/ecg and corresponding eeg/_events.tsv data'
)
parser.add_argument(
    '--sfreq',
    type=int,
    default=256,
    help='Target sampling frequency for ECG'
)
parser.add_argument(
    '--sample_limit',
    type=int,
    default=500,
    help='Max non-seizure epochs per EDF file for ECG'
)
parser.add_argument(
    '--n_jobs',
    type=int,
    default=-1,
    help='Number of parallel jobs for joblib (-1 uses all cores)'
)
parser.add_argument(
    '--debug',
    action='store_true',
    help='Enable debug level logging.'
)
parser.add_argument(
    '--max_epochs_per_batch',
    type=int,
    default=5000,
    help='Maximum number of epochs to process in a single batch'
)
parser.add_argument(
    '--process_by_subject',
    action='store_true',
    help='Process and save data by subject instead of all at once'
)
parser.add_argument(
    '--skip_feature_extraction',
    action='store_true',
    help='Skip ECG feature extraction to speed up processing'
)
parser.add_argument(
    '--max_subjects',
    type=int,
    default=None,
    help='Limit processing to this many subjects per category'
)
parser.add_argument(
    '--node_id',
    type=int,
    default=0,
    help='Current node ID (for distributed processing)'
)
parser.add_argument(
    '--total_nodes',
    type=int,
    default=1,
    help='Total number of nodes (for distributed processing)'
)
parser.add_argument(
    '--parallel_batches',
    type=int,
    default=1,
    help='Number of batches to process in parallel'
)
parser.add_argument(
    '--process_subset',
    action='store_true',
    help='Process only a subset of files (for testing/quick results)'
)
parser.add_argument(
    '--subset_fraction',
    type=float,
    default=0.2,
    help='Fraction of files to process if process_subset is True'
)
parser.add_argument(
    '--category',
    choices=['left_bte_crosstop', 'right_bte_crosstop', 'both_bte_no_crosstop',
             'all'],
    default='all',
    help='Process only a specific BTE category'
)
parser.add_argument(
    '--feature_subset',
    action='store_true',
    help='Extract only essential ECG features (faster processing)'
)
args = parser.parse_args()

# configs
BASE_PATH = args.base_path
SFREQ = args.sfreq
BUFFER_SEC = 5
EPOCH_SAMPLE_LIMIT_NON_SEIZURE = args.sample_limit
N_JOBS = args.n_jobs

np.random.seed(42)

# same as eeg!!
EPOCH_CONFIG = {
    "preictal": (-10, 0),
    "ictal": (0, 10),
    "onset": (-5, 5),
    "non_seizure": (0, 10),
}

LOG_FILENAME = os.path.join(BASE_PATH, "ecg_preprocessing_categorized.log")
logging.basicConfig(
    filename=LOG_FILENAME,
    level=logging.DEBUG if args.debug else logging.INFO,
    format="%(asctime)s - %(levelname)s - %(processName)s - %(filename)s:%(lineno)d - %(message)s",
    filemode='w'
)
logger = logging.getLogger()


def extract_subject_id(filepath):
    """
    Extract the subject ID from a file path using multiple methods.
    Returns the subject ID in format 'sub-XXX'.
    """
    sub_match = re.search(r'/sub-([^/]+)/', filepath)
    if sub_match:
        return f"sub-{sub_match.group(1)}"

    filename = os.path.basename(filepath)
    parts = filename.split('_')
    sub_part = next((p for p in parts if p.startswith('sub-')), None)
    if sub_part:
        return sub_part

    path_parts = filepath.split(os.sep)
    sub_dir = next((p for p in path_parts if p.startswith('sub-')), None)
    if sub_dir:
        return sub_dir

    full_match = re.search(r'sub-(\d+)', filepath)
    if full_match:
        return f"sub-{full_match.group(1)}"

    logger.warning(f"Could not extract subject ID from {filepath}")
    return "sub-unknown"


def extract_session_id(filepath):
    """
    Extract the session ID from a file path using multiple methods.
    Returns the session ID in format 'ses-XXX'.
    """

    ses_match = re.search(r'/ses-([^/]+)/', filepath)
    if ses_match:
        return f"ses-{ses_match.group(1)}"

    filename = os.path.basename(filepath)
    parts = filename.split('_')
    ses_part = next((p for p in parts if p.startswith('ses-')), None)
    if ses_part:
        return ses_part

    path_parts = filepath.split(os.sep)
    ses_dir = next((p for p in path_parts if p.startswith('ses-')), None)
    if ses_dir:
        return ses_dir

    full_match = re.search(r'ses-(\d+)', filepath)
    if full_match:
        return f"ses-{full_match.group(1)}"

    return "ses-001"


def extract_ecg_features_from_epoch_data(
        epoch_data_segment, sampling_rate, epoch_id_str="N/A") -> typing.Union[dict, None]:
    """
    Extracts ECG features (HR, HRV) from a single epoch's data segment.
    Focused only on essential features for seizure onset prediction.
    """
    logger.debug(f"Starting ECG feature extraction for epoch: {epoch_id_str}")
    if nk is None:
        logger.warning(f"NeuroKit2 not available for epoch {epoch_id_str}. Skipping ECG feature extraction.")
        return {}

    if epoch_data_segment.ndim > 1:
        logger.debug(f"Epoch {epoch_id_str}: data has shape {epoch_data_segment.shape}, squeezing.")
        epoch_data_segment = epoch_data_segment.squeeze()

    if np.isnan(epoch_data_segment).all() or \
       np.all(epoch_data_segment == 0) or \
       np.std(epoch_data_segment) < 1e-7:
        logger.warning(
            f"ECG signal for epoch {epoch_id_str} is flat, zero, or NaN (std: {np.std(epoch_data_segment)}). "
            "Skipping feature extraction."
        )
        return {}

    features = {}
    try:
        logger.debug(f"Epoch {epoch_id_str}: Processing with nk.ecg_process.")
        ecg_signals, info = nk.ecg_process(
            epoch_data_segment, sampling_rate=sampling_rate
        )
        if info is None or "ECG_R_Peaks" not in info or \
           len(info["ECG_R_Peaks"]) == 0:
            logger.warning(
                f"Not enough R-peaks for features in {epoch_id_str} after nk.ecg_process."
            )
            return features

        rpeaks = np.asarray(info["ECG_R_Peaks"])
        logger.debug(f"Epoch {epoch_id_str}: Found {len(rpeaks)} R-peaks.")

        if rpeaks.size < 4:
            logger.warning(
                f"Too few R-peaks ({rpeaks.size}) for robust HRV in {epoch_id_str}. "
                "Attempting HR calculation if possible."
            )
            if rpeaks.size >= 1:
                try:
                    logger.debug(f"Epoch {epoch_id_str}: Calculating HR with {rpeaks.size} R-peaks.")
                    hr_signals = nk.ecg_rate(
                        rpeaks, sampling_rate=sampling_rate,
                        desired_length=len(epoch_data_segment)
                    )
                    features["HR_Mean"] = np.nanmean(hr_signals) \
                        if hr_signals.size > 0 else np.nan
                    features["HR_Std"] = np.nanstd(hr_signals) \
                        if hr_signals.size > 0 else np.nan
                    logger.debug(f"Epoch {epoch_id_str}: Calculated HR_Mean: {features.get('HR_Mean')}, HR_Std: {features.get('HR_Std')}")
                except Exception as e:
                    logger.warning(f"Epoch {epoch_id_str}: HR calculation failed with few R-peaks: {e}")
                    pass
            return features

        # heart rate features
        logger.debug(f"Epoch {epoch_id_str}: Calculating HR signals.")
        hr_signals = nk.ecg_rate(
            rpeaks, sampling_rate=sampling_rate,
            desired_length=len(epoch_data_segment)
        )
        features["HR_Mean"] = np.nanmean(hr_signals) if hr_signals.size > 0 else np.nan
        features["HR_Std"] = np.nanstd(hr_signals) if hr_signals.size > 0 else np.nan
        logger.debug(f"Epoch {epoch_id_str}: Calculated HR_Mean: {features['HR_Mean']}, HR_Std: {features['HR_Std']}")

        if args.feature_subset:
            if rpeaks.size >= 2:
                # calculate RR intervals
                rri = np.diff(rpeaks) / sampling_rate * 1000 # ms
                features["RMSSD"] = np.sqrt(np.mean(np.square(np.diff(rri))))
                features["SDNN"] = np.std(rri)
            return features

        # time domain HRV
        try:
            logger.debug(f"Epoch {epoch_id_str}: Calculating HRV time-domain features.")
            hrv_time = nk.hrv_time(rpeaks, sampling_rate=sampling_rate, show=False)
            for col in ["RMSSD", "MeanNN", "SDNN", "MedianNN", "pNN50"]:
                for prefix in ["HRV_", ""]:
                    if f"{prefix}{col}" in hrv_time.columns:
                        features[col] = hrv_time[f"{prefix}{col}"].iloc[0]
                        logger.debug(f"Epoch {epoch_id_str}: Found {col}: {features[col]}")
                        break
        except Exception as e:
            logger.warning(f"Epoch {epoch_id_str}: HRV time-domain calculation failed: {e}")
            for col in ["RMSSD", "MeanNN", "SDNN", "MedianNN", "pNN50"]:
                features[col] = np.nan

        # freq domain HRV features
        try:
            logger.debug(f"Epoch {epoch_id_str}: Calculating HRV frequency-domain features.")
            hrv_freq = nk.hrv_frequency(
                rpeaks, sampling_rate=sampling_rate, show=False
            )
            for col_nk, col_feat in [
                    ("LF", "LF"), ("HF", "HF"),
                    ("LF/HF", "LFHF"), ("LFHF", "LFHF")]:
                for prefix in ["HRV_", ""]:
                    if f"{prefix}{col_nk}" in hrv_freq.columns:
                        features[col_feat] = hrv_freq[f"{prefix}{col_nk}"].iloc[0]
                        logger.debug(f"Epoch {epoch_id_str}: Found {col_feat}: {features[col_feat]}")
                        break
        except Exception as e:
            logger.warning(f"Epoch {epoch_id_str}: HRV frequency-domain calculation failed: {e}")
            for col_feat in ["LF", "HF", "LFHF"]:
                features[col_feat] = np.nan

    except Exception as e:
        logger.warning(
            f"Feature extraction failed for epoch {epoch_id_str}: {e}"
        )

    expected_keys = [
        "HR_Mean", "HR_Std",
        "RMSSD", "MeanNN", "SDNN", "MedianNN", "pNN50",
        "LF", "HF", "LFHF"
    ]

    for k in expected_keys:
        if k not in features:
            features[k] = np.nan
            logger.debug(f"Epoch {epoch_id_str}: Key '{k}' was missing, added as NaN.")
    
    logger.debug(f"Epoch {epoch_id_str}: Feature extraction completed. Features: {list(features.keys())}")
    return features


def create_epoch_for_file(
        raw, event_onset_time, tmin, tmax, epoch_key_local, subject_id, session_id):
    """
    Creates a single MNE epoch with subject ID in metadata, matching EEG preprocessing format.
    """
    fname = os.path.basename(raw.filenames[0]) if raw.filenames else "UnkFile"
    start_samp = int((event_onset_time + tmin) * raw.info['sfreq'])
    end_samp = int((event_onset_time + tmax) * raw.info['sfreq'])

    if start_samp < 0 or end_samp > raw.n_times:
        logger.warning(
            f"Epoch {epoch_key_local} @{event_onset_time:.2f}s in {fname} OOB. "
            f"Need [{start_samp/raw.info['sfreq']:.2f}s, "
            f"{end_samp/raw.info['sfreq']:.2f}s] vs "
            f"Raw dur: {raw.times[-1]:.2f}s. Skipping."
        )
        return None

    try:
        event_type_id = {'preictal': 1, 'ictal': 2, 'onset': 3, 'non_seizure': 4}.get(epoch_key_local, 1)
        event_arr = np.array([[int(event_onset_time * raw.info['sfreq']), 0,
                               event_type_id]])

        epoch = mne.Epochs(raw, event_arr, tmin=tmin, tmax=tmax,
                          baseline=None, preload=True, verbose=False, picks='ecg')

        epoch.metadata = pd.DataFrame({
            'subject_id': [subject_id],
            'session_id': [session_id],
            'epoch_type_class': [epoch_key_local],
            'unique_epoch_id': [f"{subject_id}_{session_id}_{epoch_key_local}_0000"],  # Index will need update if multiple such events per file
            'event_onset': [event_onset_time]
        })

        return epoch
    except Exception as e:
        logger.warning(
            f"Failed epoch: {epoch_key_local} @{event_onset_time:.2f}s "
            f"in {fname}: {e}"
        )
        return None


def process_non_seizure_epochs_for_file(
        raw, seizure_intervals_samples, epoch_duration_sec,
        target_sfreq_local, max_epochs_per_file_local, buffer_sec_local,
        epoch_config_local_ns, subject_id, session_id):
    """
    Extracts non-seizure epochs from a single raw file.
    Aligned with EEG preprocessing approach.
    """
    fname = os.path.basename(raw.filenames[0]) if raw.filenames else "UnkFile"
    mask = np.ones(raw.n_times, dtype=bool)
    safety_buf = int(buffer_sec_local * target_sfreq_local)

    for start_s, end_s in seizure_intervals_samples:
        mask[max(0, start_s - safety_buf):min(raw.n_times, end_s + safety_buf)] = False

    epoch_len_samps = int(epoch_duration_sec * target_sfreq_local)
    if epoch_len_samps <= 0:
        logger.error(f"Non-seizure epoch len <=0 for {fname}. Skipping.")
        return None

    valid_starts = np.where(np.convolve(
        mask, np.ones(epoch_len_samps, dtype=int), 'valid'
    ) == epoch_len_samps)[0]

    if len(valid_starts) == 0:
        logger.info(f"No valid non-seizure segments in {fname}.")
        return None

    indices_to_select = []
    if max_epochs_per_file_local and len(valid_starts) > max_epochs_per_file_local:
        step = len(valid_starts) // max_epochs_per_file_local
        indices_to_select = np.arange(0, len(valid_starts),
                                      step)[:max_epochs_per_file_local]
    else:
        indices_to_select = np.arange(len(valid_starts))

    actual_selected_starts = valid_starts[indices_to_select]

    if len(actual_selected_starts) == 0:
        logger.info(f"No non-overlapping non-seizure epochs for {fname} "
                    f"after selection.")
        return None

    events_arr = np.array([[s, 0, 4] for s in actual_selected_starts])
    tmin_ns, tmax_ns = epoch_config_local_ns['non_seizure']
    try:
        epochs = mne.Epochs(raw, events_arr, tmin=tmin_ns, tmax=tmax_ns,
                            baseline=None, preload=True, verbose=False,
                            picks='ecg')

        metadata_entries = []
        for i in range(len(epochs)):
            metadata_entries.append({
                'subject_id': subject_id,
                'session_id': session_id,
                'epoch_type_class': 'non_seizure',
                'unique_epoch_id': f"{subject_id}_{session_id}_non_seizure_{i:04d}",
                'event_onset': actual_selected_starts[i] / raw.info['sfreq']
            })

        epochs.metadata = pd.DataFrame(metadata_entries)
        return epochs
    except Exception as e:
        logger.error(f"Error creating non-seizure epochs for {fname}: {e}")
        return None


def fallback_ecg_rejection(
        epochs, file_identifier_for_log="N/A",
        threshold_uV=500.0, min_samples=None):
    """
    Basic fallback rejection for ECG epochs.
    Aligned with EEG rejection approach but adapted for ECG characteristics.
    """
    logger.debug(f"Applying fallback ECG rejection. Initial epochs: "
                 f"{len(epochs)}, Threshold: {threshold_uV} uV, "
                 f"Min samples: {min_samples}")

    if len(epochs) == 0:
        logger.info("Fallback rejection: No epochs to process.")
        return epochs

    data = epochs.get_data(copy=False)

    threshold_V = threshold_uV * 1e-6
    bad_amp = np.any(np.abs(data) > threshold_V, axis=(1, 2))

    var = np.var(data, axis=(1, 2))
    bad_var = np.zeros(len(var), dtype=bool)

    if len(var) > 1 and np.std(var) > 1e-14:
        m, s = np.mean(var), np.std(var)
        bad_var = np.abs(var - m) > 3 * s

    bad_flat = var < 1e-12

    bad = bad_amp | bad_var | bad_flat
    keep = ~bad
    num_original, num_kept = len(epochs), keep.sum()

    if min_samples and num_kept < min_samples and num_original >= min_samples:
        logger.warning(
            f"Fallback rejection for {file_identifier_for_log} initially "
            f"kept {num_kept}/{num_original}. Trying to keep {min_samples}."
        )
        if np.std(var) > 1e-14:
            idx_to_keep = np.argsort(np.abs(var - np.mean(var)))[:min_samples]
        else:
            idx_to_keep = np.arange(min(min_samples, num_original))

        keep = np.zeros(num_original, dtype=bool)
        keep[idx_to_keep] = True
        num_kept = keep.sum()

    logger.info(
        f"Fallback rejection for {file_identifier_for_log}: "
        f"kept {num_kept}/{num_original} epochs."
    )
    return epochs[keep]


def get_eeg_bte_category(ecg_file_path: str) -> typing.Union[str, None]:
    """
    Determines the BTE category of the corresponding EEG file.
    """
    logger.debug(f"Determining EEG BTE category for ECG file: {ecg_file_path}")

    subject_id = extract_subject_id(ecg_file_path)
    session_id = extract_session_id(ecg_file_path)

    eeg_file_path = ecg_file_path.replace('/ecg/', '/eeg/').replace('_ecg.edf',
                                                                    '_eeg.edf')
    logger.debug(f"Corresponding EEG file path to check: {eeg_file_path}")

    if not os.path.exists(eeg_file_path):
        logger.warning(f"EEG file not found at {eeg_file_path}")
        return None

    try:
        logger.debug(f"Reading info from EEG file: {eeg_file_path}")
        eeg_info = mne.io.read_raw_edf(eeg_file_path, preload=False,
                                       verbose=False).info
        ch_names = eeg_info['ch_names']
        logger.debug(f"EEG channels found: {ch_names}")

        has_bte_left = 'BTEleft SD' in ch_names
        has_bte_right = 'BTEright SD' in ch_names
        has_crosstop = 'CROSStop SD' in ch_names

        if has_bte_left and has_crosstop:
            logger.info(f"EEG BTE category for {eeg_file_path}: left_bte_crosstop")
            return 'left_bte_crosstop'
        elif has_bte_right and has_crosstop:
            logger.info(f"EEG BTE category for {eeg_file_path}: right_bte_crosstop")
            return 'right_bte_crosstop'
        elif has_bte_left and has_bte_right and not has_crosstop:
            logger.info(f"EEG BTE category for {eeg_file_path}: both_bte_no_crosstop")
            return 'both_bte_no_crosstop'
        else:
            logger.info(f"EEG file {eeg_file_path} does not fit defined BTE categories.")
            return None
    except Exception as e:
        logger.error(f"Could not read/inspect EEG file {eeg_file_path}: {e}", exc_info=True)
        return None


def find_corresponding_events_file(ecg_file_path: str) -> typing.Union[str, None]:
    """
    Finds the _events.tsv file for a given ECG file.
    Expected in the corresponding eeg directory.
    """
    logger.debug(f"Finding events file for ECG file: {ecg_file_path}")
    events_tsv = ecg_file_path.replace('/ecg/', '/eeg/').replace('_ecg.edf', '_events.tsv')
    
    if os.path.exists(events_tsv):
        logger.info(f"Found events file: {events_tsv}")
        return events_tsv
    else:
        logger.warning(f"No events file found at {events_tsv}")
        return None


def process_single_ecg_file(
        ecg_path, target_sfreq, epoch_config_ref,
        sample_limit_ns_ref, buffer_sec_ref):
    """
    Processes a single ECG file and determines its EEG BTE category.
    Returns (epochs_dict, bte_category_string, subject_id, session_id) to match EEG processing.
    """
    original_basename = os.path.basename(ecg_path)
    logger.info(f"--- Processing ECG file: {original_basename} ---")

    subject_id = extract_subject_id(ecg_path)
    session_id = extract_session_id(ecg_path)
    logger.info(f"Extracted subject ID: {subject_id}, session ID: {session_id}")

    local_epochs = {key: [] for key in epoch_config_ref}

    bte_category = get_eeg_bte_category(ecg_path)
    if bte_category is None:
        logger.warning(f"Could not determine BTE category for {original_basename}. Skipping.")
        return None, None, subject_id, session_id

    if args.category != 'all' and bte_category != args.category:
        logger.info(f"Skipping {original_basename} with category {bte_category} (only processing {args.category})")
        return None, None, subject_id, session_id

    try:
        logger.info(f"Loading ECG file: {ecg_path}")
        raw = mne.io.read_raw_edf(ecg_path, preload=True, verbose=False)
        logger.info(f"Raw data loaded: {raw.n_times} samples, sfreq: {raw.info['sfreq']} Hz")

        if raw.n_times == 0:
            logger.warning(f"ECG file {original_basename} is empty (0 samples). Skipping.")
            return None, bte_category, subject_id, session_id

        ecg_ch_names = [ch for ch in raw.ch_names if "ECG" in ch.upper() or "EKG" in ch.upper()]
        if not ecg_ch_names:
            logger.warning(f"No ECG channel found in {original_basename}. Skipping.")
            return None, bte_category, subject_id, session_id

        picked_ecg_channel = ecg_ch_names[0]
        logger.info(f"Using ECG channel: '{picked_ecg_channel}'.")
        raw.pick(picks=[picked_ecg_channel])
        raw.set_channel_types({picked_ecg_channel: "ecg"})

        logger.info("Filtering ECG data (0.5-40 Hz) and notch (50 Hz).")
        raw.filter(0.5, 40., fir_design='firwin', picks='ecg', verbose=False)
        raw.notch_filter(50, fir_design='firwin', picks='ecg', verbose=False)

        if raw.info["sfreq"] != target_sfreq:
            logger.info(f"Resampling ECG from {raw.info['sfreq']} Hz to {target_sfreq} Hz")
            raw.resample(target_sfreq, npad='auto', verbose=False)

        events_tsv_path = find_corresponding_events_file(ecg_path)
        if not events_tsv_path:
            logger.warning(f"No events file found for {original_basename}.")
            return None, bte_category, subject_id, session_id

        logger.info(f"Loading events from: {events_tsv_path}")
        try:
            events_df = pd.read_csv(events_tsv_path, sep='\t')
        except Exception as e:
            logger.error(f"Error reading events file {events_tsv_path}: {e}")
            return None, bte_category, subject_id, session_id

        seizure_intervals_samples = []
        sz_events_df = events_df[events_df['eventType'].str.startswith('sz_', na=False)] \
            if 'eventType' in events_df.columns else pd.DataFrame()

        for _, r in sz_events_df.iterrows():
            try:
                on_val = pd.to_numeric(r['onset'])
                dur_val = pd.to_numeric(r['duration'])
                if pd.isna(on_val) or pd.isna(dur_val):
                    continue
                on_s = int(on_val * raw.info['sfreq'])
                dur_s = int(dur_val * raw.info['sfreq'])
                seizure_intervals_samples.append((on_s, on_s + dur_s))
            except (ValueError, TypeError) as ve:
                logger.warning(f"Value/Type Error in onset/duration: {ve} for {original_basename}. Skipping event.")
                continue

        if not sz_events_df.empty:
            for idx_sz, sz_row in sz_events_df.iterrows():
                try:
                    event_on = pd.to_numeric(sz_row['onset'])
                    if pd.isna(event_on): 
                        continue
                except (ValueError, TypeError) as ve:
                    logger.warning(f"Value/Type Error in onset: {ve} for {original_basename}. Skipping sz_event row.")
                    continue

                for key_cls in ('preictal', 'ictal', 'onset'):
                    tmin, tmax = epoch_config_ref[key_cls]
                    epoch = create_epoch_for_file(raw, event_on, tmin, tmax, key_cls, subject_id, session_id)
                    if epoch:
                        epoch.metadata['unique_epoch_id'] = f"{subject_id}_{session_id}_{key_cls}_{idx_sz:04d}"
                        local_epochs[key_cls].append(epoch)
        elif not events_df.empty:
            logger.info(f"No seizure events in {events_tsv_path} for {original_basename}.")

        ns_dur = epoch_config_ref['non_seizure'][1] - epoch_config_ref['non_seizure'][0]
        ns_eps = process_non_seizure_epochs_for_file(
            raw, seizure_intervals_samples, ns_dur, target_sfreq,
            sample_limit_ns_ref, buffer_sec_ref, epoch_config_ref,
            subject_id, session_id
        )
        if ns_eps and len(ns_eps) > 0:
            local_epochs['non_seizure'].append(ns_eps)

        counts_str_parts = []
        for k, v_list in local_epochs.items():
            current_count = 0
            if v_list:
                for ep_item in v_list:
                    if isinstance(ep_item, mne.BaseEpochs):
                        current_count += len(ep_item)
            counts_str_parts.append(f"{k.capitalize()}: {current_count}")

        logger.info(
            f"Finished {original_basename} ({subject_id}). ECG epochs this file - "
            f"{', '.join(counts_str_parts)}"
        )

        del raw
        gc.collect()
        return local_epochs, bte_category, subject_id, session_id

    except Exception as e:
        logger.error(f"Error processing {original_basename}: {e}", exc_info=True)
        if 'raw' in locals(): del raw
        gc.collect()
        return None, bte_category, subject_id, session_id


def process_epoch_batch(epochs_batch, key_class, category_suffix, batch_id, output_dir):
    """
    Process and save a single batch of epochs.
    Optimized for parallel processing.
    """
    try:
        all_eps = mne.concatenate_epochs(epochs_batch)
        if all_eps.metadata is None:
            logger.warning(f"Missing metadata after concatenation for batch {batch_id}. Creating empty DataFrame.")
            all_eps.metadata = pd.DataFrame(index=range(len(all_eps)))

        min_s = 10 if key_class != 'non_seizure' else 50
        file_id_log = f"{batch_id}"

        eps_to_save = fallback_ecg_rejection(
            all_eps, file_identifier_for_log=file_id_log, min_samples=min_s
        )

        if len(eps_to_save) > 0:
            logger.info(f"Extracting ECG features for {len(eps_to_save)} epochs in batch {batch_id}")

            metadata_df = eps_to_save.metadata.copy().reset_index(drop=True) if eps_to_save.metadata is not None else pd.DataFrame(index=range(len(eps_to_save)))

            if not args.skip_feature_extraction:
                if len(eps_to_save) > 50:
                    epoch_datas = [eps_to_save[i].get_data(copy=False).squeeze() for i in range(len(eps_to_save))]
                    epoch_ids = [metadata_df.loc[i, 'unique_epoch_id'] if 'unique_epoch_id' in metadata_df.columns else f"epoch_{i}" for i in range(len(eps_to_save))]
                    all_features = Parallel(n_jobs=min(8, os.cpu_count()))(
                        delayed(extract_ecg_features_from_epoch_data)(
                            epoch_data, eps_to_save.info['sfreq'], epoch_id
                        ) for epoch_data, epoch_id in zip(epoch_datas, epoch_ids)
                    )
                    for i, features in enumerate(all_features):
                        for feat_name, feat_val in features.items():
                            metadata_df.loc[i, feat_name] = feat_val
                else:
                    for i in range(len(eps_to_save)):
                        epoch_data = eps_to_save[i].get_data(copy=False).squeeze()
                        epoch_id_str = metadata_df.loc[i, 'unique_epoch_id'] if 'unique_epoch_id' in metadata_df.columns else f"epoch_{i}"

                        features = extract_ecg_features_from_epoch_data(
                            epoch_data, eps_to_save.info['sfreq'], epoch_id_str
                        )

                        for feat_name, feat_val in features.items():
                            metadata_df.loc[i, feat_name] = feat_val
            else:
                logger.info(f"Skipping feature extraction for batch {batch_id} as requested via --skip_feature_extraction")

            eps_to_save.metadata = metadata_df
            category_dir = os.path.join(output_dir, category_suffix)
            os.makedirs(category_dir, exist_ok=True)
            out_fname = os.path.join(category_dir, f"{key_class}_{batch_id}-epo.fif")
            eps_to_save.save(out_fname, overwrite=True, verbose=False)
            logger.info(f"Saved {len(eps_to_save)} clean epochs in batch {batch_id} to: {out_fname}")

            if eps_to_save.metadata is not None:
                meta_fname = os.path.join(category_dir, f"{key_class}_{batch_id}_metadata.csv")
                eps_to_save.metadata.to_csv(meta_fname, index=False)
                logger.info(f"Saved metadata for batch {batch_id} to {meta_fname}")
        else:
            logger.warning(f"No epochs left after rejection for batch {batch_id}. Nothing saved.")

    except Exception as e:
        logger.error(f"Error processing batch {batch_id}: {e}", exc_info=True)
    for ep in epochs_batch:
        del ep
    if 'all_eps' in locals(): del all_eps
    if 'eps_to_save' in locals(): del eps_to_save
    gc.collect()


def process_epochs_in_batches(epochs_list, batch_size, key_class, category_suffix, output_dir):
    """
    Process epochs in smaller batches to reduce memory usage.
    Optimized for parallel processing.
    """
    if not epochs_list:
        logger.info(f"No epochs to process for {key_class} ({category_suffix}).")
        return
    total_epochs = sum(len(ep) for ep in epochs_list if isinstance(ep, mne.BaseEpochs))
    if total_epochs == 0:
        logger.warning(f"No valid MNE Epochs objects with data for {key_class} ({category_suffix}).")
        return

    logger.info(f"Processing {total_epochs} {key_class} epochs for {category_suffix} in batches of {batch_size}")

    all_subjects = set()
    for ep in epochs_list:
        if ep.metadata is not None and 'subject_id' in ep.metadata.columns:
            all_subjects.update(ep.metadata['subject_id'].unique())

    logger.info(f"Found {len(all_subjects)} unique subjects in {key_class} ({category_suffix}) epochs")
    batches = []
    current_batch = []
    current_count = 0

    for ep in epochs_list:
        if current_count + len(ep) > batch_size and current_batch:
            batches.append((current_batch, current_count, len(batches) + 1))
            current_batch = []
            current_count = 0
        current_batch.append(ep)
        current_count += len(ep)
    if current_batch:
        batches.append((current_batch, current_count, len(batches) + 1))
    if args.parallel_batches > 1 and len(batches) > 1:
        logger.info(f"Processing {len(batches)} batches in parallel (max {args.parallel_batches} at once)")

        Parallel(n_jobs=min(args.parallel_batches, len(batches)))(
            delayed(process_epoch_batch)(
                batch_eps, key_class, category_suffix,
                f"{key_class}_{category_suffix}_batch{batch_num}", output_dir
            ) for batch_eps, _, batch_num in batches
        )
    else:
        for batch_eps, batch_count, batch_num in batches:
            logger.info(f"Processing batch {batch_num} with {batch_count} epochs")
            process_epoch_batch(
                batch_eps, key_class, category_suffix,
                f"{key_class}_{category_suffix}_batch{batch_num}", output_dir
            )
            gc.collect()


def process_by_subject(results, output_path):
    """
    Process and save data by subject.
    Matches EEG preprocessing functionality.
    """
    logger.info("Processing data by subject...")

    subject_data = {}
    
    for result_tuple in results:
        if result_tuple and result_tuple[0] is not None and result_tuple[1] is not None:
            file_epochs_dict, bte_cat_str, subject_id, session_id = result_tuple
            
            if subject_id not in subject_data:
                subject_data[subject_id] = {
                    'left_bte_crosstop': {key: [] for key in EPOCH_CONFIG},
                    'right_bte_crosstop': {key: [] for key in EPOCH_CONFIG},
                    'both_bte_no_crosstop': {key: [] for key in EPOCH_CONFIG}
                }

            target_container = subject_data[subject_id][bte_cat_str]
            
            for key, ep_list_from_file in file_epochs_dict.items():
                if ep_list_from_file:
                    for ep_obj in ep_list_from_file:
                        if isinstance(ep_obj, mne.BaseEpochs) and len(ep_obj) > 0:
                            target_container[key].append(ep_obj)

    if args.max_subjects and len(subject_data) > args.max_subjects:
        logger.info(f"Limiting to maximum {args.max_subjects} subjects (from {len(subject_data)})")
        selected_subjects = list(subject_data.keys())[:args.max_subjects]
        subject_data = {subj: subject_data[subj] for subj in selected_subjects}

    for subject_id, categories in subject_data.items():
        logger.info(f"Processing subject: {subject_id}")
        subject_output_path = os.path.join(output_path, subject_id)
        os.makedirs(subject_output_path, exist_ok=True)

        for cat_name, container in categories.items():
            if args.category != 'all' and cat_name != args.category:
                continue

            has_data = any(len(epochs_list) > 0 for epochs_list in container.values())
            if not has_data:
                continue

            for epoch_type, epochs_list in container.items():
                if not epochs_list:
                    continue

                process_epochs_in_batches(
                    epochs_list, 
                    args.max_epochs_per_batch, 
                    epoch_type, 
                    cat_name, 
                    subject_output_path
                )

                container[epoch_type] = []
                gc.collect()

        del subject_data[subject_id]
        gc.collect()

        logger.info(f"Finished processing subject: {subject_id}")


def save_epochs_by_category(epochs_dict, category_suffix, base_out_path):
    """
    Concatenate, reject, and save epochs for a given category in batches.
    Matches EEG preprocessing functionality.
    """
    logger.info(f"\n--- Processing aggregated {category_suffix} epochs ---")
    output_category_dir = os.path.join(base_out_path, category_suffix)
    os.makedirs(output_category_dir, exist_ok=True)
    logger.info(f"Output directory for {category_suffix}: {output_category_dir}")

    for key_class, collected_ep_list in epochs_dict.items():
        process_epochs_in_batches(
            collected_ep_list, 
            args.max_epochs_per_batch, 
            key_class, 
            category_suffix, 
            base_out_path
        )

        epochs_dict[key_class] = []
        gc.collect()


def main():
    """Main execution flow for the ECG preprocessing pipeline."""
    logger.info(" Starting ECG Preprocessing Pipeline")
    logger.info(f"Script executed at: {pd.Timestamp.now()}")
    logger.info(f"Logging to file: {LOG_FILENAME}")

    logger.info(f"--- Configuration ---")
    logger.info(f"Base path: {BASE_PATH}")
    logger.info(f"Target sampling frequency: {SFREQ} Hz")
    logger.info(f"Non-seizure epoch sample limit per file: {EPOCH_SAMPLE_LIMIT_NON_SEIZURE}")
    logger.info(f"Buffer around seizure events: {BUFFER_SEC} seconds")
    logger.info(f"Number of parallel jobs: {N_JOBS if N_JOBS != -1 else os.cpu_count()}")
    logger.info(f"Max epochs per batch: {args.max_epochs_per_batch}")
    logger.info(f"Processing by subject: {args.process_by_subject}")
    logger.info(f"Skip feature extraction: {args.skip_feature_extraction}")
    logger.info(f"Feature subset mode: {args.feature_subset}")
    logger.info(f"Max subjects: {args.max_subjects}")
    logger.info(f"Processing category: {args.category}")
    logger.info(f"Node ID / Total nodes: {args.node_id} / {args.total_nodes}")
    logger.info(f"Parallel batches: {args.parallel_batches}")
    logger.info(f"Process subset: {args.process_subset} ({args.subset_fraction if args.process_subset else 'N/A'})")
    logger.info(f"NeuroKit2 available: {'Yes' if nk else 'No (feature extraction limited)'}")
    logger.info(f"Autoreject available: {'Yes' if AutoReject else 'No (using fallback rejection)'}")

    # find all ECG files using the same pattern as in EEG script
    logger.info("Searching for ECG files...")
    all_ecg_files = glob.glob(
        os.path.join(BASE_PATH, "sub-*", "ses-*", "ecg", "*_ecg.edf")
    )
    if not all_ecg_files:
        logger.error("No ECG files found. Check --base_path.")
        return
    logger.info(f"Found {len(all_ecg_files)} ECG files.")

    if args.total_nodes > 1:
        files_per_node = len(all_ecg_files) // args.total_nodes
        start_idx = args.node_id * files_per_node
        end_idx = start_idx + files_per_node if args.node_id < args.total_nodes - 1 else len(all_ecg_files)
        node_ecg_files = all_ecg_files[start_idx:end_idx]
        logger.info(f"Node {args.node_id}/{args.total_nodes}: Processing {len(node_ecg_files)}/{len(all_ecg_files)} files")
        all_ecg_files = node_ecg_files

    if args.process_subset:
        subset_size = int(len(all_ecg_files) * args.subset_fraction)
        if subset_size > 0:
            np.random.seed(42)
            selected_indices = np.random.choice(len(all_ecg_files),
                                                subset_size, replace=False)
            all_ecg_files = [all_ecg_files[i] for i in selected_indices]
            logger.info(f"Processing subset: {len(all_ecg_files)} files ({args.subset_fraction*100:.1f}% of total)")
        else:
            logger.warning(f"Subset size would be 0 with fraction {args.subset_fraction}. Using at least 1 file.")
            all_ecg_files = [all_ecg_files[0]]

    if args.n_jobs == 1:
        logger.info("Using sequential processing")
        results = []
        for ecg_path in tqdm(all_ecg_files, desc="Processing ECG files"):
            try:
                result = process_single_ecg_file(
                    ecg_path, SFREQ, EPOCH_CONFIG,
                    args.sample_limit, BUFFER_SEC
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {ecg_path}: {e}", exc_info=True)
                results.append((None, None, None, None))
        logger.info(f"Using parallel processing (n_jobs={args.n_jobs})")
        results = Parallel(n_jobs=args.n_jobs)(
            delayed(process_single_ecg_file)(
                ecg_path, SFREQ, EPOCH_CONFIG,
                args.sample_limit, BUFFER_SEC
            ) for ecg_path in tqdm(all_ecg_files, desc="Processing ECG files")
        )

    logger.info("File processing complete. Aggregating results.")

    if args.process_by_subject:
        output_path = os.path.join(BASE_PATH, "derivatives",
                                   "ecg_preprocessed_by_subject")
        os.makedirs(output_path, exist_ok=True)
        logger.info(f"Processing data by subject. Saving to: {output_path}")
        process_by_subject(results, output_path)
    else:
        epochs_left_bte_crosstop = {key: [] for key in EPOCH_CONFIG}
        epochs_right_bte_crosstop = {key: [] for key in EPOCH_CONFIG}
        epochs_both_bte_no_crosstop = {key: [] for key in EPOCH_CONFIG}

        for result_tuple in results:
            if result_tuple and result_tuple[0] is not None and result_tuple[1] is not None:
                file_epochs_dict, bte_cat_str, _, _ = result_tuple
                if args.category != 'all' and bte_cat_str != args.category:
                    continue
                if bte_cat_str == 'left_bte_crosstop':
                    target_container = epochs_left_bte_crosstop
                elif bte_cat_str == 'right_bte_crosstop':
                    target_container = epochs_right_bte_crosstop
                elif bte_cat_str == 'both_bte_no_crosstop':
                    target_container = epochs_both_bte_no_crosstop
                else:
                    logger.warning(f"Unknown BTE category: {bte_cat_str} from a file processing result, skipping.")
                    continue

                for key, ep_list_from_file in file_epochs_dict.items():
                    if ep_list_from_file:
                        for ep_obj in ep_list_from_file:
                            if isinstance(ep_obj, mne.BaseEpochs) and len(ep_obj) > 0:
                                target_container[key].append(ep_obj)
        if args.max_subjects:
            logger.info(f"Limiting to maximum {args.max_subjects} subjects per category")
            for category_data in [epochs_left_bte_crosstop, epochs_right_bte_crosstop, epochs_both_bte_no_crosstop]:
                for epoch_type, epochs_list in category_data.items():
                    all_subjects = set()
                    for ep in epochs_list:
                        if ep.metadata is not None and 'subject_id' in ep.metadata.columns:
                            all_subjects.update(ep.metadata['subject_id'].unique())

                    if len(all_subjects) > args.max_subjects:

                        selected_subjects = list(all_subjects)[:args.max_subjects]
                        logger.info(f"For {epoch_type}, limiting from {len(all_subjects)} to {len(selected_subjects)} subjects")

                        filtered_epochs = []
                        for ep in epochs_list:
                            if ep.metadata is not None and 'subject_id' in ep.metadata.columns:
                                mask = ep.metadata['subject_id'].isin(selected_subjects)
                                if mask.any():
                                    filtered_epochs.append(ep[mask])

                        category_data[epoch_type] = filtered_epochs

        logger.info("\n--- Aggregated Epoch Counts Before Final Processing ---")
        for cat_name, container in [
            ("left_bte_crosstop", epochs_left_bte_crosstop),
            ("right_bte_crosstop", epochs_right_bte_crosstop),
            ("both_bte_no_crosstop", epochs_both_bte_no_crosstop)
        ]:

            if args.category != 'all' and cat_name != args.category:
                continue

            logger.info(f"Category: {cat_name}")
            for epoch_type, epochs_list in container.items():
                total_count = sum(len(ep) for ep in epochs_list if isinstance(ep, mne.BaseEpochs))

                unique_subjects_in_cat_type = set()
                if total_count > 0:

                    for ep in epochs_list:
                        if isinstance(ep, mne.BaseEpochs) and ep.metadata is not None and 'subject_id' in ep.metadata.columns:
                            unique_subjects_in_cat_type.update(ep.metadata['subject_id'].unique())

                logger.info(f"  {epoch_type}: {total_count} epochs from {len(unique_subjects_in_cat_type)} unique subjects")

        output_path = os.path.join(BASE_PATH, "derivatives", 
                                   "ecg_preprocessed_categorized")
        os.makedirs(output_path, exist_ok=True)
        logger.info(f"Saving preprocessed data to: {output_path}")

        if args.category == 'all' or args.category == 'left_bte_crosstop':
            save_epochs_by_category(epochs_left_bte_crosstop, "left_bte_crosstop", output_path)
            del epochs_left_bte_crosstop
            gc.collect()

        if args.category == 'all' or args.category == 'right_bte_crosstop':
            save_epochs_by_category(epochs_right_bte_crosstop, "right_bte_crosstop", output_path)
            del epochs_right_bte_crosstop
            gc.collect()

        if args.category == 'all' or args.category == 'both_bte_no_crosstop':
            save_epochs_by_category(epochs_both_bte_no_crosstop, "both_bte_no_crosstop", output_path)
            del epochs_both_bte_no_crosstop
            gc.collect()

    logger.info("ECG preprocessing script finished successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Unhandled exception in main: {e}", exc_info=True)
        sys.exit(1)
