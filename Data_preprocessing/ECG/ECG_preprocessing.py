#!/usr/bin/env python3
import mne
import pandas as pd
import numpy as np
import os
import glob
import logging
import argparse
import gc
from joblib import Parallel, delayed


try:
    import neurokit2 as nk
except ImportError:
    print("NeuroKit2 not found. Please install it: pip install neurokit2")
    nk = None

try:
    from autoreject import AutoReject
except ImportError:
    print("autoreject not found. Please install it: pip install autoreject")
    AutoReject = None


parser = argparse.ArgumentParser(
    description="ECG preprocessing with epoch ID generation."
)
parser.add_argument(
    '--base_path',
    required=True,
    help='Root folder containing sub-*/ses-*/ecg and '
         'corresponding eeg/_events.tsv data'
)
parser.add_argument(
    '--sfreq',
    type=int,
    default=256,
    help='Target sampling frequency for ECG'
)
parser.add_argument(
    '--sample_limit_ns',
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
args = parser.parse_args()


BASE_PATH = args.base_path
SFREQ = args.sfreq
BUFFER_SEC = 5
EPOCH_SAMPLE_LIMIT_NON_SEIZURE = args.sample_limit_ns
N_JOBS = args.n_jobs

np.random.seed(42)

EPOCH_CONFIG = {
    "preictal": (-10, 0),
    "ictal": (0, 10),
    "onset": (-5, 5),
    "non_seizure": (0, 10),
}


LOG_FILENAME = os.path.join(BASE_PATH, "ecg_preprocessing_log.log")
logging.basicConfig(
    filename=LOG_FILENAME,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(processName)s - %(message)s",
    filemode='a'
)
logger = logging.getLogger()


def extract_ecg_features_from_epoch_data(
        epoch_data_segment, sampling_rate, epoch_id_str="N/A"):
    """
    Extracts ECG features from a single epoch's data segment.

    Args:
        epoch_data_segment (np.array): Single channel ECG data (n_times,).
        sampling_rate (int): The sampling rate of the ECG data.
        epoch_id_str (str): Identifier for logging purposes.

    Returns:
        dict | None: Dictionary of features or None if extraction fails.
    """
    if nk is None:
        logger.error("NeuroKit2 is not available. cant extract ECG features.")
        return None

    if epoch_data_segment.ndim > 1:
        epoch_data_segment = epoch_data_segment.squeeze()

    try:
        ecg_signals, info = nk.ecg_process(
            epoch_data_segment, sampling_rate=sampling_rate
        )
        if info is None or "ECG_R_Peaks" not in info or \
           not info["ECG_R_Peaks"]:
            logger.warning(
                f"Not enough R-peaks found for feature extraction in epoch "
                f"{epoch_id_str}."
            )
            return None
    except Exception as e:
        logger.warning(
            f"nk.ecg_process failed for epoch {epoch_id_str}: {e}"
        )
        return None

    rpeaks = np.asarray(info["ECG_R_Peaks"])
    if rpeaks.size < 2:
        logger.warning(
            f"Less than 2 R-peaks found ({rpeaks.size}) for epoch "
            f"{epoch_id_str}, cannot compute HRV."
        )
        try:
            hr = nk.ecg_rate(
                rpeaks,
                sampling_rate=sampling_rate,
                desired_length=len(epoch_data_segment)
            )
            return {
                "HR_Mean": np.mean(hr) if hr.size > 0 else np.nan,
                "HR_Std": np.std(hr) if hr.size > 0 else np.nan
            }
        except Exception as hr_e:
            logger.warning(f"HR extraction also failed for {epoch_id_str}: "
                           f"{hr_e}")
            return {"HR_Mean": np.nan, "HR_Std": np.nan}

    features = {}
    try:
        hr = nk.ecg_rate(
            rpeaks,
            sampling_rate=sampling_rate,
            desired_length=len(epoch_data_segment)
        )
        features["HR_Mean"] = np.mean(hr) if hr.size > 0 else np.nan
        features["HR_Std"] = np.std(hr) if hr.size > 0 else np.nan

        hrv_time = nk.hrv_time(rpeaks, sampling_rate=sampling_rate, show=False)
        for col in ["RMSSD", "MeanNN", "SDNN", "MedianNN", "pNN50"]:
            if f"HRV_{col}" in hrv_time.columns:
                features[col] = hrv_time[f"HRV_{col}"].iloc[0]
            elif col in hrv_time.columns:
                features[col] = hrv_time[col].iloc[0]
    except Exception as e:
        logger.warning(
            f"Time-domain HRV/HR extraction failed for epoch {epoch_id_str}: "
            f"{e}"
        )

    try:
        hrv_freq = nk.hrv_frequency(
            rpeaks, sampling_rate=sampling_rate, show=False
        )
        for col in ["LF", "HF", "LFHF"]:
            hrv_col_name = f"HRV_{col}"
            if hrv_col_name in hrv_freq.columns:
                features[col] = hrv_freq[hrv_col_name].iloc[0]
            elif col in hrv_freq.columns:
                features[col] = hrv_freq[col].iloc[0]
    except Exception as e:
        logger.warning(
            f"Frequency-domain HRV extraction failed for {epoch_id_str}: "
            f"{e}"
        )

    expected_features = [
        "HR_Mean", "HR_Std", "RMSSD", "MeanNN", "SDNN", "MedianNN", "pNN50",
        "LF", "HF", "LFHF"
    ]
    for feat in expected_features:
        if feat not in features:
            features[feat] = np.nan

    return features


def create_mne_epoch_with_metadata(
        raw_ecg, event_details_list, tmin, tmax, epoch_key):
    """
    Creates MNE epochs from a raw file based on a list of event details.
    Each event detail is a dict:
        {'onset_sec': float, 'id_suffix': str, 'orig_event_type': str}
    """
    if not event_details_list:
        return None

    events_array = []
    metadata_list = []

    fname_parts = os.path.basename(raw_ecg.filenames[0]).split('_')
    subject_id = "N/A"
    session_id = "N/A"
    run_id = "N/A"
    for part in fname_parts:
        if part.startswith("sub-"):
            subject_id = part
        elif part.startswith("ses-"):
            session_id = part
        elif part.startswith("run-"):
            run_id = part

    original_filename = os.path.basename(raw_ecg.filenames[0])

    for i, event_detail in enumerate(event_details_list):
        onset_sec = event_detail['onset_sec']
        event_sample = int(onset_sec * raw_ecg.info['sfreq'])

        start_samp_epoch = event_sample + int(tmin * raw_ecg.info['sfreq'])
        end_samp_epoch = event_sample + int(tmax * raw_ecg.info['sfreq'])
        if start_samp_epoch < 0 or end_samp_epoch > raw_ecg.n_times:
            logger.warning(
                f"Epoch for {epoch_key}_{event_detail['id_suffix']} in "
                f"{original_filename} out of bounds. Skipping."
            )
            continue

        events_array.append([event_sample, 0, i + 1])
        unique_epoch_id = (
            f"{subject_id}_{session_id}_{run_id}_"
            f"{epoch_key}_{event_detail['id_suffix']}"
        )
        metadata_list.append({
            'unique_epoch_id': unique_epoch_id,
            'original_filename': original_filename,
            'epoch_type_class': epoch_key,
            'original_event_type': event_detail.get(
                'orig_event_type', epoch_key
            ),
            'event_onset_sec': onset_sec,
            'tmin_applied': tmin,
            'tmax_applied': tmax
        })

    if not events_array:
        return None

    try:
        epochs = mne.Epochs(
            raw_ecg, np.array(events_array), tmin=tmin, tmax=tmax,
            baseline=None, preload=True, verbose=False, picks='ecg'
        )
        if len(epochs) != len(metadata_list):
            logger.error(
                f"Mismatch between created epochs ({len(epochs)}) and "
                f"metadata ({len(metadata_list)}) for {original_filename}, "
                f"key {epoch_key}."
            )
            if len(epochs) > 0 and len(epochs) < len(metadata_list):
                metadata_list = metadata_list[:len(epochs)]
            elif len(epochs) == 0:
                return None

        metadata_df = pd.DataFrame(metadata_list)
        epochs.metadata = metadata_df
        return epochs
    except Exception as e:
        logger.error(
            f"Error creating MNE epochs for {original_filename}, "
            f"key {epoch_key}: {e}"
        )
        return None


def process_non_seizure_segments(
        raw_ecg, seizure_intervals_samples, epoch_duration_sec,
        target_sfreq_local, max_epochs_per_file_local, buffer_sec_local):
    """
    Identifies non-seizure segments and prepares event details list for
    epoching.
    """
    non_seizure_mask = np.ones(raw_ecg.n_times, dtype=bool)
    safety_buffer_samples = int(buffer_sec_local * target_sfreq_local)

    for start_samp, end_samp in seizure_intervals_samples:
        buffer_start = max(0, start_samp - safety_buffer_samples)
        buffer_end = min(raw_ecg.n_times, end_samp + safety_buffer_samples)
        non_seizure_mask[buffer_start:buffer_end] = False

    epoch_length_samples = int(epoch_duration_sec * target_sfreq_local)
    if epoch_length_samples == 0:
        return []

    valid_starts_samples = np.where(
        np.convolve(
            non_seizure_mask,
            np.ones(epoch_length_samples, dtype=int),
            'valid'
        ) == epoch_length_samples
    )[0]

    if len(valid_starts_samples) == 0:
        return []

    selected_starts_details = []
    last_selected_end_sample = -1

    for start_samp_candidate in valid_starts_samples:
        if start_samp_candidate >= last_selected_end_sample:
            onset_sec = start_samp_candidate / target_sfreq_local
            selected_starts_details.append({
                'onset_sec': onset_sec,
                'id_suffix': (
                    f"segment{len(selected_starts_details):04d}_"
                    f"time{onset_sec:.2f}"
                ),
                'orig_event_type': 'non_seizure_segment'
            })
            last_selected_end_sample = (
                start_samp_candidate + epoch_length_samples
            )
            if max_epochs_per_file_local and \
               len(selected_starts_details) >= max_epochs_per_file_local:
                break
    return selected_starts_details


def find_events_file(ecg_file_path: str) -> str | None:
    """
    Finds the corresponding _events.tsv file for an _ecg.edf file.
    """
    session_path = os.path.dirname(os.path.dirname(ecg_file_path))
    eeg_folder_path = os.path.join(session_path, 'eeg')
    base_filename = os.path.basename(ecg_file_path).replace('_ecg.edf', '')
    expected_events_filename = f"{base_filename}_events.tsv"
    expected_events_path = os.path.join(
        eeg_folder_path, expected_events_filename
    )

    if os.path.exists(expected_events_path):
        return expected_events_path
    else:
        logger.warning(f"Expected events file not found at: "
                       f"{expected_events_path}")
        return None


def process_single_ecg_file(
        ecg_path, target_sfreq, epoch_config_ref,
        sample_limit_ns_ref, buffer_sec_ref):
    """
    Processes a single ECG file: loads, filters, resamples, extracts epochs
    with metadata.
    Returns a dictionary of MNE Epochs objects or None if processing fails.
    """
    original_basename = os.path.basename(ecg_path)
    logger.info(f"Processing ECG file: {original_basename}")

    epochs_from_this_file = {key: None for key in epoch_config_ref}

    try:
        raw = mne.io.read_raw_edf(ecg_path, preload=True, verbose=False)
        if raw.n_times == 0:
            logger.warning(
                f"Empty recording in {original_basename}. Skipping."
            )
            return None

        ecg_ch_names = [ch for ch in raw.ch_names if "ECG" in ch.upper()]
        if not ecg_ch_names:
            logger.error(f"No ECG found in {original_basename}. Skipping.")
            return None

        picked_ecg_channel = ecg_ch_names[0]
        logger.info(
            f"Using ECG channel: {picked_ecg_channel} for {original_basename}"
        )
        raw.pick(picks=[picked_ecg_channel])
        raw.set_channel_types({picked_ecg_channel: "ecg"})

        raw.filter(0.5, 40., fir_design='firwin', picks='ecg', verbose=False)
        raw.notch_filter(50, fir_design='firwin', picks='ecg', verbose=False)
        if raw.info["sfreq"] != target_sfreq:
            raw.resample(target_sfreq, npad='auto', verbose=False)

        events_tsv_path = find_events_file(ecg_path)
        if not events_tsv_path or not os.path.exists(events_tsv_path):
            logger.warning(
                f"No events file for {original_basename}. Cannot create "
                "seizure-related or targeted non-seizure epochs."
            )
            return None

        events_df = pd.read_csv(events_tsv_path, sep='\t')
        seizure_events_df = events_df[
            events_df['eventType'].str.startswith("sz_", na=False)
        ]

        seizure_intervals_samples = []
        for _, r_event in seizure_events_df.iterrows():
            try:
                onset_val = pd.to_numeric(r_event['onset'])
                duration_val = pd.to_numeric(r_event['duration'])
                if pd.isna(onset_val) or pd.isna(duration_val):
                    continue
                onset_samples = int(onset_val * raw.info['sfreq'])
                duration_samples = int(duration_val * raw.info['sfreq'])
                seizure_intervals_samples.append(
                    (onset_samples, onset_samples + duration_samples)
                )
            except ValueError:
                continue

        if not seizure_events_df.empty:
            for key_class in ("preictal", "ictal", "onset"):
                tmin, tmax = epoch_config_ref[key_class]
                event_details_for_class = []
                for idx, r_event in seizure_events_df.iterrows():
                    try:
                        onset_val = pd.to_numeric(r_event['onset'])
                        if pd.isna(onset_val):
                            continue
                        event_details_for_class.append({
                            'onset_sec': onset_val,
                            'id_suffix': (
                                f"event{idx}_time{onset_val:.2f}"
                            ),
                            'orig_event_type': r_event['eventType']
                        })
                    except ValueError:
                        continue

                if event_details_for_class:
                    epochs_for_class = create_mne_epoch_with_metadata(
                        raw, event_details_for_class,
                        tmin, tmax, key_class
                    )
                    if epochs_for_class is not None and \
                       len(epochs_for_class) > 0:
                        epochs_from_this_file[key_class] = epochs_for_class

        ns_epoch_duration_sec = (
            epoch_config_ref['non_seizure'][1] -
            epoch_config_ref['non_seizure'][0]
        )
        non_seizure_event_details = process_non_seizure_segments(
            raw, seizure_intervals_samples,
            ns_epoch_duration_sec, target_sfreq,
            sample_limit_ns_ref, buffer_sec_ref
        )
        if non_seizure_event_details:
            tmin_ns, tmax_ns = epoch_config_ref['non_seizure']
            ns_epochs = create_mne_epoch_with_metadata(
                raw, non_seizure_event_details,
                tmin_ns, tmax_ns, "non_seizure"
            )
            if ns_epochs is not None and len(ns_epochs) > 0:
                epochs_from_this_file["non_seizure"] = ns_epochs

        counts_str_parts = []
        for k, ep_obj in epochs_from_this_file.items():
            counts_str_parts.append(
                f"{k.capitalize()}: {len(ep_obj) if ep_obj else 0}"
            )
        logger.info(
            f"Finished {original_basename}. Epochs extracted - "
            f"{', '.join(counts_str_parts)}"
        )

        del raw
        gc.collect()
        return epochs_from_this_file

    except Exception as e:
        logger.error(
            f"Unhandled error in process_single_ecg_file for "
            f"{original_basename}: {e}", exc_info=True
        )
        if 'raw' in locals():
            del raw
        gc.collect()
        return None


def main():
    """
    Main function to orchestrate ECG preprocessing.
    """
    logger.info("==== ECG Preprocessing Script Started ====")

    ecg_files_to_process = glob.glob(
        os.path.join(BASE_PATH, "sub-*", "ses-*", "ecg", "*_ecg.edf")
    )
    if not ecg_files_to_process:
        logger.error(f"No ECG EDF files found under {BASE_PATH}. Exiting.")
        return

    logger.info(
        f"Found {len(ecg_files_to_process)} ECG files for parallel processing."
    )

    try:
        from tqdm import tqdm
        iterable_files = tqdm(ecg_files_to_process, desc="Processing ECG")
    except ImportError:
        logger.info("tqdm not found, processing without progress bar.")
        iterable_files = ecg_files_to_process

    results_from_parallel = Parallel(n_jobs=N_JOBS)(
        delayed(process_single_ecg_file)(
            f,
            SFREQ,
            EPOCH_CONFIG,
            EPOCH_SAMPLE_LIMIT_NON_SEIZURE,
            BUFFER_SEC
        ) for f in iterable_files
    )

    logger.info("Parallel processing complete. Aggregating results.")

    final_epochs_aggregated = {key: [] for key in EPOCH_CONFIG}
    for file_output_dict in results_from_parallel:
        if file_output_dict:
            for key, epochs_obj in file_output_dict.items():
                if epochs_obj is not None and len(epochs_obj) > 0:
                    final_epochs_aggregated[key].append(epochs_obj)

    logger.info("Aggregation complete. Starting final rejection and saving.")
    all_extracted_features_list = []

    for key_class, list_of_epochs_objects in \
            final_epochs_aggregated.items():
        if not list_of_epochs_objects:
            logger.info(
                f"No epochs to process for class: {key_class} "
                "after aggregation."
            )
            continue

        try:
            valid_epochs_to_cat = [
                ep for ep in list_of_epochs_objects
                if isinstance(ep, mne.BaseEpochs)
            ]
            if not valid_epochs_to_cat:
                logger.warning(
                    f"No valid MNE Epochs objects to concatenate for key: "
                    f"{key_class}"
                )
                continue

            epochs_all_for_class = mne.concatenate_epochs(
                valid_epochs_to_cat, verbose=False
            )
            logger.info(
                f"Aggregated {len(epochs_all_for_class)} epochs for class: "
                f"{key_class}"
            )

            epochs_clean = None
            if AutoReject is not None:
                try:
                    logger.info(f"Attempting AutoReject for {key_class}...")
                    ar = AutoReject(
                        n_interpolate=[0, 1, 2], random_state=42,
                        n_jobs=1, verbose=False
                    )
                    epochs_clean, _ = ar.fit_transform(
                        epochs_all_for_class, return_log=True
                    )
                    logger.info(
                        f"AutoReject kept {len(epochs_clean)} / "
                        f"{len(epochs_all_for_class)} for {key_class}"
                    )
                except Exception as ar_exception:
                    logger.warning(
                        f"AutoReject failed for {key_class}: {ar_exception}. "
                        "Using fallback rejection."
                    )
                    epochs_clean = None

            if epochs_clean is None:
                min_s = 20 if key_class != "non_seizure" else 100
                epochs_clean = fallback_ecg_rejection(
                    epochs_all_for_class, min_samples=min_s
                )

            if len(epochs_clean) == 0:
                logger.warning(
                    f"No epochs remained for {key_class} after rejection."
                )
                continue

            for i in range(len(epochs_clean)):
                epoch_segment_data = epochs_clean[i].get_data().squeeze()
                current_epoch_id_for_log = "N/A_in_feature_extraction"
                current_meta_row = {}
                if epochs_clean.metadata is not None and \
                   i < len(epochs_clean.metadata):
                    current_meta_row = epochs_clean.metadata.iloc[i].to_dict()
                    current_epoch_id_for_log = current_meta_row.get(
                        'unique_epoch_id', current_epoch_id_for_log
                    )

                features = extract_ecg_features_from_epoch_data(
                    epoch_segment_data, epochs_clean.info['sfreq'],
                    current_epoch_id_for_log
                )

                combined_feature_row = {**current_meta_row}
                if features:
                    combined_feature_row.update(features)
                all_extracted_features_list.append(combined_feature_row)

                if features and epochs_clean.metadata is not None:
                    for feat_key, feat_val in features.items():
                        epochs_clean.metadata.loc[
                            epochs_clean.metadata.index[i], feat_key
                        ] = feat_val

            output_fif_fname = os.path.join(
                BASE_PATH, f"{key_class}_ecg-clean-epo.fif"
            )
            epochs_clean.save(output_fif_fname, overwrite=True, verbose=False)
            logger.info(
                f"Saved {len(epochs_clean)} clean ECG epochs for {key_class} "
                f"to: {output_fif_fname}"
            )

        except Exception as e:
            logger.error(
                f"Error during final processing/saving for ECG class "
                f"{key_class}: {e}", exc_info=True
            )

    if all_extracted_features_list:
        features_df = pd.DataFrame(all_extracted_features_list)
        features_csv_fname = os.path.join(
            BASE_PATH, "all_classes_ecg-features_detailed.csv"
        )
        features_df.to_csv(features_csv_fname, index=False)
        logger.info(
            f"Saved detailed ECG features for {len(features_df)} epochs "
            f"across all classes to: {features_csv_fname}"
        )

    logger.info("==== ECG Preprocessing Script Finished ====")


def fallback_ecg_rejection(
        epochs: mne.Epochs, threshold_uV: int = 500, *,
        min_samples: int | None = None) -> mne.Epochs:
    """Basic fallback rejection for ECG epochs."""
    data = epochs.get_data()
    bad_amp = (np.abs(data) > threshold_uV * 1e-6).any(axis=(1, 2))

    var = data.var(axis=(1, 2))
    m, s = var.mean(), var.std()
    if s == 0:
        bad_var = np.zeros(len(var), dtype=bool)
    else:
        bad_var = np.abs(var - m) > 3 * s

    bad = bad_amp | bad_var
    keep = ~bad

    num_original = len(epochs)
    num_kept = keep.sum()

    if min_samples and num_kept < min_samples and num_original >= min_samples:
        logger.warning(
            f"Fallback ECG rejection initially kept {num_kept}/{num_original} "
            f"epochs. Attempting to keep at least {min_samples}."
        )
        if s == 0:
            idx_to_keep = np.arange(min(min_samples, num_original))
        else:
            idx_sorted_by_variance_diff = np.argsort(np.abs(var - m))
            idx_to_keep = idx_sorted_by_variance_diff[:min_samples]

        keep = np.zeros(num_original, dtype=bool)
        keep[idx_to_keep] = True
        num_kept = keep.sum()

    logger.info(f"Fallback rejection: kept {num_kept}/{num_original} epochs.")
    return epochs[keep]


if __name__ == "__main__":
    main()
