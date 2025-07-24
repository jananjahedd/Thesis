"""
File: EEG_preprocessing.py
Author:Janan Jahed
Description: This file handles the preprocessing of the seizure EEG data, for the two BTE channels 
"""
import mne
import pandas as pd
import numpy as np
import os
import glob
import logging
import gc
import re
import argparse
from joblib import Parallel, delayed


#debugging
try:
    from tqdm import tqdm
except ImportError:
    print("tqdm not found.")
    def tqdm(iterable, *args, **kwargs):
        if hasattr(iterable, '__len__'):
            print(f"Processing {len(iterable)} items")
        else:
            print("Processing items")
        return iterable

try:
    from autoreject import AutoReject
except ImportError:
    print("autoreject not found. Fallback rejection will be used.")
    AutoReject = None

#parsers
parser = argparse.ArgumentParser(
    description="EEG preprocessing with sliding windows - only files with corresponding ECG (SeizeIT2 optimized)"
)
parser.add_argument(
    '--base_path',
    required=True,
    help='Root folder for sub-*ses-*/eeg data'
)
parser.add_argument(
    '--sfreq', type=int, default=256, help='Target sampling frequency'
)
parser.add_argument(
    '--n_jobs', type=int, default=-1,
    help='Number of parallel jobs (-1 for all cores, 1 for sequential)'
)
parser.add_argument(
    '--debug', action='store_true',
    help='Enable debug logging'
)
parser.add_argument(
    '--max_epochs_per_batch', type=int, default=5000,
    help='Maximum number of epochs to process in a single batch'
)
parser.add_argument(
    '--process_by_subject', action='store_true',
    help='Process and save data by subject instead of all at once'
)
parser.add_argument(
    '--max_non_seizure_per_file', type=int, default=75,
    help='Maximum non-seizure windows per file (conservative for 795:1 imbalance)'
)
parser.add_argument(
    '--seizure_buffer_sec', type=float, default=5.0,
    help='Buffer around seizures to avoid in non-seizure windows'
)
parser.add_argument(
    '--min_seizure_duration', type=float, default=30.0,
    help='Minimum duration in seconds for a seizure to be processed.'
)

args = parser.parse_args()

BASE_PATH = args.base_path
SFREQ = args.sfreq
BUFFER_SEC = args.seizure_buffer_sec
MIN_SEIZURE_DURATION_SEC = args.min_seizure_duration

WINDOW_SIZE_SEC = 60.0
WINDOW_OVERLAP_SEC = 10.0
WINDOW_STEP_SEC = WINDOW_SIZE_SEC - WINDOW_OVERLAP_SEC

LOG_FILENAME = os.path.join(BASE_PATH, "eeg_preprocessing_paired_files.log")
logging.basicConfig(
    filename=LOG_FILENAME,
    level=logging.DEBUG if args.debug else logging.INFO,
    format="%(asctime)s - %(levelname)s - %(processName)s - %(message)s",
    filemode='w'
)
logger = logging.getLogger()


def find_paired_files(base_path):
    logger.info("Finding EEG files with corresponding ECG files.")
    eeg_files = glob.glob(os.path.join(base_path, "sub-*", "ses-*", "eeg",
                                       "*_eeg.edf"), recursive=True)
    ecg_files_set = set(glob.glob(os.path.join(base_path, "sub-*", "ses-*",
                                               "ecg", "*_ecg.edf"), recursive=True))

    paired_files = []
    base_ecg_names = {os.path.basename(f).replace('_ecg.edf', ''): f for f in ecg_files_set}

    for eeg_path in eeg_files:
        base_eeg_name = os.path.basename(eeg_path).replace('_eeg.edf', '')

        if base_eeg_name in base_ecg_names:
            ecg_path = base_ecg_names[base_eeg_name]
            events_path = os.path.join(os.path.dirname(eeg_path),
                                       base_eeg_name + '_events.tsv')

            if os.path.exists(events_path):
                paired_files.append((eeg_path, ecg_path, events_path))
            else:
                logger.warning(f"Events file missing for {base_eeg_name}: {events_path}")

    logger.info(f"Found {len(paired_files)} EEG-ECG pairs with events files")
    return paired_files


def extract_subject_id(filepath):
    match = re.search(r'sub-([a-zA-Z0-9]+)', filepath)
    return match.group(0) if match else "sub-unknown"


def extract_session_id(filepath):
    match = re.search(r'ses-([a-zA-Z0-9]+)', filepath)
    return match.group(0) if match else "ses-unknown"


def create_sliding_window_epochs(raw, seizure_intervals_samples, subject_id,
                                 session_id,
                                 window_size_sec, window_step_sec,
                                 max_non_seizure_per_file,
                                 buffer_sec):
    fname = os.path.basename(raw.filenames[0]) if raw.filenames else "UnkFile"
    window_size_samples = int(window_size_sec * raw.info['sfreq'])
    window_step_samples = int(window_step_sec * raw.info['sfreq'])
    buffer_samples = int(buffer_sec * raw.info['sfreq'])
    total_samples = raw.n_times
    window_starts = np.arange(0, total_samples - window_size_samples + 1,
                              window_step_samples)

    if len(window_starts) == 0:
        logger.warning(f"No valid windows for {fname} (duration too short)")
        return None

    for s, d in seizure_intervals_samples]:
        buffered_seizure_intervals = [(max(0, s - buffer_samples), min(total_samples, s + d + buffer_samples))

    seizure_labels = []
    valid_starts = []

    for start_sample in window_starts:
        end_sample = start_sample + window_size_samples
        is_seizure = any(start_sample < s + d and end_sample > s for s, d in seizure_intervals_samples)

        if not is_seizure:
            in_buffer_zone = any(start_sample < buf_end and end_sample > buf_start for buf_start, buf_end in buffered_seizure_intervals)
            if not in_buffer_zone:
                seizure_labels.append(0)
                valid_starts.append(start_sample)
        else:
            seizure_labels.append(1)
            valid_starts.append(start_sample)

    if not valid_starts:
        logger.warning(f"No valid windows after filtering for {fname}")
        return None

    seizure_indices = [i for i, label in enumerate(seizure_labels) if label == 1]
    non_seizure_indices = [i for i, label in enumerate(seizure_labels) if label == 0]

    selected_indices = seizure_indices
    if len(non_seizure_indices) > max_non_seizure_per_file:
        step = len(non_seizure_indices) // max_non_seizure_per_file or 1
        selected_non_seizure_indices = non_seizure_indices[::step][:max_non_seizure_per_file]
        selected_indices.extend(selected_non_seizure_indices)
    else:
        selected_indices.extend(non_seizure_indices)

    selected_indices.sort()

    if not selected_indices:
        logger.warning(f"No windows selected after balancing for {fname}")
        return None

    final_starts = [valid_starts[i] for i in selected_indices]
    final_labels = [seizure_labels[i] for i in selected_indices]

    events = np.array([[s, 0, l + 1] for s, l in zip(final_starts,
                                                     final_labels)])

    try:
        epochs = mne.Epochs(raw, events, tmin=0, tmax=window_size_sec,
                            baseline=None, preload=True,
                            verbose=False, picks='eeg')
        metadata_entries = [{
            'subject_id': subject_id, 'session_id': session_id,
            'seizure_label': event[2] - 1,
            'label_name': 'seizure' if event[2] - 1 == 1 else 'non_seizure',
            'unique_epoch_id': f"{subject_id}_{session_id}_s{event[0]}_w{int(window_size_sec)}",
            'window_start_sample': event[0], 'window_start_time': event[0] / raw.info['sfreq']
        } for event in epochs.events]
        epochs.metadata = pd.DataFrame(metadata_entries)

        n_seizure = (epochs.metadata['seizure_label'] == 1).sum()
        n_non_seizure = (epochs.metadata['seizure_label'] == 0).sum()
        logger.info(f"Created {len(epochs)} windows for {fname}: {n_seizure} "
                    f"seizure, {n_non_seizure} non-seizure")
        return epochs
    except Exception as e:
        logger.error(f"Can't create sliding window epochs for {fname}: {e}",
                     exc_info=True)
        return None


def fallback_rejection(epochs, file_identifier_for_log="N/A",
                       threshold_uV=150,
                       min_samples=10):
    if len(epochs) == 0:
        return epochs
    data = epochs.get_data()
    bad_amp = np.any(np.abs(data) > threshold_uV * 1e-6, axis=(1, 2))
    var = np.var(data, axis=(1, 2))
    m, s = np.mean(var), np.std(var)
    bad_var = (np.abs(var - m) > 3 * s) if s > 0 else np.zeros_like(var,
                                                                    dtype=bool)
    keep = ~(bad_amp | bad_var)

    if keep.sum() < min_samples and len(epochs) >= min_samples:
        logger.warning(f"Fallback rejection for {file_identifier_for_log} kept "
                       f"{keep.sum()}/{len(epochs)}. "
                       f"Trying to keep {min_samples}.")
        idx_to_keep = np.argsort(np.abs(var - m))[:min_samples]
        keep = np.zeros(len(epochs), dtype=bool)
        keep[idx_to_keep] = True

    logger.info(f"Fallback rejection for {file_identifier_for_log}: "
                f"kept {keep.sum()}/{len(epochs)} epochs.")
    return epochs[keep]


def process_single_paired_file(file_tuple, target_sfreq, window_size_sec,
                               window_step_sec,
                               max_non_seizure_per_file,
                               buffer_sec,
                               min_duration_sec):
    eeg_path, ecg_path, events_path = file_tuple
    original_basename = os.path.basename(eeg_path)

    try:
        events_df = pd.read_csv(events_path, sep='\t')

        seizure_events_in_file = events_df[
            events_df['eventType'].str.startswith('sz_', na=False) &
            (pd.to_numeric(events_df['duration'],
                           errors='coerce') >= min_duration_sec)
        ]
    except Exception as e:
        logger.error(f"Could not parse events file {original_basename}: "
                     f"{e}")
        return None, None, None, None, None

    if seizure_events_in_file.empty:
        logger.info(f"No seizures >= {min_duration_sec}s found in "
                    f"{original_basename}")
        return None, None, None, None, None

    logger.info(f"Found {len(seizure_events_in_file)} seizures >= "
                f"{min_duration_sec}s in {original_basename}.")

    subject_id = extract_subject_id(eeg_path)
    session_id = extract_session_id(eeg_path)

    try:
        raw = mne.io.read_raw_edf(eeg_path, preload=True, verbose=False)
        if raw.n_times == 0:
            return None, None, subject_id, session_id, ecg_path

        ch_names = raw.ch_names
        bte_left_ch, bte_right_ch, crosstop_ch = 'BTEleft SD', 'BTEright SD',
        'CROSStop SD'
        if bte_left_ch in ch_names and crosstop_ch in ch_names:
            picked_chs, bte_category = [bte_left_ch, crosstop_ch],
            'left_bte_crosstop'
        elif bte_right_ch in ch_names and crosstop_ch in ch_names:
            picked_chs, bte_category = [bte_right_ch, crosstop_ch],
            'right_bte_crosstop'
        elif bte_left_ch in ch_names and bte_right_ch in ch_names:
            picked_chs, bte_category = [bte_left_ch, bte_right_ch],
            'both_bte_no_crosstop'
        else:
            logger.warning(f"Could not form  pair for {original_basename}.")
            return None, None, subject_id, session_id, ecg_path

        raw.pick(picked_chs)
        raw.filter(l_freq=0.5, h_freq=40., fir_design='firwin', verbose=False)
        raw.notch_filter(freqs=50, fir_design='firwin', verbose=False)
        if raw.info['sfreq'] != target_sfreq:
            raw.resample(target_sfreq, npad='auto', verbose=False)

        seizure_intervals_samples = []
        for _, row in seizure_events_in_file.iterrows():
            onset = pd.to_numeric(row['onset'], errors='coerce')
            duration = pd.to_numeric(row['duration'], errors='coerce')
            if pd.notna(onset) and pd.notna(duration):
                seizure_intervals_samples.append(
                    (int(onset * raw.info['sfreq']),
                     int(duration * raw.info['sfreq']))
                )

        epochs = create_sliding_window_epochs(
            raw, seizure_intervals_samples, subject_id, session_id,
            window_size_sec, window_step_sec, max_non_seizure_per_file,
            buffer_sec
        )

        if epochs is None:
            return None, bte_category, subject_id, session_id, ecg_path

        del raw
        gc.collect()
        return epochs, bte_category, subject_id, session_id, ecg_path

    except Exception as e:
        logger.error(f"Error processing {original_basename}: {e}",
                     exc_info=True)
        return None, None, subject_id, session_id, ecg_path


def process_epoch_batch(epochs_batch, category_suffix, batch_id, output_dir):
    try:
        all_eps = mne.concatenate_epochs(epochs_batch)
        if all_eps.metadata is None or 'seizure_label' not in all_eps.metadata.columns:
            logger.warning(f"Missing metadata/label for batch {batch_id}.")
            eps_to_save = fallback_rejection(all_eps,
                                             file_identifier_for_log=batch_id)
        else:
            seizure_epochs = all_eps[all_eps.metadata['seizure_label'] == 1]
            non_seizure_epochs = all_eps[
                all_eps.metadata['seizure_label'] == 0]
            logger.info(f"Applying rejection to {len(non_seizure_epochs)} "
                        f"non-seizure epochs for batch {batch_id}.")
            clean_non_seizure_epochs = fallback_rejection(
                non_seizure_epochs,
                file_identifier_for_log=f"{batch_id}_non_seizure")
            logger.info(f"Kept all {len(seizure_epochs)} seizure epochs for "
                        f"batch {batch_id} without rejection.")

            final_epoch_list = []
            if len(seizure_epochs) > 0:
                final_epoch_list.append(seizure_epochs)
            if len(clean_non_seizure_epochs) > 0:
                final_epoch_list.append(clean_non_seizure_epochs)

            if final_epoch_list:
                eps_to_save = mne.concatenate_epochs(final_epoch_list)
            else:
                eps_to_save = mne.EpochsArray(np.empty((
                    0, all_eps.get_data().shape[1],
                    all_eps.get_data().shape[2])),
                    all_eps.info, verbose=False)

        if len(eps_to_save) > 0:
            category_dir = os.path.join(output_dir, category_suffix)
            os.makedirs(category_dir, exist_ok=True)
            out_fname = os.path.join(category_dir, f"paired_windows_{batch_id}-epo.fif")
            eps_to_save.save(out_fname, overwrite=True, verbose=False)

            n_seizure = (eps_to_save.metadata['seizure_label'] == 1).sum()
            n_non_seizure = (eps_to_save.metadata['seizure_label'] == 0).sum()
            n_subjects = eps_to_save.metadata['subject_id'].nunique()
            logger.info(f"Saved batch {batch_id}: {len(eps_to_save)} windows "
                        f"({n_seizure} seizure, {n_non_seizure} non-seizure) "
                        f"from {n_subjects} subjects")

            meta_fname = os.path.join(category_dir, f"paired_windows_{batch_id}_metadata.csv")
            eps_to_save.metadata.to_csv(meta_fname, index=False)
        else:
            logger.warning(f"No epochs left after rejection {batch_id}")

    except Exception as e:
        logger.error(f"Error processing batch {batch_id}: {e}", exc_info=True)
    finally:
        del epochs_batch, all_eps, eps_to_save
        gc.collect()


def process_epochs_in_batches(epochs_list, batch_size, category_suffix,
                              output_dir):
    if not epochs_list:
        return
    total_epochs = sum(len(ep) for ep in epochs_list if isinstance(
        ep, mne.BaseEpochs))
    if total_epochs == 0:
        return

    batch_num = 1
    current_batch = []
    current_count = 0
    for ep in epochs_list:
        if current_count + len(ep) > batch_size and current_batch:
            process_epoch_batch(current_batch, category_suffix,
                                f"{category_suffix}_batch{batch_num}",
                                output_dir)
            current_batch = []
            current_count = 0
            batch_num += 1
            gc.collect()
        current_batch.append(ep)
        current_count += len(ep)
    if current_batch:
        process_epoch_batch(current_batch, category_suffix,
                            f"{category_suffix}_batch{batch_num}", output_dir)
    gc.collect()


def process_by_subject(results, output_path):
    logger.info("Processing data by subject...")
    subject_data = {}
    for result_tuple in results:
        if result_tuple and result_tuple[0] is not None:
            epochs, bte_cat, subject_id, _, _ = result_tuple
            if subject_id not in subject_data:
                subject_data[subject_id] = {
                    'left_bte_crosstop': [], 'right_bte_crosstop': [],
                    'both_bte_no_crosstop': []
                }
            if bte_cat in subject_data[subject_id]:
                subject_data[subject_id][bte_cat].append(epochs)

    for subject_id, categories in subject_data.items():
        logger.info(f"Processing subject: {subject_id}")
        subject_output_path = os.path.join(output_path, subject_id)
        os.makedirs(subject_output_path, exist_ok=True)
        for cat_name, epochs_list in categories.items():
            if epochs_list:
                process_epochs_in_batches(epochs_list,
                                          args.max_epochs_per_batch,
                                          cat_name, subject_output_path)
        logger.info(f"Finished processing subject: {subject_id}")
    gc.collect()


if __name__ == "__main__":
    logger.info("Starting EEG preprocessing with paired ECG files")
    logger.info(f"Window size: {WINDOW_SIZE_SEC}s, Step: {WINDOW_STEP_SEC}s")
    logger.info(f"Max non-seizure per file: {args.max_non_seizure_per_file}")
    logger.info(f"Seizure buffer: {BUFFER_SEC}s, "
                f"Min Duration: {MIN_SEIZURE_DURATION_SEC}s")
    logger.info(f"Base path: {BASE_PATH}, Target sfreq: {SFREQ}")

    paired_files = find_paired_files(BASE_PATH)
    if not paired_files:
        logger.error("No EEG-ECG pairs found. Check --base_path.")
        exit()

    logger.info(f"Found {len(paired_files)} EEG-ECG pairs to process")
    logger.info(f"Using parallel processing (n_jobs={args.n_jobs})")

    results = Parallel(n_jobs=args.n_jobs)(
        delayed(process_single_paired_file)(
            file_tuple, SFREQ, WINDOW_SIZE_SEC, WINDOW_STEP_SEC,
            args.max_non_seizure_per_file, BUFFER_SEC, MIN_SEIZURE_DURATION_SEC
        ) for file_tuple in tqdm(paired_files, desc="Processing paired EEG")
    )

    logger.info("File processing complete. Aggregating results.")

    valid_results = [r for r in results if r is not None and r[0] is not None]

    if args.process_by_subject:
        output_path = os.path.join(BASE_PATH, "derivatives",
                                   "eeg_paired_by_subject")
        os.makedirs(output_path, exist_ok=True)
        process_by_subject(valid_results, output_path)
    else:

        output_path = os.path.join(BASE_PATH, "derivatives",
                                   "eeg_paired_windows")
        os.makedirs(output_path, exist_ok=True)
        all_epochs_by_cat = {'left_bte_crosstop': [], 'right_bte_crosstop': [],
                             'both_bte_no_crosstop': []}
        for epochs, cat, _, _, _ in valid_results:
            if cat in all_epochs_by_cat:
                all_epochs_by_cat[cat].append(epochs)
        for cat, epochs_list in all_epochs_by_cat.items():
            process_epochs_in_batches(epochs_list, args.max_epochs_per_batch,
                                      cat, output_path)

    logger.info("Complete")
