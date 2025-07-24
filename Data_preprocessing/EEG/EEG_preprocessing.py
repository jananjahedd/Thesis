"""
File: EEG_preprocessing.py
Author: Janan Jahed
Description: This file handles the preprocessing of seizure EEG data from
             the two BTE channels. It identifies EEG files with corresponding
             ECG data, creates sliding window epochs, labels them as seizure
             or non-seizure, and saves the processed data.
"""
import argparse
import gc
import glob
import logging
import os
import re
from joblib import Parallel, delayed

import mne
import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    print("tqdm not found. Progress bars will be basic.")

    def tqdm(iterable, *args, **kwargs):
        """Basic fallback for tqdm if not installed."""
        if hasattr(iterable, '__len__'):
            print(f"Processing {len(iterable)} items")
        else:
            print("Processing items")
        return iterable

parser = argparse.ArgumentParser(
    description="EEG preprocessing with sliding windows for paired ECG files."
)
parser.add_argument(
    '--base_path',
    required=True,
    help='Root folder for sub-*/ses-*/eeg data'
)
parser.add_argument(
    '--sfreq', type=int, default=256,
    help='Target sampling frequency'
)
parser.add_argument(
    '--n_jobs', type=int, default=-1,
    help='Number of parallel jobs (-1 for all cores, 1 for sequential)'
)
parser.add_argument(
    '--debug', action='store_true',
    help='Enable debug level logging'
)
parser.add_argument(
    '--max_epochs_per_batch', type=int, default=5000,
    help='Maximum number of epochs to process in a single memory batch'
)
parser.add_argument(
    '--process_by_subject', action='store_true',
    help='Process and save data into subject-specific folders'
)
parser.add_argument(
    '--max_non_seizure_per_file', type=int, default=75,
    help='Maximum non-seizure windows to keep per file for balancing'
)
parser.add_argument(
    '--seizure_buffer_sec', type=float, default=5.0,
    help='Buffer (in seconds) around seizures to exclude from non-seizure'
)
parser.add_argument(
    '--min_seizure_duration', type=float, default=30.0,
    help='Minimum duration (in seconds) for a seizure to be processed'
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
    """Find EEG files that have a corresponding ECG and events file

    Args:
        base_path (str): The root directory to search within

    Returns:
        list: A list of tuples, where each tuple contains the paths to the
              paired EEG, ECG, and events files
    """
    logger.info("Finding EEG files with corresponding ECG files.")
    eeg_files = glob.glob(os.path.join(base_path, "sub-*", "ses-*", "eeg",
                                       "*_eeg.edf"), recursive=True)
    ecg_files_set = set(glob.glob(os.path.join(base_path, "sub-*", "ses-*",
                                               "ecg", "*_ecg.edf"),
                                  recursive=True))

    paired_files = []
    base_ecg_names = {os.path.basename(f).replace('_ecg.edf', ''): f
                      for f in ecg_files_set}

    for eeg_path in eeg_files:
        base_eeg_name = os.path.basename(eeg_path).replace('_eeg.edf', '')
        if base_eeg_name in base_ecg_names:
            ecg_path = base_ecg_names[base_eeg_name]
            events_path = os.path.join(os.path.dirname(eeg_path),
                                       base_eeg_name + '_events.tsv')
            if os.path.exists(events_path):
                paired_files.append((eeg_path, ecg_path, events_path))
            else:
                logger.warning(f"Events file missing for {base_eeg_name}")

    logger.info(f"Found {len(paired_files)} EEG-ECG pairs with events files.")
    return paired_files


def extract_subject_id(filepath):
    """Extract the subject ID (e.g., 'sub-01') from a filepath

    Args:
        filepath (str): The path to the file

    Returns:
        str: The extracted subject ID or 'sub-unknown' if not found
    """
    match = re.search(r'sub-([a-zA-Z0-9]+)', filepath)
    return match.group(0) if match else "sub-unknown"


def extract_session_id(filepath):
    """Extract the session ID (e.g., 'ses-01') from a filepath.

    Args:
        filepath (str): The path to the file.

    Returns:
        str: The extracted session ID or 'ses-unknown' if not found.
    """
    match = re.search(r'ses-([a-zA-Z0-9]+)', filepath)
    return match.group(0) if match else "ses-unknown"


def create_sliding_window_epochs(raw, seizure_intervals, subject_id,
                                 session_id, window_size_sec,
                                 window_step_sec, max_non_seizure, buffer_sec):
    """Create labeled, balanced sliding window epochs from raw data

    Args:
        raw (mne.io.Raw): The MNE raw object
        seizure_intervals (list): List of (start, duration) tuples in samples
        subject_id (str): The subject identifier
        session_id (str): The session identifier
        window_size_sec (float): Window size in seconds
        window_step_sec (float): Window step in seconds
        max_non_seizure (int): Max non-seizure windows to keep
        buffer_sec (float): Buffer around seizures to exclude

    Returns:
        mne.Epochs: An MNE Epochs object containing the windows, or None
    """
    fname = os.path.basename(raw.filenames[0]) if raw.filenames else "UnkFile"
    sfreq = raw.info['sfreq']
    win_size_s = int(window_size_sec * sfreq)
    win_step_s = int(window_step_sec * sfreq)
    buffer_s = int(buffer_sec * sfreq)
    total_s = raw.n_times

    starts = np.arange(0, total_s - win_size_s + 1, win_step_s)
    if len(starts) == 0:
        logger.warning(f"No valid windows for {fname} (duration too short).")
        return None

    buffered_intervals = [(max(0, s - buffer_s),
                           min(total_s, s + d + buffer_s))
                          for s, d in seizure_intervals]

    labels, valid_starts = [], []
    for start_s in starts:
        end_s = start_s + win_size_s
        is_seizure = any(start_s < s + d and end_s > s for s,
                         d in seizure_intervals)
        if is_seizure:
            labels.append(1)
            valid_starts.append(start_s)
        else:
            in_buffer = any(start_s < buf_e and end_s > buf_s for buf_s, buf_e
                            in buffered_intervals)
            if not in_buffer:
                labels.append(0)
                valid_starts.append(start_s)

    if not valid_starts:
        logger.warning(f"No valid windows after filtering for {fname}.")
        return None

    # Balance non-seizure windows
    seizure_indices = [i for i, lbl in enumerate(labels) if lbl == 1]
    non_seizure_indices = [i for i, lbl in enumerate(labels) if lbl == 0]

    selected_indices = seizure_indices
    if len(non_seizure_indices) > max_non_seizure:
        step = len(non_seizure_indices) // max_non_seizure or 1
        selected_non_seizure = non_seizure_indices[::step][:max_non_seizure]
        selected_indices.extend(selected_non_seizure)
    else:
        selected_indices.extend(non_seizure_indices)
    selected_indices.sort()

    if not selected_indices:
        logger.warning(f"No windows selected after balancing for {fname}.")
        return None

    final_starts = [valid_starts[i] for i in selected_indices]
    final_labels = [labels[i] for i in selected_indices]
    events = np.array([[s, 0, l + 1] for s, l in zip(final_starts,
                                                     final_labels)])

    try:
        epochs = mne.Epochs(raw, events, tmin=0, tmax=window_size_sec,
                            baseline=None, preload=True, verbose=False,
                            picks='eeg')
        meta_list = [{
            'subject_id': subject_id,
            'session_id': session_id,
            'seizure_label': event[2] - 1,
            'unique_epoch_id': f"{subject_id}_{session_id}_s{event[0]}",
            'window_start_sample': event[0],
        } for event in epochs.events]
        epochs.metadata = pd.DataFrame(meta_list)
        n_sz = (epochs.metadata['seizure_label'] == 1).sum()
        n_nsz = (epochs.metadata['seizure_label'] == 0).sum()
        logger.info(f"Created {len(epochs)} windows for {fname}: "
                    f"{n_sz} seizure, {n_nsz} non-seizure.")
        return epochs
    except Exception as e:
        logger.error(f"Failed to create Epochs object for {fname}: {e}",
                     exc_info=True)
        return None


def fallback_rejection(epochs, file_id="N/A", threshold_uV=150, min_keep=10):
    """A simple artifact rejection based on amplitude and variance

    Args:
        epochs (mne.Epochs): epochs to clean
        file_id (str): for logging purposes
        threshold_uV (float): peak-to-peak amplitude threshold in microvolts
        min_keep (int): Min number of epochs to keep if possible

    Returns:
        mne.Epochs: The cleaned epochs.
    """
    if len(epochs) == 0:
        return epochs
    data = epochs.get_data()
    bad_amp = np.any(np.abs(data) > threshold_uV * 1e-6, axis=(1, 2))
    var = np.var(data, axis=(1, 2))
    m, s = np.mean(var), np.std(var)
    bad_var = (np.abs(var - m) > 3 * s) if s > 0 else np.zeros_like(var, bool)
    keep = ~(bad_amp | bad_var)

    if keep.sum() < min_keep and len(epochs) >= min_keep:
        logger.warning(f"Rejection for {file_id} too strict "
                       f"({keep.sum()}/{len(epochs)})."
                       f" Relaxing to keep {min_keep}.")
        idx_to_keep = np.argsort(np.abs(var - m))[:min_keep]
        keep = np.zeros(len(epochs), dtype=bool)
        keep[idx_to_keep] = True

    logger.info(f"Fallback rejection for {file_id}: "
                f"kept {keep.sum()}/{len(epochs)} epochs.")
    return epochs[keep]


def process_single_paired_file(file_tuple, target_sfreq, window_size_sec,
                               window_step_sec, max_non_seizure, buffer_sec,
                               min_duration_sec):
    """Process one EEG file: load, filter, and create windowed epochs

    Args:
        file_tuple (tuple): Paths to EEG, ECG, and events files
        target_sfreq (int): Target sampling frequency
        window_size_sec (float): Window size in seconds
        window_step_sec (float): Window step in seconds
        max_non_seizure (int): max non-seizure windows to keep
        buffer_sec (float): Buffer around seizures
        min_duration_sec (float): min seizure duration to consider

    Returns:
        tuple: (epochs, bte_category, subject_id, session_id, ecg_path)
               or (None, None, ...) on failure.
    """
    eeg_path, ecg_path, events_path = file_tuple
    fname = os.path.basename(eeg_path)

    try:
        events_df = pd.read_csv(events_path, sep='\t')
        sz_events = events_df[
            events_df['eventType'].str.startswith('sz_', na=False) &
            (pd.to_numeric(events_df['duration'],
                           'coerce') >= min_duration_sec)
        ]
    except Exception as e:
        logger.error(f"Could not parse events file {fname}: {e}")
        return None, None, None, None, None

    if sz_events.empty:
        logger.info(f"No valid seizures found in {fname}.")
        return None, None, None, None, None

    logger.info(f"Found {len(sz_events)} valid seizures in {fname}.")
    subject_id = extract_subject_id(eeg_path)
    session_id = extract_session_id(eeg_path)

    try:
        raw = mne.io.read_raw_edf(eeg_path, preload=True, verbose=False)
        if raw.n_times == 0:
            return None, None, subject_id, session_id, ecg_path

        ch_names = raw.ch_names
        bte_l, bte_r, cross_t = 'BTEleft SD', 'BTEright SD', 'CROSStop SD'
        if bte_l in ch_names and cross_t in ch_names:
            picks, cat = [bte_l, cross_t], 'left_bte_crosstop'
        elif bte_r in ch_names and cross_t in ch_names:
            picks, cat = [bte_r, cross_t], 'right_bte_crosstop'
        elif bte_l in ch_names and bte_r in ch_names:
            picks, cat = [bte_l, bte_r], 'both_bte_no_crosstop'
        else:
            logger.warning(f"Could not form channel pair for {fname}.")
            return None, None, subject_id, session_id, ecg_path

        raw.pick(picks)
        raw.filter(l_freq=0.5, h_freq=40., fir_design='firwin', verbose=False)
        raw.notch_filter(freqs=50, fir_design='firwin', verbose=False)
        if raw.info['sfreq'] != target_sfreq:
            raw.resample(target_sfreq, npad='auto', verbose=False)

        seizure_intervals = [
            (int(onset * raw.info['sfreq']), int(dur * raw.info['sfreq']))
            for onset, dur in zip(pd.to_numeric(sz_events['onset'], 'coerce'),
                                  pd.to_numeric(sz_events['duration'],
                                                'coerce'))
            if pd.notna(onset) and pd.notna(dur)
        ]

        epochs = create_sliding_window_epochs(
            raw, seizure_intervals, subject_id, session_id, window_size_sec,
            window_step_sec, max_non_seizure, buffer_sec
        )
        del raw
        gc.collect()
        return epochs, cat, subject_id, session_id, ecg_path

    except Exception as e:
        logger.error(f"Error processing {fname}: {e}", exc_info=True)
        return None, None, subject_id, session_id, ecg_path


def process_epoch_batch(epochs_batch, category, batch_id, output_dir):
    """Process and save a batch of epochs after artifact rejection

    Args:
        epochs_batch (list): List of MNE Epochs objects
        category (str): The category name for sub-folder creation
        batch_id (str): A unique identifier for the batch file
        output_dir (str): The root directory to save files
    """
    try:
        all_epochs = mne.concatenate_epochs(epochs_batch)
        seizure_epochs = all_epochs[all_epochs.metadata['seizure_label'] == 1]
        non_seizure_epochs = all_epochs[
            all_epochs.metadata['seizure_label'] == 0]

        logger.info(f"Applying rejection to {len(non_seizure_epochs)} "
                    f"non-seizure epochs for batch {batch_id}.")
        clean_non_seizure = fallback_rejection(
            non_seizure_epochs, file_id=f"{batch_id}_non_seizure")
        logger.info(f"Kept all {len(seizure_epochs)} seizure epochs for "
                    f"batch {batch_id} (no rejection).")

        final_list = [ep for ep in [seizure_epochs, clean_non_seizure]
                      if len(ep) > 0]
        if final_list:
            epochs_to_save = mne.concatenate_epochs(final_list)
        else:
            None

        if epochs_to_save and len(epochs_to_save) > 0:
            category_dir = os.path.join(output_dir, category)
            os.makedirs(category_dir, exist_ok=True)
            out_fname = os.path.join(category_dir,
                                     f"paired_{batch_id}-epo.fif")
            epochs_to_save.save(out_fname, overwrite=True, verbose=False)

            n_sz = (epochs_to_save.metadata['seizure_label'] == 1).sum()
            n_nsz = (epochs_to_save.metadata['seizure_label'] == 0).sum()
            n_subs = epochs_to_save.metadata['subject_id'].nunique()
            logger.info(f"Saved batch {batch_id}: {len(epochs_to_save)} "
                        f"windows ({n_sz} seizure, {n_nsz} non-seizure) "
                        f"from {n_subs} subjects.")
        else:
            logger.warning(f"No epochs left after rejection for {batch_id}.")
    except Exception as e:
        logger.error(f"Error processing batch {batch_id}: {e}", exc_info=True)
    finally:
        del epochs_batch, all_epochs, epochs_to_save
        gc.collect()


def process_epochs_in_batches(epochs_list, batch_size, category, output_dir):
    """Group epochs into batches and process them to manage memory

    Args:
        epochs_list (list): A list of MNE Epochs objects
        batch_size (int): The maximum number of epochs per batch
        category (str): The category name for the data
        output_dir (str): The root directory for saving
    """
    if not epochs_list:
        return

    batch_num, current_batch, current_count = 1, [], 0
    for epochs in epochs_list:
        if current_count + len(epochs) > batch_size and current_batch:
            process_epoch_batch(current_batch, category,
                                f"{category}_batch{batch_num}", output_dir)
            current_batch, current_count, batch_num = [], 0, batch_num + 1
            gc.collect()
        current_batch.append(epochs)
        current_count += len(epochs)
    if current_batch:
        process_epoch_batch(current_batch, category,
                            f"{category}_batch{batch_num}", output_dir)
    gc.collect()


def process_by_subject(results, output_path):
    """Organize and process results on a per-subject basis

    Args:
        results (list): The list of tuples from `process_single_paired_file`
        output_path (str): The root directory to save subject folders
    """
    logger.info("Aggregating and processing data by subject")
    subject_data = {}
    for epochs, cat, sub_id, _, _ in results:
        if epochs is None:
            continue
        if sub_id not in subject_data:
            subject_data[sub_id] = {
                'left_bte_crosstop': [], 'right_bte_crosstop': [],
                'both_bte_no_crosstop': []
            }
        if cat in subject_data[sub_id]:
            subject_data[sub_id][cat].append(epochs)

    for subject_id, categories in subject_data.items():
        logger.info(f"Processing subject: {subject_id}")
        subject_output_path = os.path.join(output_path, subject_id)
        os.makedirs(subject_output_path, exist_ok=True)
        for cat_name, epochs_list in categories.items():
            if epochs_list:
                process_epochs_in_batches(epochs_list,
                                          args.max_epochs_per_batch,
                                          cat_name, subject_output_path)
    gc.collect()


def main():
    """Main function to run the EEG preprocessing pipeline."""
    logger.info("Starting EEG preprocessing for paired files.")
    logger.info(f"Window size: {WINDOW_SIZE_SEC}s, Step: {WINDOW_STEP_SEC}s")
    logger.info(f"Seizure buffer: {BUFFER_SEC}s, "
                f"Min Duration: {MIN_SEIZURE_DURATION_SEC}s")

    paired_files = find_paired_files(BASE_PATH)
    if not paired_files:
        logger.error("No EEG-ECG pairs found.")
        return

    logger.info(f"Found {len(paired_files)} pairs to process using "
                f"n_jobs={args.n_jobs}.")

    results = Parallel(n_jobs=args.n_jobs)(
        delayed(process_single_paired_file)(
            file_tuple, SFREQ, WINDOW_SIZE_SEC, WINDOW_STEP_SEC,
            args.max_non_seizure_per_file, BUFFER_SEC,
            MIN_SEIZURE_DURATION_SEC
        ) for file_tuple in tqdm(paired_files, desc="Processing EEG files")
    )

    logger.info("File processing complete. Aggregating results.")
    valid_results = [r for r in results if r is not None and r[0] is not None]
    logger.info(f"Successfully processed {len(valid_results)}.")

    if args.process_by_subject:
        output_path = os.path.join(BASE_PATH, "derivatives",
                                   "eeg_paired_by_subject")
        os.makedirs(output_path, exist_ok=True)
        process_by_subject(valid_results, output_path)
    else:
        output_path = os.path.join(BASE_PATH, "derivatives",
                                   "eeg_paired_windows")
        os.makedirs(output_path, exist_ok=True)
        all_epochs_by_cat = {
            'left_bte_crosstop': [], 'right_bte_crosstop': [],
            'both_bte_no_crosstop': []
        }
        for epochs, cat, _, _, _ in valid_results:
            if cat in all_epochs_by_cat:
                all_epochs_by_cat[cat].append(epochs)
        for cat, epochs_list in all_epochs_by_cat.items():
            process_epochs_in_batches(epochs_list, args.max_epochs_per_batch,
                                      cat, output_path)

    logger.info("Preprocessing pipeline complete.")


if __name__ == "__main__":
    main()
