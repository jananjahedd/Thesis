"""
Filename: ECG_preprocessing.py
Author: Janan Jahed
Description: This files handles the preprocessing of the ECG single lead channel as well as aligning it with the EEG data
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

print("ECG preprocessing script started!")
print(f"Python executable: {os.sys.executable}")
print(f"Current working directory: {os.getcwd()}")

#debugging
try:
    from tqdm import tqdm
except ImportError:
    print("tqdm not found.")

    def tqdm(iterable, *args, **kwargs):
        if hasattr(iterable, '__len__'):
            print(f"Processing {len(iterable)} items...")
        else:
            print("Processing items...")
        return iterable

try:
    from autoreject import AutoReject
except ImportError:
    print("autoreject not found. Fallback rejection will be used.")
    AutoReject = None

#parser
parser = argparse.ArgumentParser(
    description="ECG preprocessing with sliding windows)"
)
parser.add_argument(
    '--base_path',
    required=True,
    help='Root folder for sub-*/ses-*/ecg data'
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
    help='Maximum non-seizure windows per file'
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

LOG_FILENAME = os.path.join(BASE_PATH, "ecg_preprocessing_paired_files.log")
logging.basicConfig(
    filename=LOG_FILENAME,
    level=logging.DEBUG if args.debug else logging.INFO,
    format="%(asctime)s - %(levelname)s - %(processName)s - %(message)s",
    filemode='w'
)
logger = logging.getLogger()


def find_paired_files(base_path):
    ecg_files = glob.glob(os.path.join(base_path, "sub-*", "ses-*", "ecg",
                                       "*_ecg.edf"), recursive=True)
    eeg_files_set = set(glob.glob(os.path.join(base_path, "sub-*", "ses-*",
                                               "eeg",
                                               "*_eeg.edf"), recursive=True))

    base_eeg_names = {os.path.basename(f).replace('_eeg.edf', ''): f for f in eeg_files_set}

    paired_files = []
    for ecg_path in ecg_files:
        base_ecg_name = os.path.basename(ecg_path).replace('_ecg.edf', '')

        if base_ecg_name in base_eeg_names:
            eeg_path = base_eeg_names[base_ecg_name]
            events_path = os.path.join(os.path.dirname(eeg_path),
                                       base_ecg_name + '_events.tsv')

            if os.path.exists(events_path):
                paired_files.append((eeg_path, ecg_path, events_path))

    logger.info(f"Found {len(paired_files)} ECG-EEG pairs with events files")
    return paired_files


def extract_subject_id(filepath):
    match = re.search(r'sub-([a-zA-Z0-9]+)', filepath)
    return match.group(0) if match else "sub-unknown"


def extract_session_id(filepath):
    match = re.search(r'ses-([a-zA-Z0-9]+)', filepath)
    return match.group(0) if match else "ses-unknown"


def create_sliding_window_epochs(raw, seizure_intervals_samples, subject_id,
                                 session_id,
                                 window_size_sec,
                                 window_step_sec,
                                 max_non_seizure_per_file,
                                 buffer_sec):
    fname = os.path.basename(raw.filenames[0]) if raw.filenames else "UnkFile"
    window_size_samples = int(window_size_sec * raw.info['sfreq'])
    window_step_samples = int(window_step_sec * raw.info['sfreq'])
    buffer_samples = int(buffer_sec * raw.info['sfreq'])
    total_samples = raw.n_times
    window_starts = np.arange(0, total_samples - window_size_samples + 1,
                              window_step_samples)

    if len(window_starts) == 0: return None

    buffered_seizure_intervals = [(max(0, s - buffer_samples),
                                   min(total_samples, s + d + buffer_samples)) for s, d in seizure_intervals_samples]

    seizure_labels = []
    valid_starts = []

    for start_sample in window_starts:
        end_sample = start_sample + window_size_samples
        is_seizure = any(start_sample < s + d and end_sample > s for s,
                         d in seizure_intervals_samples)

        if not is_seizure:
            in_buffer_zone = any(start_sample < buf_end and end_sample > buf_start for buf_start, buf_end in buffered_seizure_intervals)
            if not in_buffer_zone:
                seizure_labels.append(0)
                valid_starts.append(start_sample)
        else:
            seizure_labels.append(1)
            valid_starts.append(start_sample)

    if not valid_starts:
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

    if not selected_indices:
        return None

    selected_indices.sort()
    final_starts = [valid_starts[i] for i in selected_indices]
    final_labels = [seizure_labels[i] for i in selected_indices]

    events = np.array([[s, 0, l + 1] for s, l in zip(final_starts,
                                                     final_labels)])

    try:
        epochs = mne.Epochs(raw, events, tmin=0, tmax=window_size_sec,
                            baseline=None, preload=True, verbose=False,
                            picks='ecg')

        metadata_entries = [{
            'subject_id': subject_id, 'session_id': session_id,
            'seizure_label': event[2] - 1,
            'label_name': 'seizure' if event[2] - 1 == 1 else 'non_seizure',
            'unique_epoch_id': f"{subject_id}_{session_id}_s{event[0]}_w{int(window_size_sec)}",
            'window_start_sample': event[0],
            'window_start_time': event[0] / raw.info['sfreq']
        } for event in epochs.events]
        epochs.metadata = pd.DataFrame(metadata_entries)

        n_seizure = (epochs.metadata['seizure_label'] == 1).sum()
        n_non_seizure = (epochs.metadata['seizure_label'] == 0).sum()
        logger.info(f"Created {len(epochs)} ECG windows for {fname}: {n_seizure} seizure, {n_non_seizure} non-seizure")
        return epochs
    except Exception as e:
        logger.error(f"Error creating sliding window epochs for {fname}: {e}",
                     exc_info=True)
        return None


def fallback_rejection(epochs, file_identifier_for_log="N/A",
                       threshold_uV=2000, min_samples=10):
    if len(epochs) == 0:
        return epochs
    data = epochs.get_data()
    bad_amp = np.any(np.abs(data) > threshold_uV * 1e-6, axis=(1, 2))
    var = np.var(data, axis=(1, 2))
    m, s = np.mean(var), np.std(var)
    bad_var = (np.abs(var - m) > 3 * s) if s > 0 else np.zeros_like(var, dtype=bool)
    keep = ~(bad_amp | bad_var)

    if keep.sum() < min_samples and len(epochs) >= min_samples:
        idx_to_keep = np.argsort(np.abs(var - m))[:min_samples]
        keep = np.zeros(len(epochs), dtype=bool)
        keep[idx_to_keep] = True

    logger.info(f"Fallback rejection for {file_identifier_for_log}: kept {keep.sum()}/{len(epochs)} epochs.")
    return epochs[keep]


def process_single_paired_file(file_tuple, target_sfreq, window_size_sec,
                               window_step_sec,
                               max_non_seizure_per_file,
                               buffer_sec,
                               min_duration_sec):
    eeg_path, ecg_path, events_path = file_tuple
    original_basename = os.path.basename(ecg_path)

    try:
        events_df = pd.read_csv(events_path, sep='\t')
        seizure_events_in_file = events_df[
            events_df['eventType'].str.startswith('sz_', na=False) &
            (pd.to_numeric(events_df['duration'],
                           errors='coerce') >= min_duration_sec)
        ]
    except Exception as e:
        logger.error(f"Could not read or parse events file "
                     f"{original_basename}: {e}")
        return None, None

    if seizure_events_in_file.empty:
        return None, None

    logger.info(f"Found {len(seizure_events_in_file)} seizures >= "
                f"{min_duration_sec}s in events for "
                f"{original_basename}. Processing.")

    subject_id = extract_subject_id(ecg_path)
    session_id = extract_session_id(ecg_path)

    try:
        raw = mne.io.read_raw_edf(ecg_path, preload=True, verbose=False)
        if raw.n_times == 0: return None, subject_id

        raw.set_channel_types({ch: 'ecg' for ch in raw.ch_names})

        raw.filter(l_freq=0.5, h_freq=40., fir_design='firwin',
                   verbose=False, picks='ecg')
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
            return None, subject_id

        del raw
        gc.collect()
        return epochs, subject_id

    except Exception as e:
        logger.error(f"Error processing {original_basename}: {e}",
                     exc_info=True)
        return None, subject_id


def process_epoch_batch(epochs_batch, subject_id, batch_num, output_dir):
    batch_id = f"{subject_id}_batch{batch_num}"
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
            logger.info(f"Applying rejection to {len(non_seizure_epochs)}"
                        f" non-seizure epochs for batch {batch_id}...")
            clean_non_seizure_epochs = fallback_rejection(
                non_seizure_epochs,
                file_identifier_for_log=f"{batch_id}_non_seizure")
            logger.info(f"Kept all {len(seizure_epochs)} seizure epochs "
                        f"for batch {batch_id} without rejection.")

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
                    all_eps.get_data().shape[2])), all_eps.info, verbose=False)

        if len(eps_to_save) > 0:
            subject_dir = os.path.join(output_dir, subject_id)
            os.makedirs(subject_dir, exist_ok=True)
            out_fname = os.path.join(subject_dir,
                                     f"ecg_paired_windows_{batch_id}-epo.fif")
            eps_to_save.save(out_fname, overwrite=True, verbose=False)

            n_seizure = (eps_to_save.metadata['seizure_label'] == 1).sum()
            n_non_seizure = (eps_to_save.metadata['seizure_label'] == 0).sum()
            logger.info(f"Saved batch {batch_id}: {len(eps_to_save)} windows "
                        f"({n_seizure} seizure, {n_non_seizure} non-seizure)")

            meta_fname = os.path.join(subject_dir,
                                      f"ecg_paired_windows_{batch_id}_metadata.csv")
            eps_to_save.metadata.to_csv(meta_fname, index=False)
        else:
            logger.warning(f"No epochs left after rejection for batch "
                           f"{batch_id}")

    except Exception as e:
        logger.error(f"Error processing batch {batch_id}: {e}", exc_info=True)
    finally:
        del epochs_batch, all_eps, eps_to_save
        gc.collect()

def process_by_subject(results, output_path):
    logger.info("Aggregating and processing data by subject...")
    subject_data = {}
    for epochs, subject_id in results:
        if epochs is not None and subject_id is not None:
            if subject_id not in subject_data:
                subject_data[subject_id] = []
            subject_data[subject_id].append(epochs)

    for subject_id, epochs_list in subject_data.items():
        logger.info(f"Processing subject: {subject_id}")
        batch_num = 1
        current_batch = []
        current_count = 0
        for ep in epochs_list:
            if current_count + len(ep) > args.max_epochs_per_batch and current_batch:
                process_epoch_batch(current_batch, subject_id, batch_num,
                                    output_path)
                current_batch = []
                current_count = 0
                batch_num += 1
                gc.collect()
            current_batch.append(ep)
            current_count += len(ep)
        if current_batch:
            process_epoch_batch(current_batch, subject_id, batch_num,
                                output_path)
        logger.info(f"Finished processing subject: {subject_id}")
    gc.collect()


if __name__ == "__main__":
    logger.info("Starting ECG preprocessing with paired files")
    logger.info(f"Window size: {WINDOW_SIZE_SEC}s, Step: {WINDOW_STEP_SEC}s")
    logger.info(f"Max non-seizure per file: {args.max_non_seizure_per_file}")
    logger.info(f"Seizure buffer: {BUFFER_SEC}s, Minimum Seizure Duration: "
                f"{MIN_SEIZURE_DURATION_SEC}s")
    logger.info(f"Base path: {BASE_PATH}, Target sfreq: {SFREQ}")

    paired_files = find_paired_files(BASE_PATH)
    if not paired_files:
        logger.error("No ECG-EEG pairs found. Check --base_path.")
        exit()

    logger.info(f"Found {len(paired_files)} ECG-EEG pairs to process")
    logger.info(f"Using parallel processing (n_jobs={args.n_jobs})")

    results = Parallel(n_jobs=args.n_jobs)(
        delayed(process_single_paired_file)(
            file_tuple, SFREQ, WINDOW_SIZE_SEC, WINDOW_STEP_SEC,
            args.max_non_seizure_per_file, BUFFER_SEC, MIN_SEIZURE_DURATION_SEC
        ) for file_tuple in tqdm(paired_files,
                                 desc="Processing paired ECG files")
    )

    logger.info("File processing complete. Aggregating results.")
    valid_results = [r for r in results if r is not None and r[0] is not None]

    if args.process_by_subject:
        output_path = os.path.join(BASE_PATH, "derivatives",
                                   "ecg_paired_by_subject")
        os.makedirs(output_path, exist_ok=True)
        process_by_subject(valid_results, output_path)
    else:
        all_epochs = [r[0] for r in valid_results]
        output_path = os.path.join(BASE_PATH, "derivatives", "ecg_paired_all")
        os.makedirs(output_path, exist_ok=True)
        # process_epochs_in_batches(all_epochs, args.max_epochs_per_batch,
        # 'all_subjects', output_path)

    logger.info("Done")
    logger.info("ECG data is now ready")
