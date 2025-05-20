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
from joblib import Parallel, delayed

try:
    from tqdm import tqdm
except ImportError:
    print("tqdm not found")

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

parser = argparse.ArgumentParser(
    description="EEG preprocessing: separates outputs by category."
)
parser.add_argument(
    '--base_path',
    required=True,
    help='Root folder for sub-*/ses-*/eeg data'
)
parser.add_argument(
    '--sfreq', type=int, default=256, help='Target sampling frequency'
)
parser.add_argument(
    '--sample_limit', type=int, default=500,
    help='Max non-seizure epochs per EDF file'
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
args = parser.parse_args()

BASE_PATH = args.base_path
SFREQ = args.sfreq
BUFFER_SEC = 5
EPOCH_CONFIG = {
    'preictal': (-10, 0), 'ictal': (0, 10),
    'onset': (-5, 5), 'non_seizure': (0, 10)
}

LOG_FILENAME = os.path.join(BASE_PATH, "eeg_preprocessing_categorized.log")
logging.basicConfig(
    filename=LOG_FILENAME,
    level=logging.DEBUG if args.debug else logging.INFO,
    format="%(asctime)s - %(levelname)s - %(processName)s - %(message)s",
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


def fallback_rejection(epochs, file_identifier_for_log="N/A",
                       threshold_uV=150, min_samples=None):
    """Rejects epochs based on peak-to-peak amplitude and variance."""
    data = epochs.get_data()
    bad_amp = np.any(np.abs(data) > threshold_uV, axis=(1, 2))
    var = np.var(data, axis=(1, 2))
    m, s = np.mean(var), np.std(var)
    bad_var = np.zeros(len(var), dtype=bool) if s == 0 \
        else (np.abs(var - m) > 3 * s)
    bad = bad_amp | bad_var
    keep = ~bad
    num_original, num_kept = len(epochs), keep.sum()

    if min_samples and num_kept < min_samples and num_original >= min_samples:
        logger.warning(
            f"Fallback rejection for {file_identifier_for_log} initially "
            f"kept {num_kept}/{num_original}. Trying to keep {min_samples}."
        )
        idx_to_keep = np.argsort(np.abs(var - m))[:min_samples] if s != 0 \
            else np.arange(min(min_samples, num_original))
        keep = np.zeros(num_original, dtype=bool)
        keep[idx_to_keep] = True
        num_kept = keep.sum()
    logger.info(
        f"Fallback rejection for {file_identifier_for_log}: "
        f"kept {num_kept}/{num_original} epochs."
    )
    return epochs[keep]


def create_epoch_for_file(raw, event_onset_time, tmin, tmax, epoch_key_local,
                          subject_id, session_id):
    """Creates a single MNE epoch with subject ID in metadata."""
    fname = os.path.basename(raw.filenames[0]) if raw.filenames else "UnkFile"
    start_samp = int((event_onset_time + tmin) * raw.info['sfreq'])
    end_samp = int((event_onset_time + tmax) * raw.info['sfreq'])

    if start_samp < 0 or end_samp > raw.n_times:
        logger.warning(
            f"Epoch {epoch_key_local} @{event_onset_time:.2f}s in {fname} OOB."
            f"Need [{start_samp/raw.info['sfreq']:.2f}s, "
            f"{end_samp/raw.info['sfreq']:.2f}s] vs "
            f"Raw dur: {raw.times[-1]:.2f}s. Skipping."
        )
        return None
    try:

        event_type_id = {'preictal': 1, 'ictal': 2, 'onset': 3,
                         'non_seizure': 4}.get(epoch_key_local, 1)
        event_arr = np.array([[int(event_onset_time * raw.info['sfreq']), 0,
                               event_type_id]])
        epoch = mne.Epochs(raw, event_arr, tmin=tmin, tmax=tmax,
                           baseline=None, preload=True, verbose=False, picks='eeg')
        #for adding the metadata
        epoch.metadata = pd.DataFrame({
            'subject_id': [subject_id],
            'session_id': [session_id],
            'epoch_type_class': [epoch_key_local],
            'unique_epoch_id': [f"{subject_id}_{session_id}_{epoch_key_local}_0000"],
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
    """Extracts non-seizure epochs from a single raw file."""
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
        indices_to_select = np.arange(0, len(valid_starts), step)[:max_epochs_per_file_local]
    else:
        indices_to_select = np.arange(len(valid_starts))

    actual_selected_starts = valid_starts[indices_to_select]

    if len(actual_selected_starts) == 0:
        logger.info(f"No non-seizure epochs without overlap for {fname}foumd.")
        return None

    events_arr = np.array([[s, 0, 4] for s in actual_selected_starts])
    tmin_ns, tmax_ns = epoch_config_local_ns['non_seizure']
    try:
        epochs = mne.Epochs(raw, events_arr, tmin=tmin_ns, tmax=tmax_ns,
                          baseline=None, preload=True, verbose=False, picks='eeg')
        
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


def process_single_edf_file(
        edf_path, target_sfreq, epoch_config_param,
        file_sample_limit_param, file_buffer_sec_param):
    """
    Processes one EDF. Determines channel type (left_bte_crosstop,
    right_bte_crosstop, both_bte_no_crosstop) & extracts epochs.
    Returns (epochs_dict, bte_type_string, subject_id, session_id).
    """
    original_basename = os.path.basename(edf_path)
    logger.info(f"Processing file: {original_basename}")
    local_epochs = {key: [] for key in epoch_config_param}
    bte_category = None
    picked_chs_for_this_file = None

    subject_id = extract_subject_id(edf_path)
    session_id = extract_session_id(edf_path)
    logger.info(f"Extracted subject ID: {subject_id}, session ID: {session_id} for {original_basename}")

    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        if raw.n_times == 0:
            logger.warning(f"Empty file: {original_basename}. Skipping.")
            return None, None, subject_id, session_id

        ch_names = raw.ch_names
        bte_left_ch = 'BTEleft SD'
        bte_right_ch = 'BTEright SD'
        crosstop_ch = 'CROSStop SD'

        has_bte_left = bte_left_ch in ch_names
        has_bte_right = bte_right_ch in ch_names
        has_crosstop = crosstop_ch in ch_names

        if has_bte_left and has_crosstop:
            picked_chs_for_this_file = [bte_left_ch, crosstop_ch]
            bte_category = 'left_bte_crosstop'
        elif has_bte_right and has_crosstop:
            picked_chs_for_this_file = [bte_right_ch, crosstop_ch]
            bte_category = 'right_bte_crosstop'
        elif has_bte_left and has_bte_right and not has_crosstop:
            picked_chs_for_this_file = [bte_left_ch, bte_right_ch]
            bte_category = 'both_bte_no_crosstop'
        else:
            logger.warning(
                f" Cant form a required 2 channel pair for {original_basename}. "
                f"Available: {ch_names}. Skipping."
            )
            return None, None, subject_id, session_id

        logger.info(f"Using channels: {picked_chs_for_this_file} "
                    f"({bte_category}) for {original_basename}")
        raw.pick(picked_chs_for_this_file)

        raw.filter(l_freq=0.5, h_freq=40., fir_design='firwin', verbose=False)
        raw.notch_filter(freqs=50, fir_design='firwin', verbose=False)
        if raw.info['sfreq'] != target_sfreq:
            raw.resample(target_sfreq, npad='auto', verbose=False)

        events_tsv = edf_path.replace('_eeg.edf', '_events.tsv')
        events_df = pd.DataFrame()
        if os.path.exists(events_tsv):
            try:
                events_df = pd.read_csv(events_tsv, sep='\t')
            except Exception as e_csv:
                logger.error(f"Error reading CSV {events_tsv}: {e_csv}.")
        else:
            logger.warning(f"No events file: {events_tsv} for {original_basename}.")

        seizure_intervals_samps = []
        sz_events_df = events_df[events_df['eventType'].str.startswith('sz_', na=False)] \
            if 'eventType' in events_df.columns else pd.DataFrame()

        for _, r in sz_events_df.iterrows():
            try:
                on_val = pd.to_numeric(r['onset'])
                dur_val = pd.to_numeric(r['duration'])
                if pd.isna(on_val) or pd.isna(dur_val): continue
                on_s = int(on_val * raw.info['sfreq'])
                dur_s = int(dur_val * raw.info['sfreq'])
                seizure_intervals_samps.append((on_s, on_s + dur_s))
            except (ValueError, TypeError) as ve:
                logger.warning(f"Value/Type Error in onset/duration: {ve} for {original_basename}. Skipping event.")
                continue

        if not sz_events_df.empty:
            #for creating seizure related epochs, so onset, ictal, precital
            for idx_sz, sz_row in sz_events_df.iterrows():
                try:
                    event_on = pd.to_numeric(sz_row['onset'])
                    if pd.isna(event_on):
                        continue
                except (ValueError, TypeError) as ve:
                    logger.warning(f"Value/Type Error in onset: {ve} for {original_basename}. Skipping sz_event row.")
                    continue

                for key_cls in ('preictal', 'ictal', 'onset'):
                    tmin, tmax = epoch_config_param[key_cls]
                    epoch = create_epoch_for_file(raw, event_on, tmin, tmax, key_cls, subject_id, session_id)
                    if epoch:
                        # Update unique_epoch_id to be more specific if multiple seizures in one file - gemini..
                        epoch.metadata['unique_epoch_id'] = f"{subject_id}_{session_id}_{key_cls}_{idx_sz:04d}"
                        local_epochs[key_cls].append(epoch)
        elif not events_df.empty:
            logger.info(f"No sz_ events in {events_tsv} for {original_basename}.")

        ns_dur = epoch_config_param['non_seizure'][1] - epoch_config_param['non_seizure'][0]
        ns_eps = process_non_seizure_epochs_for_file(
            raw, seizure_intervals_samps, ns_dur, target_sfreq,
            file_sample_limit_param, file_buffer_sec_param, epoch_config_param,
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
            f"Finished {original_basename} ({subject_id}). Epochs this file - "
            f"{', '.join(counts_str_parts)}"
        )

        del raw
        gc.collect()
        return local_epochs, bte_category, subject_id, session_id

    except Exception as e:
        logger.error(f"Error in {original_basename}: {e}", exc_info=True)
        if 'raw' in locals():
            del raw
        gc.collect()
        return None, None, subject_id, session_id


def process_epochs_in_batches(epochs_list, batch_size, key_class,
                              category_suffix, output_dir):
    """Process epochs in smaller batches to reduce memory usage."""
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

    batch_num = 1
    current_batch = []
    current_count = 0

    for ep in epochs_list:
        if current_count + len(ep) > batch_size and current_batch:
            logger.info(f"Processing batch {batch_num} with {current_count} E")
            process_epoch_batch(current_batch, key_class, category_suffix,
                                f"{key_class}_{category_suffix}_batch{batch_num}",
                                output_dir)
            current_batch = []
            current_count = 0
            batch_num += 1
            gc.collect()

        current_batch.append(ep)
        current_count += len(ep)

    if current_batch:
        logger.info(f"Processing final batch {batch_num} with {current_count} E")
        process_epoch_batch(current_batch, key_class, category_suffix,
                            f"{key_class}_{category_suffix}_batch{batch_num}",
                            output_dir)
        gc.collect()


def process_epoch_batch(epochs_batch, key_class, category_suffix, batch_id,
                        output_dir):
    """Process and save a single batch of epochs."""
    try:
        all_eps = mne.concatenate_epochs(epochs_batch)

        if all_eps.metadata is None:
            logger.warning(f"Missing metadata after concatenation for batch {batch_id}. Creating empty DataFrame.")
            all_eps.metadata = pd.DataFrame(index=range(len(all_eps)))

        min_s = 10 if key_class != 'non_seizure' else 50
        file_id_log = f"{batch_id}"

        eps_to_save = fallback_rejection(
            all_eps, file_identifier_for_log=file_id_log, min_samples=min_s
        )

        if len(eps_to_save) > 0:
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
    if 'all_eps' in locals():
        del all_eps
    if 'eps_to_save' in locals():
        del eps_to_save
    gc.collect()


def process_by_subject(results, output_path):
    """Process and save data by subject."""
    logger.info("Processing data by subject...")

    # organize by subject
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

            # add the data to the category its in
            target_container = subject_data[subject_id][bte_cat_str]

            for key, ep_list_from_file in file_epochs_dict.items():
                if ep_list_from_file:
                    for ep_obj in ep_list_from_file:
                        if isinstance(ep_obj, mne.BaseEpochs) and len(ep_obj) > 0:
                            target_container[key].append(ep_obj)

    for subject_id, categories in subject_data.items():
        logger.info(f"Processing subject: {subject_id}")
        subject_output_path = os.path.join(output_path, subject_id)
        os.makedirs(subject_output_path, exist_ok=True)

        for cat_name, container in categories.items():
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
    """Concatenates, rejects, and saves epochs for a given category in batches."""
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


if __name__ == "__main__":
    logger.info(
        f"Starting EEG preprocessing. Base: {BASE_PATH}, Sfreq: {SFREQ}, "
        f"Jobs: {args.n_jobs}, Sample Limit per file (non-seizure): {args.sample_limit}, "
        f"Max epochs per batch: {args.max_epochs_per_batch}, "
        f"Process by subject: {args.process_by_subject}"
    )
    all_edf_files = glob.glob(
        os.path.join(BASE_PATH, "sub-*", "ses-*", "eeg", "*_eeg.edf")
    )
    if not all_edf_files:
        logger.error("No EDF files found. Check --base_path.")
        exit()
    logger.info(f"Found {len(all_edf_files)} EDF files.")

    if args.n_jobs == 1:
        logger.info("Using sequential processing")
        results = []
        for edf_path in tqdm(all_edf_files, desc="Processing EEG files"):
            try:
                result = process_single_edf_file(
                    edf_path, SFREQ, EPOCH_CONFIG,
                    args.sample_limit, BUFFER_SEC
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {edf_path}: {e}", exc_info=True)
                results.append((None, None, None, None))
    else:
        logger.info(f"Using parallel processing (n_jobs={args.n_jobs})")
        results = Parallel(n_jobs=args.n_jobs)(
            delayed(process_single_edf_file)(
                edf_path, SFREQ, EPOCH_CONFIG,
                args.sample_limit, BUFFER_SEC
            ) for edf_path in tqdm(all_edf_files, desc="Processing EEG files")
        )

    logger.info("File processing complete. Aggregating results.")

    if args.process_by_subject:
        output_path = os.path.join(BASE_PATH, "derivatives",
                                   "eeg_preprocessed_by_subject")
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

        logger.info("\n--- Aggregated Epoch Counts Before Final Processing ---")
        for cat_name, container in [
            ("left_bte_crosstop", epochs_left_bte_crosstop),
            ("right_bte_crosstop", epochs_right_bte_crosstop),
            ("both_bte_no_crosstop", epochs_both_bte_no_crosstop)
        ]:
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
                                   "eeg_preprocessed_batched")
        os.makedirs(output_path, exist_ok=True)
        logger.info(f"Saving preprocessed data to: {output_path}")

        save_epochs_by_category(epochs_left_bte_crosstop, "left_bte_crosstop",
                                output_path)
        del epochs_left_bte_crosstop
        gc.collect()

        save_epochs_by_category(epochs_right_bte_crosstop, "right_bte_crosstop",
                                output_path)
        del epochs_right_bte_crosstop
        gc.collect()

        save_epochs_by_category(epochs_both_bte_no_crosstop,
                                "both_bte_no_crosstop", output_path)
        del epochs_both_bte_no_crosstop
        gc.collect()

    logger.info("EEG preprocessing script finished successfully.")
