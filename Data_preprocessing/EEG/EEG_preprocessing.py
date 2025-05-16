#!/usr/bin/env python3
import os
import glob
import logging
import argparse
import pandas as pd
import numpy as np
import mne
from tqdm import tqdm
from itertools import zip_longest
import gc
from joblib import Parallel, delayed

logging.basicConfig(
    filename='preprocessing_log.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a'
)
logger = logging.getLogger()

epoch_config = {
    'preictal': (-10, 0),
    'ictal':    (0, 10),
    'onset':    (-5, 5),
    'non_seizure': (0, 10)
}


def fallback_rejection(epochs, threshold_uV=150, min_samples=None):
    """
    Rejects epochs based on peak-to-peak amplitude and variance.
    """
    data = epochs.get_data(units="uV")
    bad_amp = np.any(np.abs(data) > threshold_uV, axis=(1, 2))
    var = np.var(data, axis=(1, 2))
    m, s = np.mean(var), np.std(var)
    bad_var = np.abs(var - m) > 3 * s
    bad = bad_amp | bad_var
    keep = ~bad
    num_original = len(epochs)
    num_kept = keep.sum()

    if min_samples and num_kept < min_samples and num_original >= min_samples:
        logger.warning(f"Fallback rejection initially kept {num_kept}/{num_original} epochs. "
                       f"Attempting to keep at least {min_samples} based on variance from file: {epochs.info.get('subject_info', {}).get('description', 'N/A')}.") # Added filename to log
        idx_sorted_by_variance_diff = np.argsort(np.abs(var - m))
        keep_indices = idx_sorted_by_variance_diff[:min_samples]
        keep = np.zeros(num_original, dtype=bool)
        keep[keep_indices] = True
        num_kept = keep.sum()

    logger.info(f"Fallback rejection for {epochs.info.get('subject_info', {}).get('description', 'N/A')}: kept {num_kept}/{num_original} epochs.")
    return epochs[keep]

def create_epoch_for_file(raw, event_onset_time, tmin, tmax, epoch_key_local, times_ref_dict_local):
    """
    Creates a single MNE epoch.
    Note: times_ref_dict_local is for consistency within epochs from THE SAME FILE if needed,
    but global consistency is harder with parallel. We assume MNE is consistent if params are.
    """
    start_samp_raw = int((event_onset_time + tmin) * raw.info['sfreq'])
    end_samp_raw   = int((event_onset_time + tmax) * raw.info['sfreq'])

    if start_samp_raw < 0 or end_samp_raw > raw.n_times:
        logger.warning(f"Epoch {epoch_key_local} for event at {event_onset_time:.2f}s in {raw.filenames[0]} is OOB. Skipping.")
        return None
    try:
        event_sample = int(event_onset_time * raw.info['sfreq'])
        events_array = np.array([[event_sample, 0, 1]])
        ep = mne.Epochs(raw, events_array, tmin=tmin, tmax=tmax,
                        baseline=None, preload=True, verbose=False, picks='eeg')

        if epoch_key_local not in times_ref_dict_local:
            times_ref_dict_local[epoch_key_local] = ep.times
        elif not np.allclose(times_ref_dict_local[epoch_key_local], ep.times):
            logger.error(f"Time vector mismatch for {epoch_key_local} in {raw.filenames[0]}! Skipping epoch.")
            return None
        return ep
    except Exception as e:
        logger.warning(f"Failed to create epoch for {epoch_key_local} at {event_onset_time:.2f}s in {raw.filenames[0]}: {e}")
        return None

def process_non_seizure_epochs_for_file(raw, seizure_intervals_samples, epoch_duration_sec, 
                                        target_sfreq_local, max_epochs_per_file_local, buffer_sec_local, epoch_config_local):
    """
    Extracts non-seizure epochs from a single raw file.
    """
    non_seizure_mask = np.ones(raw.n_times, dtype=bool)
    safety_buffer_samples = int(buffer_sec_local * target_sfreq_local)

    for start_samp, end_samp in seizure_intervals_samples:
        buffer_start = max(0, start_samp - safety_buffer_samples)
        buffer_end   = min(raw.n_times, end_samp + safety_buffer_samples)
        non_seizure_mask[buffer_start:buffer_end] = False

    epoch_length_samples = int(epoch_duration_sec * target_sfreq_local)
    if epoch_length_samples == 0 :
        logger.error(f"Epoch length for non-seizure is zero for {raw.filenames[0]}. Skipping non-seizure processing.")
        return None

    valid_starts = np.where(np.convolve(non_seizure_mask, np.ones(epoch_length_samples, dtype=int), 'valid') == epoch_length_samples)[0]

    if len(valid_starts) == 0:
        logger.info(f"No valid segments for non-seizure epochs in {raw.filenames[0]}.")
        return None

    selected_starts = []
    last_selected_end = -1
    for start_candidate in valid_starts:
        if start_candidate >= last_selected_end:
            selected_starts.append(start_candidate)
            last_selected_end = start_candidate + epoch_length_samples
            if max_epochs_per_file_local and len(selected_starts) >= max_epochs_per_file_local:
                break

    if not selected_starts:
        logger.info(f"No non-overlapping non-seizure epochs selected in {raw.filenames[0]}.")
        return None

    events_array = np.array([[s, 0, 2] for s in selected_starts])
    tmin_ns, tmax_ns = epoch_config_local['non_seizure']

    try:
        non_seizure_eps = mne.Epochs(raw, events_array, tmin=tmin_ns, tmax=tmax_ns,
                                     baseline=None, preload=True, verbose=False, picks='eeg')
        return non_seizure_eps
    except Exception as e:
        logger.error(f"Error creating non-seizure epochs for {raw.filenames[0]}: {e}")
        return None


def process_single_edf_file(edf_path, target_sfreq, epoch_config_dict, file_sample_limit, file_buffer_sec):
    """
    Processes a single EDF file: loads, filters, resamples, and extracts epochs.
    MODIFIED: Processes the file if at least two channels from
    ['BTEleft SD', 'BTEright SD', 'CROSStop SD'] are found, using all such found channels.
    Returns a dictionary of epoch lists for this file.
    """
    original_basename = os.path.basename(edf_path)
    logger.info(f"Processing file: {original_basename}")

    local_epochs_for_file = {key: [] for key in epoch_config_dict}
    local_times_reference = {}

    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

        if raw.n_times == 0:
            logger.warning(f"Empty recording (0 samples) in {original_basename}. Skipping.")
            return None

        ch_names_in_file = raw.ch_names

        target_channel_candidates = ['BTEleft SD', 'BTEright SD', 'CROSStop SD']
        picked_channels_for_file = []

        for ch_name in target_channel_candidates:
            if ch_name in ch_names_in_file:
                picked_channels_for_file.append(ch_name)

        if len(picked_channels_for_file) >= 2:
            logger.info(f"Found {len(picked_channels_for_file)} target channel(s) (>=2 required). "
                        f"Using: {picked_channels_for_file} for {original_basename}")
            raw.pick(picked_channels_for_file)

        else:
            logger.warning(f"Found only {len(picked_channels_for_file)} target channel(s) from the set "
                         f"({', '.join(target_channel_candidates)}). Need at least 2 to proceed. "
                         f"Available channels in file: {ch_names_in_file}. Skipping file.")
            return None

        raw.filter(l_freq=0.5, h_freq=40., fir_design='firwin', verbose=False)
        raw.notch_filter(freqs=50, fir_design='firwin', verbose=False)

        if raw.info['sfreq'] != target_sfreq:
            raw.resample(target_sfreq, npad='auto', verbose=False)

        events_tsv_path = edf_path.replace('_eeg.edf', '_events.tsv')
        seizure_intervals_samples = []

        if os.path.exists(events_tsv_path):
            events_df = pd.read_csv(events_tsv_path, sep='\t')
            seizure_related_events = events_df[events_df['eventType'].str.startswith('sz_', na=False)]

            for _, row in seizure_related_events.iterrows():
                try:
                    onset_val = pd.to_numeric(row['onset'])
                    duration_val = pd.to_numeric(row['duration'])
                    if pd.isna(onset_val) or pd.isna(duration_val):
                        logger.warning(f"Skipping seizure event due to non-numeric onset/duration in {events_tsv_path} for {original_basename}")
                        continue
                    onset_samples = int(onset_val * raw.info['sfreq'])
                    duration_samples = int(duration_val * raw.info['sfreq'])
                    seizure_intervals_samples.append((onset_samples, onset_samples + duration_samples))
                except ValueError as ve:
                    logger.warning(f"Skipping seizure event due to ValueError in onset/duration: {ve} in {events_tsv_path} for {original_basename}")
                    continue

            if not seizure_related_events.empty:
                for _, seizure_event_row in seizure_related_events.iterrows():
                    try:
                        event_onset_time = pd.to_numeric(seizure_event_row['onset'])
                        if pd.isna(event_onset_time):
                            logger.warning(f"Skipping seizure event due to non-numeric onset in {events_tsv_path} for {original_basename}")
                            continue
                    except ValueError as ve:
                        logger.warning(f"Skipping seizure event due to ValueError in onset: {ve} in {events_tsv_path} for {original_basename}")
                        continue

                    for key_class in ('preictal', 'ictal', 'onset'):
                        tmin, tmax = epoch_config_dict[key_class]
                        epoch = create_epoch_for_file(raw, event_onset_time, tmin, tmax, key_class, local_times_reference)
                        if epoch is not None:
                            local_epochs_for_file[key_class].append(epoch)
            else:
                if os.path.exists(events_tsv_path):
                    logger.info(f"No seizure events (sz_*) found in {events_tsv_path} for {original_basename}.")
        else:
            logger.warning(f"Events file not found: {events_tsv_path} for {original_basename}. Skipping event-based epoching.")


        ns_epoch_duration_sec = epoch_config_dict['non_seizure'][1] - epoch_config_dict['non_seizure'][0]
        non_seizure_eps_current_file = process_non_seizure_epochs_for_file(
            raw, seizure_intervals_samples,
            ns_epoch_duration_sec, target_sfreq,
            file_sample_limit, file_buffer_sec, epoch_config_dict
        )
        if non_seizure_eps_current_file is not None and len(non_seizure_eps_current_file) > 0:
            local_epochs_for_file['non_seizure'].append(non_seizure_eps_current_file)

        counts_str_parts = []
        for k, v_list in local_epochs_for_file.items():
            current_count = 0
            if v_list:
                for ep_item in v_list:
                    if isinstance(ep_item, mne.BaseEpochs):
                        current_count += len(ep_item)
            counts_str_parts.append(f"{k.capitalize()}: {current_count}")
        logger.info(f"Finished {original_basename}. Epochs this file - {', '.join(counts_str_parts)}")

        del raw
        gc.collect()
        return local_epochs_for_file

    except Exception as e:
        logger.error(f"Unhandled error processing file {edf_path}: {e}", exc_info=True)
        if 'raw' in locals(): del raw
        gc.collect()
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="EEG preprocessing for seizure onset detection (Parallelized)")
    parser.add_argument('--base_path', required=True,
                        help='Root folder containing sub-*/ses-*/eeg data')
    parser.add_argument('--sfreq', type=int, default=256,
                        help='Target sampling frequency')
    parser.add_argument('--sample_limit', type=int, default=500,
                        help='Max non-seizure epochs per EDF file')
    parser.add_argument('--n_jobs', type=int, default=-1,
                        help='Number of parallel jobs for joblib (-1 uses all cores)')
    args = parser.parse_args()

    base_path_main = args.base_path
    sfreq_main = args.sfreq
    buffer_sec_main = 5

    logger.info(f"Starting EEG preprocessing. Base path: {base_path_main}, Target sfreq: {sfreq_main} Hz.")

    all_edf_files = glob.glob(os.path.join(base_path_main, "sub-*", "ses-*", "eeg", "*_eeg.edf"))
    if not all_edf_files:
        logger.error("No EDF files found. Please check the base_path.")
        exit()

    logger.info(f"Found {len(all_edf_files)} EDF files to process.")
    results = Parallel(n_jobs=args.n_jobs)(
        delayed(process_single_edf_file)(
            edf_path,
            sfreq_main,
            epoch_config,
            args.sample_limit,
            buffer_sec_main
        ) for edf_path in tqdm(all_edf_files, desc="Processing EEG files")
    )

    logger.info("Parallel processing complete. Aggregating results...")
    final_aggregated_containers = {key: [] for key in epoch_config}
    for file_result_dict in results:
        if file_result_dict:
            for key, epoch_objects_from_file in file_result_dict.items():
                if epoch_objects_from_file:
                    for ep_obj in epoch_objects_from_file:
                         if isinstance(ep_obj, mne.BaseEpochs):
                            final_aggregated_containers[key].append(ep_obj)
                         else:
                            logger.warning(f"Expected MNE Epochs object, got {type(ep_obj)} for key {key}. Skipping this item.")


    logger.info("Aggregation complete. Starting final rejection and saving.")
    for key_class, collected_epoch_objects_list in final_aggregated_containers.items():
        if not collected_epoch_objects_list:
            logger.info(f"No epochs to process for type: {key_class} after aggregation.")
            continue

        try:
            valid_epochs_to_cat = [ep for ep in collected_epoch_objects_list if isinstance(ep, mne.BaseEpochs)]
            if not valid_epochs_to_cat:
                logger.warning(f"No valid MNE Epochs objects to concatenate for key: {key_class}")
                continue

            all_epochs_for_key = mne.concatenate_epochs(valid_epochs_to_cat, verbose=False)
            logger.info(f"Concatenated {len(all_epochs_for_key)} total epochs for type: {key_class}")

            min_s = 10 if key_class != 'non_seizure' else 50
            epochs_to_save = fallback_rejection(all_epochs_for_key, min_samples=min_s)

            if len(epochs_to_save) > 0:
                output_fname = os.path.join(base_path_main, f"{key_class}_epochs-clean-epo.fif")
                epochs_to_save.save(output_fname, overwrite=True, verbose=False)
                logger.info(f"Saved {len(epochs_to_save)} clean epochs for {key_class} to: {output_fname}")
            else:
                logger.warning(f"No epochs remained after rejection for type: {key_class}. Nothing saved.")
        except Exception as e:
            logger.error(f"Error during final processing/saving for type {key_class}: {e}", exc_info=True)

    logger.info("EEG preprocessing script finished.")
