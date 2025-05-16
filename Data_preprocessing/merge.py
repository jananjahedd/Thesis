#!/usr/bin/env python3
import argparse
import os
import numpy as np
import pandas as pd
import mne
import gc
import logging

np.random.seed(42)

parser = argparse.ArgumentParser(
    description="Merge and align EEG and ECG epochs, replicate for balancing, "
                "and save for DL."
)
parser.add_argument(
    '--base_path',
    required=True,
    help="Root folder containing the '*-clean-epo.fif' files."
)
parser.add_argument(
    '--outdir',
    default='dl_ready_aligned_no_aug',
    help="Output directory for .npy files and metadata."
)
parser.add_argument(
    '--ratio',
    type=float,
    default=1.0,
    help="Target ratio for replicating minority classes relative to majority "
         "(e.g., 1.0 attempts to balance)."
)
parser.add_argument(
    '--no_ecg',
    action='store_true',
    help="Ignore ECG modality; Xeeg will be EEG, Xecg will be zeros."
)
args = parser.parse_args()

EPOCH_TYPES = ['preictal', 'ictal', 'onset', 'non_seizure']
MINORITY_CLASSES = ['preictal', 'ictal', 'onset']
MAJORITY_CLASS = 'non_seizure'


def eeg_fif_path(typ):
    """Constructs path to EEG clean epoch FIF file."""
    return os.path.join(args.base_path, f"{typ}_epochs-clean-epo.fif")


def ecg_fif_path(typ):
    """Constructs path to ECG clean epoch FIF file."""
    return os.path.join(args.base_path, f"{typ}_ecg-clean-epo.fif")


def aug_eeg(x_eeg_single_epoch, sf):
    """
    Returns a copy of the EEG epoch. Transformative augmentations removed(atm).
    """
    return x_eeg_single_epoch.copy()


def aug_ecg_waveform(x_ecg_single_epoch, sf):
    """
    Returns a copy of the ECG epoch waveform.
    """
    return x_ecg_single_epoch.copy()


def main():
    """
    Main function to load, align, replicate for balancing,
    and save EEG/ECG epoch data.
    """
    output_dir = os.path.join(args.base_path, args.outdir)
    os.makedirs(output_dir, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(output_dir, "merge_log.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode='w'
    )
    logger = logging.getLogger(__name__)

    label_map = {name: i for i, name in enumerate(EPOCH_TYPES)}

    all_aligned_eeg_data = []
    all_aligned_ecg_data = []
    all_labels = []
    all_aligned_metadata = []

    logger.info("--- Loading, Aligning, and Replicating Epochs ---")

    sf = None
    seq_len_eeg, seq_len_ecg = None, None
    n_channels_eeg, n_channels_ecg = None, 1

    for typ_for_info in EPOCH_TYPES:
        eeg_fp = eeg_fif_path(typ_for_info)
        if os.path.exists(eeg_fp):
            try:
                temp_eeg = mne.read_epochs(eeg_fp, preload=False,
                                           verbose=False)
                if len(temp_eeg.events) > 0:
                    sf = temp_eeg.info['sfreq']
                    _, n_channels_eeg, seq_len_eeg = temp_eeg.get_data(
                        copy=False).shape
                    if not args.no_ecg:
                        ecg_fp = ecg_fif_path(typ_for_info)
                        if os.path.exists(ecg_fp):
                            temp_ecg = mne.read_epochs(
                                ecg_fp, preload=False, verbose=False
                            )
                            if len(temp_ecg.events) > 0:
                                _, n_channels_ecg, seq_len_ecg = \
                                    temp_ecg.get_data(copy=False).shape
                            else:
                                seq_len_ecg = seq_len_eeg
                    break
            except Exception as e:
                logger.error(f"Error reading info from {eeg_fp}: {e}")
                continue

    if sf is None or seq_len_eeg is None:
        logger.error("Could not determine sfreq or sequence length from "
                     "preprocessed files. Exiting.")
        return

    logger.info(f"Determined sfreq: {sf}, EEG: {seq_len_eeg} samples, "
                f"{n_channels_eeg} channels.")
    if not args.no_ecg and seq_len_ecg:
        logger.info(f"ECG: {seq_len_ecg} samples, {n_channels_ecg} channels.")
    elif not args.no_ecg and not seq_len_ecg:
        logger.warning("ECG sequence length not determined, use EEG length"
                       " for dummy ECG if needed.")
        seq_len_ecg = seq_len_eeg

    original_counts = {}

    for typ in EPOCH_TYPES:
        logger.info(f"\nProcessing type: {typ}")
        eeg_fp = eeg_fif_path(typ)
        if not os.path.exists(eeg_fp):
            logger.warning(f"EEG file not found for {typ}: {eeg_fp}. Skipping")
            original_counts[typ] = 0
            continue

        try:
            eeg_epochs = mne.read_epochs(eeg_fp, preload=True, verbose=False)
        except Exception as e:
            logger.error(f"Could not load EEG epochs for {typ} from {eeg_fp}: "
                         f"{e}. Skipping.")
            original_counts[typ] = 0
            continue

        if eeg_epochs.metadata is None or \
           'unique_epoch_id' not in eeg_epochs.metadata.columns:
            logger.error(f"EEG metadata missing 'unique_epoch_id' for {typ}. "
                         "Cannot align. Skipping.")
            original_counts[typ] = 0
            del eeg_epochs
            gc.collect()
            continue

        eeg_meta = eeg_epochs.metadata.copy()
        eeg_data_arr = eeg_epochs.get_data(copy=True)
        del eeg_epochs
        gc.collect()

        original_counts[typ] = len(eeg_meta)
        logger.info(f"Loaded {len(eeg_meta)} EEG epochs for {typ}.")

        ecg_meta, ecg_data_arr = None, None
        if not args.no_ecg:
            ecg_fp = ecg_fif_path(typ)
            if not os.path.exists(ecg_fp):
                logger.warning(f"ECG file not found for {typ}: {ecg_fp}.")
            else:
                try:
                    ecg_epochs = mne.read_epochs(ecg_fp, preload=True,
                                                 verbose=False)
                    if len(ecg_epochs.events) == 0:
                        logger.warning(f"ECG file {ecg_fp} for {typ} "
                                       "contains no epochs.")
                    elif ecg_epochs.metadata is None or 'unique_epoch_id' not in ecg_epochs.metadata.columns:
                        logger.error(f"ECG metadata missing 'unique_epoch_id' "
                                     f"for {typ}. Skipping ECG for this type.")
                    else:
                        ecg_meta = ecg_epochs.metadata.copy()
                        ecg_data_arr = ecg_epochs.get_data(copy=True)
                        logger.info(f"Loaded {len(ecg_meta)} ECG epochs for "
                                    f"{typ}")
                    del ecg_epochs
                    gc.collect()
                except Exception as e:
                    logger.error(f"Could not load ECG epochs for {typ} from "
                                 f"{ecg_fp}: {e}")

        current_eeg_data = eeg_data_arr
        current_meta = eeg_meta
        _seq_len_for_dummy_ecg = seq_len_ecg if seq_len_ecg is not None else seq_len_eeg
        current_ecg_data = np.zeros(
            (len(current_eeg_data), n_channels_ecg, _seq_len_for_dummy_ecg),
            dtype='float32'
        )


        if ecg_meta is not None and ecg_data_arr is not None:
            ecg_feature_cols = [
                col for col in ecg_meta.columns
                if col.startswith('HR_') or col in [
                    "RMSSD", "MeanNN", "SDNN", "MedianNN", "pNN50",
                    "LF", "HF", "LFHF"
                ]
            ]
            merged_meta = pd.merge(
                eeg_meta,
                ecg_meta[['unique_epoch_id'] + ecg_feature_cols],
                on='unique_epoch_id',
                how='inner'
            )
            logger.info(f"Aligned {len(merged_meta)} EEG/ECG epochs for {typ}")

            if not merged_meta.empty:
                eeg_meta_reindexed = eeg_meta.set_index('unique_epoch_id')
                ecg_meta_reindexed = ecg_meta.set_index('unique_epoch_id')

                try:
                    valid_ids_in_merged = merged_meta['unique_epoch_id']
                    eeg_indices_aligned = eeg_meta_reindexed.index.get_indexer(
                        valid_ids_in_merged[
                            valid_ids_in_merged.isin(eeg_meta_reindexed.index)
                        ]
                    )
                    ecg_indices_aligned = ecg_meta_reindexed.index.get_indexer(
                        valid_ids_in_merged[
                            valid_ids_in_merged.isin(ecg_meta_reindexed.index)
                        ]
                    )

                    eeg_indices_aligned = eeg_indices_aligned[
                        eeg_indices_aligned != -1]
                    ecg_indices_aligned = ecg_indices_aligned[
                        ecg_indices_aligned != -1]

                    min_len = min(len(eeg_indices_aligned),
                                  len(ecg_indices_aligned))

                    current_eeg_data = eeg_data_arr[
                        eeg_indices_aligned[:min_len]]
                    current_ecg_data = ecg_data_arr[
                        ecg_indices_aligned[:min_len]]
                    current_meta = merged_meta.iloc[:min_len].copy()

                    if not (len(current_eeg_data) == len(current_ecg_data) ==
                            len(current_meta)):
                        logger.warning(
                            f"Post-alignment length mismatch for {typ}. "
                            "Reverting to EEG-only for this type."
                        )
                        current_eeg_data = eeg_data_arr
                        current_meta = eeg_meta
                        current_ecg_data = np.zeros(
                            (len(current_eeg_data), n_channels_ecg,
                             _seq_len_for_dummy_ecg), dtype='float32'
                        )
                except KeyError as e:
                    logger.error(f"KeyError during alignment for {typ}: {e}. "
                                 "Using EEG-only for this type.")
                    current_eeg_data = eeg_data_arr
                    current_meta = eeg_meta
                    current_ecg_data = np.zeros(
                        (len(current_eeg_data), n_channels_ecg,
                         _seq_len_for_dummy_ecg), dtype='float32'
                    )
            else:
                logger.info(f"No epochs aligned for {typ}. Using EEG-only.")
                current_eeg_data = eeg_data_arr
                current_meta = eeg_meta
                current_ecg_data = np.zeros(
                    (len(current_eeg_data), n_channels_ecg,
                     _seq_len_for_dummy_ecg), dtype='float32'
                )
        elif not args.no_ecg:
            logger.info(f"Proceeding with EEG-only for type {typ} due to "
                        "missing/unalignable ECG data.")

        num_current_samples = len(current_meta)
        if num_current_samples == 0:
            logger.info(f"No samples for {typ} to process after alignment.")
            continue

        replication_factor = 1
        if typ in MINORITY_CLASSES:
            majority_count = original_counts.get(MAJORITY_CLASS, 0)
            if majority_count > 0 and num_current_samples > 0:
                target_count_minority = int(majority_count * args.ratio)
                if target_count_minority > num_current_samples:
                    replication_factor = int(
                        np.ceil(target_count_minority / num_current_samples)
                    )
                logger.info(
                    f"Balancing {typ}: Have {num_current_samples}, "
                    f"target ~{target_count_minority} (ratio {args.ratio} "
                    f"of majority {majority_count}). "
                    f"Replication factor: {replication_factor}."
                )

        for i in range(num_current_samples):
            for r_idx in range(replication_factor):
                eeg_sample_to_add = current_eeg_data[i]
                ecg_sample_to_add = current_ecg_data[i]
                is_augmented_flag = False

                if r_idx > 0:
                    is_augmented_flag = True
                    # Augmentation functions now just return copies
                    eeg_sample_to_add = aug_eeg(current_eeg_data[i], sf)
                    if not args.no_ecg and current_ecg_data[i].size > 0:
                        ecg_sample_to_add = aug_ecg_waveform(
                            current_ecg_data[i], sf
                        )

                all_aligned_eeg_data.append(eeg_sample_to_add)
                all_aligned_ecg_data.append(ecg_sample_to_add)
                all_labels.append(label_map[typ])

                meta_row = current_meta.iloc[i].to_dict()
                meta_row['is_augmented'] = is_augmented_flag
                meta_row['replication_index'] = r_idx
                all_aligned_metadata.append(meta_row)

        del eeg_data_arr
        if ecg_data_arr is not None:
            del ecg_data_arr
        gc.collect()

    if not all_labels:
        logger.error("No data was processed or aligned. Exiting.")
        return

    indices = np.arange(len(all_labels))
    np.random.shuffle(indices)

    Xeeg_final = np.array(all_aligned_eeg_data, dtype='float32')[indices]
    Xecg_final = np.array(all_aligned_ecg_data, dtype='float32')[indices]
    y_final = np.array(all_labels, dtype='int64')[indices]
    meta_final_df = pd.DataFrame(
        all_aligned_metadata).iloc[indices].reset_index(drop=True)

    logger.info(f"\nTotal samples generated: {len(y_final)}")
    for i, name in enumerate(EPOCH_TYPES):
        logger.info(f"Class {name} (label {i}): {np.sum(y_final == i)} "
                    f"samples")

    np.save(os.path.join(output_dir, 'Xeeg.npy'), Xeeg_final)
    np.save(os.path.join(output_dir, 'Xecg.npy'), Xecg_final)
    np.save(os.path.join(output_dir, 'y.npy'), y_final)
    meta_final_df.to_csv(
        os.path.join(output_dir, 'meta_aligned.csv'), index=False
    )

    logger.info(f"\nData saved to directory: {output_dir}")
    logger.info("Xeeg.npy, Xecg.npy, y.npy, meta_aligned.csv created.")


if __name__ == "__main__":
    main()
