#!/usr/bin/env python3
import argparse
import os
import glob
import gc
import logging
import numpy as np
import pandas as pd
import mne
import pickle
import json
import re
import random
import shutil
import warnings
import sys
from datetime import datetime
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler

try:
    from tqdm import tqdm
except ImportError:
    print("tqdm not found, progress bars will not be shown")
    def tqdm(iterable, *args, **kwargs):
        if hasattr(iterable, '__len__'):
            print(f"Processing {len(iterable)} items...")
        else:
            print("Processing items...")
        return iterable

parser = argparse.ArgumentParser(description="Merge and prepare EEG and ECG data for fusion-based seizure onset prediction.")
parser.add_argument("--eeg_path", required=True, help="Path to preprocessed EEG data")
parser.add_argument("--ecg_path", required=True, help="Path to preprocessed ECG data")
parser.add_argument("--output_path", required=True, help="Path to save merged data")
parser.add_argument("--categories", nargs='+', default=["left_bte_crosstop", "right_bte_crosstop", "both_bte_no_crosstop"],
                    help="BTE categories to process")
parser.add_argument("--test_split", type=float, default=0.2, help="Proportion of data to use for testing")
parser.add_argument("--val_split", type=float, default=0.15, help="Proportion of training data to use for validation")
parser.add_argument("--augment", action="store_true", help="Apply data augmentation")
parser.add_argument("--augment_factor", type=float, default=2.0, 
                    help="Augmentation factor for minority classes")
parser.add_argument("--n_jobs", type=int, default=-1, help="Number of parallel jobs (-1 for all cores)")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
parser.add_argument("--normalize", action="store_true", help="Apply normalization")
parser.add_argument("--normalize_per_subject", action="store_true", 
                    help="Apply normalization per subject (default is global)")
parser.add_argument("--stratify", action="store_true", 
                    help="Use stratified splits based on class labels")
parser.add_argument("--balance_classes", action="store_true", 
                    help="Balance classes in training set")
parser.add_argument("--balance_method", choices=['undersample', 'oversample', 'smote', 'adasyn'], 
                    default='undersample', help="Method for balancing classes")
parser.add_argument("--debug", action="store_true", help="Enable debug logs")
parser.add_argument("--skip_subjects", nargs='+', default=[], help="Subject IDs to skip")
parser.add_argument("--epoch_types", nargs='+', default=["preictal", "ictal", "onset", "non_seizure"], 
                    help="Epoch types to include")
parser.add_argument("--only_metadata", action="store_true", 
                    help="Only process metadata without loading full data (for debugging)")
parser.add_argument("--save_format", choices=['npy', 'pickle', 'h5'], default='npy',
                    help="Format to save merged data")
parser.add_argument("--test_subjects", nargs='+', default=None,
                    help="Specific subjects to reserve for testing (to guarantee out-of-sample testing)")
parser.add_argument("--no_fusion", action="store_true", 
                    help="Skip fusion and only prepare individual modalities")
parser.add_argument("--max_subjects", type=int, default=None,
                    help="Maximum number of subjects to process (for testing)")
parser.add_argument("--skip_augmentation_for_non_seizure", action="store_true",
                    help="Skip augmentation for non-seizure epochs")

args = parser.parse_args()


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(args.output_path, f"merge_{timestamp}.log")
os.makedirs(args.output_path, exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG if args.debug else logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

np.random.seed(args.seed)
random.seed(args.seed)
if hasattr(pd, 'set_option'):
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', 100)


def find_matching_ecg_epochs(eeg_metadata, ecg_epoch_files, subject_id, epoch_type):
    """
    Find matching ECG epochs for a given subject and epoch type.

    Args:
        eeg_metadata (pd.DataFrame): Metadata for EEG epochs
        ecg_epoch_files (list): List of ECG epoch files
        subject_id (str): Subject ID to match
        epoch_type (str): Epoch type to match (preictal, ictal, onset, non_seizure)

    Returns:
        tuple: (ecg_epochs, ecg_metadata) or (None, None) if no match
    """
    matching_ecg_files = [f for f in ecg_epoch_files
                         if f"{epoch_type}_" in os.path.basename(f)]

    if not matching_ecg_files:
        logger.warning(f"No matching ECG files for {subject_id}/{epoch_type}")
        return None, None

    for ecg_file in matching_ecg_files:
        meta_file = ecg_file.replace("-epo.fif", "_metadata.csv")
        if not os.path.exists(meta_file):
            continue
        try:
            ecg_meta = pd.read_csv(meta_file)
            if 'subject_id' not in ecg_meta.columns:
                continue

            this_subject_ecg = ecg_meta[ecg_meta['subject_id'] == subject_id]
            if len(this_subject_ecg) > 0:
                try:
                    ecg_epochs = mne.read_epochs(ecg_file, verbose=False)

                    if 'subject_id' in ecg_epochs.metadata.columns:
                        subject_mask = ecg_epochs.metadata['subject_id'] == subject_id
                        if np.any(subject_mask):
                            return ecg_epochs[subject_mask], this_subject_ecg
                except Exception as e:
                    logger.warning(f"Error loading ECG file {ecg_file}: {e}")
                    continue
        except Exception as e:
            logger.warning(f"Error processing ECG metadata {meta_file}: {e}")
            continue

    return None, None


def match_eeg_ecg_epochs(eeg_epochs, ecg_epochs, eeg_meta, ecg_meta):
    """
    Match EEG and ECG epochs based on unique_epoch_id.

    Args:
        eeg_epochs (mne.Epochs): EEG epochs
        ecg_epochs (mne.Epochs): ECG epochs
        eeg_meta (pd.DataFrame): EEG metadata
        ecg_meta (pd.DataFrame): ECG metadata

    Returns:
        tuple: (matched_eeg_epochs, matched_ecg_epochs, matched_metadata)
    """
    if 'unique_epoch_id' not in eeg_meta.columns or 'unique_epoch_id' not in ecg_meta.columns:
        logger.warning("Cannot match epochs: unique_epoch_id not found in metadata")
        return None, None, None

    eeg_ids = set(eeg_meta['unique_epoch_id'])
    ecg_ids = set(ecg_meta['unique_epoch_id'])
    common_ids = eeg_ids.intersection(ecg_ids)

    if not common_ids:
        logger.warning(f"No matching epoch IDs found between EEG and ECG")
        return None, None, None

    logger.info(f"Found {len(common_ids)} common epoch IDs out of {len(eeg_ids)} EEG and {len(ecg_ids)} ECG epochs")

    eeg_idx = [i for i, epoch_id in enumerate(eeg_meta['unique_epoch_id']) if epoch_id in common_ids]
    ecg_idx = [i for i, epoch_id in enumerate(ecg_meta['unique_epoch_id']) if epoch_id in common_ids]

    if len(eeg_idx) != len(common_ids) or len(ecg_idx) != len(common_ids):
        logger.warning("Duplicate epoch IDs found. Using first occurrence only.")

    matched_eeg = eeg_epochs[eeg_idx] if eeg_idx else None
    matched_ecg = ecg_epochs[ecg_idx] if ecg_idx else None

    if matched_eeg is not None and matched_ecg is not None:

        eeg_order = list(matched_eeg.metadata['unique_epoch_id'])
        ecg_order = list(matched_ecg.metadata['unique_epoch_id'])

        if eeg_order != ecg_order:
            logger.info("Reordering ECG epochs to match EEG epoch order")

            ecg_id_to_idx = {epoch_id: i for i, epoch_id in enumerate(ecg_order)}

            new_ecg_idx = [ecg_id_to_idx[epoch_id] for epoch_id in eeg_order]
            matched_ecg = matched_ecg[new_ecg_idx]

            assert list(matched_eeg.metadata['unique_epoch_id']) == list(matched_ecg.metadata['unique_epoch_id']), \
                "Failed to reorder ECG epochs to match EEG"

    if matched_eeg is not None and matched_ecg is not None:
        combined_meta = matched_eeg.metadata.copy()

        ecg_features = [col for col in matched_ecg.metadata.columns
                       if col not in combined_meta.columns and
                       col not in ['subject_id', 'session_id', 'epoch_type_class', 'unique_epoch_id']]

        for feature in ecg_features:
            combined_meta[feature] = matched_ecg.metadata[feature].values

        return matched_eeg, matched_ecg, combined_meta

    return None, None, None


def apply_normalization(eeg_data, ecg_data, method='robust', per_subject=False, metadata=None):
    """
    Normalize EEG and ECG data.

    Args:
        eeg_data (np.ndarray): EEG data with shape (n_epochs, n_channels, n_times)
        ecg_data (np.ndarray): ECG data with shape (n_epochs, n_channels, n_times)
        method (str): Normalization method ('robust', 'standard', or 'minmax')
        per_subject (bool): Whether to normalize per subject
        metadata (pd.DataFrame): Metadata with subject IDs if per_subject=True

    Returns:
        tuple: (normalized_eeg, normalized_ecg, scalers)
    """
    if eeg_data is None or ecg_data is None:
        return None, None, None

    if per_subject and metadata is None:
        logger.warning("Cannot normalize per subject: metadata not provided")
        per_subject = False

    scalers = {'eeg': {}, 'ecg': {}}

    norm_eeg = np.zeros_like(eeg_data)
    norm_ecg = np.zeros_like(ecg_data)

    if per_subject:
        unique_subjects = metadata['subject_id'].unique()
        logger.info(f"Normalizing per subject for {len(unique_subjects)} subjects")

        for subject in unique_subjects:
            subj_mask = metadata['subject_id'] == subject
            subj_indices = np.where(subj_mask)[0]

            subj_eeg = eeg_data[subj_indices]
            subj_ecg = ecg_data[subj_indices]

            n_eeg_epochs, n_eeg_ch, n_eeg_times = subj_eeg.shape
            n_ecg_epochs, n_ecg_ch, n_ecg_times = subj_ecg.shape

            reshaped_eeg = subj_eeg.reshape(n_eeg_epochs * n_eeg_ch,
                                            n_eeg_times)
            reshaped_ecg = subj_ecg.reshape(n_ecg_epochs * n_ecg_ch,
                                            n_ecg_times)

            if method == 'robust':
                eeg_scaler = RobustScaler()
                ecg_scaler = RobustScaler()
            else:
                eeg_scaler = StandardScaler()
                ecg_scaler = StandardScaler()

            scaled_eeg = eeg_scaler.fit_transform(reshaped_eeg)
            scaled_ecg = ecg_scaler.fit_transform(reshaped_ecg)

            scaled_eeg = scaled_eeg.reshape(n_eeg_epochs, n_eeg_ch, n_eeg_times)
            scaled_ecg = scaled_ecg.reshape(n_ecg_epochs, n_ecg_ch, n_ecg_times)

            norm_eeg[subj_indices] = scaled_eeg
            norm_ecg[subj_indices] = scaled_ecg

            scalers['eeg'][subject] = eeg_scaler
            scalers['ecg'][subject] = ecg_scaler
    else:

        logger.info("Applying global normalization")

        n_eeg_epochs, n_eeg_ch, n_eeg_times = eeg_data.shape
        n_ecg_epochs, n_ecg_ch, n_ecg_times = ecg_data.shape

        reshaped_eeg = eeg_data.reshape(n_eeg_epochs * n_eeg_ch, n_eeg_times)
        reshaped_ecg = ecg_data.reshape(n_ecg_epochs * n_ecg_ch, n_ecg_times)

        if method == 'robust':
            eeg_scaler = RobustScaler()
            ecg_scaler = RobustScaler()
        else:
            eeg_scaler = StandardScaler()
            ecg_scaler = StandardScaler()

        scaled_eeg = eeg_scaler.fit_transform(reshaped_eeg)
        scaled_ecg = ecg_scaler.fit_transform(reshaped_ecg)

        norm_eeg = scaled_eeg.reshape(n_eeg_epochs, n_eeg_ch, n_eeg_times)
        norm_ecg = scaled_ecg.reshape(n_ecg_epochs, n_ecg_ch, n_ecg_times)

        scalers['eeg']['global'] = eeg_scaler
        scalers['ecg']['global'] = ecg_scaler

    return norm_eeg, norm_ecg, scalers


def augment_data(eeg_data, ecg_data, metadata, augment_factor=2.0, skip_non_seizure=True):
    """
    Apply data augmentation to EEG and ECG data.
    Uses techniques suitable for time series data.

    Args:
        eeg_data (np.ndarray): EEG data with shape (n_epochs, n_channels, n_times)
        ecg_data (np.ndarray): ECG data with shape (n_epochs, n_channels, n_times)
        metadata (pd.DataFrame): Metadata including epoch_type_class
        augment_factor (float): Factor by which to augment seizure data
        skip_non_seizure (bool): Whether to skip augmentation for non-seizure epochs

    Returns:
        tuple: (augmented_eeg, augmented_ecg, augmented_metadata)
    """
    if eeg_data is None or ecg_data is None or metadata is None:
        return None, None, None

    logger.info(f"Applying data augmentation with factor {augment_factor}")

    class_counts = metadata['epoch_type_class'].value_counts()
    logger.info(f"Original class distribution: {class_counts.to_dict()}")

    seizure_classes = ['preictal', 'ictal', 'onset']
    seizure_mask = metadata['epoch_type_class'].isin(seizure_classes)

    if skip_non_seizure:
        aug_mask = seizure_mask
        logger.info("Skipping augmentation for non-seizure epochs")
    else:
        aug_mask = np.ones(len(metadata), dtype=bool)

    aug_indices = np.where(aug_mask)[0]
    n_to_augment = int(len(aug_indices) * (augment_factor - 1))

    if n_to_augment <= 0:
        logger.info("No epochs to augment based on augment_factor")
        return eeg_data, ecg_data, metadata

    sampled_indices = np.random.choice(aug_indices, size=n_to_augment, replace=True)

    aug_eeg = []
    aug_ecg = []
    aug_meta_rows = []

    for i, idx in enumerate(sampled_indices):

        orig_eeg = eeg_data[idx].copy()
        orig_ecg = ecg_data[idx].copy()
        orig_meta = metadata.iloc[idx].copy()

        # apply augmentation techniques - choose randomly
        aug_method = np.random.choice([
            'noise', 'time_shift', 'magnitude_warp', 'time_warp',
            'channel_shuffle', 'sign_flip'
        ])

        noise_level = np.random.uniform(0.01, 0.05)
        time_shift = np.random.randint(-20, 20)  # Samples to shift

        if aug_method == 'noise':
            # add Gaussian noise
            orig_eeg += np.random.normal(0, noise_level, orig_eeg.shape) * np.std(orig_eeg)
            orig_ecg += np.random.normal(0, noise_level, orig_ecg.shape) * np.std(orig_ecg)
            method_desc = f"gaussian_noise_{noise_level:.3f}"

        elif aug_method == 'time_shift':

            orig_eeg = np.roll(orig_eeg, time_shift, axis=-1)
            orig_ecg = np.roll(orig_ecg, time_shift, axis=-1)
            method_desc = f"time_shift_{time_shift}"

        elif aug_method == 'magnitude_warp':

            scale = np.random.uniform(0.8, 1.2)
            orig_eeg *= scale
            orig_ecg *= scale
            method_desc = f"mag_scale_{scale:.2f}"

        elif aug_method == 'time_warp':
            # time warping by resampling
            stretch = np.random.uniform(0.9, 1.1)
            orig_eeg = mne.filter.resample(orig_eeg, down=stretch, axis=-1, npad='auto')
            orig_ecg = mne.filter.resample(orig_ecg, down=stretch, axis=-1, npad='auto')
            # ensure original length
            if orig_eeg.shape[-1] < eeg_data.shape[-1]:
                pad_size = eeg_data.shape[-1] - orig_eeg.shape[-1]
                orig_eeg = np.pad(orig_eeg, ((0, 0), (0, pad_size)))
                orig_ecg = np.pad(orig_ecg, ((0, 0), (0, pad_size)))
            elif orig_eeg.shape[-1] > eeg_data.shape[-1]:
                orig_eeg = orig_eeg[:, :eeg_data.shape[-1]]
                orig_ecg = orig_ecg[:, :ecg_data.shape[-1]]
            method_desc = f"time_warp_{stretch:.2f}"

        elif aug_method == 'channel_shuffle':
            # shuffle EEG channels (for multi-channel only)
            if orig_eeg.shape[0] > 1:
                shuffle_idx = np.random.permutation(orig_eeg.shape[0])
                orig_eeg = orig_eeg[shuffle_idx]
            method_desc = "channel_shuffle"

        elif aug_method == 'sign_flip':
            # flip sign of signal
            if np.random.rand() > 0.5:
                orig_eeg *= -1
            if np.random.rand() > 0.5:
                orig_ecg *= -1
            method_desc = "sign_flip"

        orig_meta['unique_epoch_id'] = f"{orig_meta['unique_epoch_id']}_aug{i}_{method_desc}"
        orig_meta['augmented'] = True
        orig_meta['aug_method'] = aug_method

        aug_eeg.append(orig_eeg)
        aug_ecg.append(orig_ecg)
        aug_meta_rows.append(orig_meta)

    combined_eeg = np.vstack([eeg_data, np.array(aug_eeg)])
    combined_ecg = np.vstack([ecg_data, np.array(aug_ecg)])
    combined_meta = pd.concat([metadata, pd.DataFrame(aug_meta_rows)], ignore_index=True)

    # Verify shapes
    assert len(combined_eeg) == len(combined_meta), "EEG and metadata length mismatch after augmentation"
    assert len(combined_ecg) == len(combined_meta), "ECG and metadata length mismatch after augmentation"

    logger.info(f"After augmentation: {len(combined_meta)} epochs (added {len(aug_meta_rows)} new epochs)")
    logger.info(f"New class distribution: {combined_meta['epoch_type_class'].value_counts().to_dict()}")

    return combined_eeg, combined_ecg, combined_meta


def balance_classes(eeg_data, ecg_data, metadata, method='undersample'):
    """
    Balance classes in the dataset.

    Args:
        eeg_data (np.ndarray): EEG data
        ecg_data (np.ndarray): ECG data
        metadata (pd.DataFrame): Metadata with epoch_type_class
        method (str): Method for balancing ('undersample', 'oversample', 'smote', 'adasyn')

    Returns:
        tuple: (balanced_eeg, balanced_ecg, balanced_metadata)
    """
    if eeg_data is None or ecg_data is None or metadata is None:
        return None, None, None

    logger.info(f"Balancing classes using method: {method}")

    labels = metadata['epoch_type_class'].values
    unique_labels = np.unique(labels)

    class_counts = metadata['epoch_type_class'].value_counts()
    logger.info(f"Original class distribution: {class_counts.to_dict()}")

    n_eeg_epochs, n_eeg_ch, n_eeg_times = eeg_data.shape
    n_ecg_epochs, n_ecg_ch, n_ecg_times = ecg_data.shape

    X_eeg_flat = eeg_data.reshape(n_eeg_epochs, -1)
    X_ecg_flat = ecg_data.reshape(n_ecg_epochs, -1)

    if method == 'undersample':

        min_class_size = class_counts.min()
        logger.info(f"Undersampling to {min_class_size} samples per class")

        undersampler = RandomUnderSampler(random_state=args.seed)
        X_combined = np.hstack([X_eeg_flat, X_ecg_flat])

        X_resampled, y_resampled = undersampler.fit_resample(X_combined, labels)

        kept_indices = undersampler.sample_indices_

    elif method == 'oversample':

        max_class_size = class_counts.max()
        logger.info(f"Oversampling minority classes to {max_class_size} samples per class")

        sampler = {
            'smote': SMOTE(random_state=args.seed),
            'adasyn': ADASYN(random_state=args.seed),
            'oversample': RandomUnderSampler(random_state=args.seed)
        }[method]

        if method == 'oversample':

            balanced_indices = []
            for label in unique_labels:
                label_indices = np.where(labels == label)[0]

                if len(label_indices) < max_class_size:

                    additional = np.random.choice(
                        label_indices, 
                        size=max_class_size - len(label_indices),
                        replace=True
                    )
                    balanced_indices.extend(np.concatenate([label_indices, additional]))
                else:
                    balanced_indices.extend(label_indices)

            kept_indices = np.array(balanced_indices)

        else:  # SMOTE or ADASYN

            X_combined = np.hstack([X_eeg_flat, X_ecg_flat])

            try:
                X_resampled, y_resampled = sampler.fit_resample(X_combined, labels)

                n_orig = len(X_eeg_flat)
                n_new = len(X_resampled)

                if n_new <= n_orig:
                    kept_indices = np.arange(n_orig)
                else:
                    kept_indices = np.arange(n_orig)

                    n_synthetic = n_new - n_orig
                    logger.info(f"Generated {n_synthetic} synthetic samples")

                    eeg_size = X_eeg_flat.shape[1]
                    ecg_size = X_ecg_flat.shape[1]

                    synthetic_combined = X_resampled[n_orig:]
                    synthetic_eeg = synthetic_combined[:, :eeg_size].reshape(-1, n_eeg_ch, n_eeg_times)
                    synthetic_ecg = synthetic_combined[:, eeg_size:].reshape(-1, n_ecg_ch, n_ecg_times)
                    synthetic_labels = y_resampled[n_orig:]

                    synthetic_meta = []
                    base_cols = metadata.columns

                    for i, label in enumerate(synthetic_labels):

                        new_row = pd.Series(index=base_cols)

                        exemplar_idx = np.where(labels == label)[0][0]
                        exemplar = metadata.iloc[exemplar_idx]

                        for col in base_cols:
                            if col == 'epoch_type_class':
                                new_row[col] = label
                            elif col == 'unique_epoch_id':
                                new_row[col] = f"{exemplar['subject_id']}_{exemplar['session_id']}_{label}_synthetic_{i:04d}"
                            elif col == 'subject_id' or col == 'session_id':
                                new_row[col] = exemplar[col]
                            else:

                                new_row[col] = exemplar[col]

                        synthetic_meta.append(new_row)

                    synth_meta_df = pd.DataFrame(synthetic_meta)
                    combined_meta = pd.concat([metadata, synth_meta_df], ignore_index=True)

                    combined_eeg = np.vstack([eeg_data, synthetic_eeg])
                    combined_ecg = np.vstack([ecg_data, synthetic_ecg])

                    return combined_eeg, combined_ecg, combined_meta

            except Exception as e:
                logger.warning(f"Error in {method}: {e}. Falling back to undersampling.")
                method = 'undersample'
                undersampler = RandomUnderSampler(random_state=args.seed)
                X_combined = np.hstack([X_eeg_flat, X_ecg_flat])
                X_resampled, y_resampled = undersampler.fit_resample(X_combined, labels)
                kept_indices = undersampler.sample_indices_
    else:
        logger.warning(f"Unrecognized balancing method: {method}. Using undersampling.")
        undersampler = RandomUnderSampler(random_state=args.seed)
        X_combined = np.hstack([X_eeg_flat, X_ecg_flat])
        X_resampled, y_resampled = undersampler.fit_resample(X_combined, labels)
        kept_indices = undersampler.sample_indices_
    balanced_eeg = eeg_data[kept_indices]
    balanced_ecg = ecg_data[kept_indices]
    balanced_meta = metadata.iloc[kept_indices].reset_index(drop=True)

    assert len(balanced_eeg) == len(balanced_meta), "EEG and metadata length mismatch after balancing"
    assert len(balanced_ecg) == len(balanced_meta), "ECG and metadata length mismatch after balancing"

    new_class_counts = balanced_meta['epoch_type_class'].value_counts()
    logger.info(f"Balanced class distribution: {new_class_counts.to_dict()}")

    return balanced_eeg, balanced_ecg, balanced_meta


def create_data_splits(eeg_data, ecg_data, metadata, test_size=0.2, val_size=0.15, 
                      stratify=True, test_subjects=None):
    """
    Create train/val/test splits of the data.
    Ensures no data leakage by keeping subjects in the same split.

    Args:
        eeg_data (np.ndarray): EEG data
        ecg_data (np.ndarray): ECG data
        metadata (pd.DataFrame): Metadata
        test_size (float): Proportion of data for testing
        val_size (float): Proportion of training data for validation
        stratify (bool): Whether to use stratified splits
        test_subjects (list): Specific subjects to reserve for testing

    Returns:
        dict: Dictionary containing train/val/test data and metadata
    """
    logger.info("Creating train/val/test splits")

    subjects = metadata['subject_id'].unique()
    logger.info(f"Total subjects: {len(subjects)}")

    y = metadata['epoch_type_class'].values

    if test_subjects is not None:

        logger.info(f"Using predefined test subjects: {test_subjects}")
        test_mask = metadata['subject_id'].isin(test_subjects)
        train_val_mask = ~test_mask

        # get test data
        X_test_eeg = eeg_data[test_mask]
        X_test_ecg = ecg_data[test_mask]
        y_test = y[test_mask]
        meta_test = metadata[test_mask].reset_index(drop=True)

        # train/val data
        X_train_val_eeg = eeg_data[train_val_mask]
        X_train_val_ecg = ecg_data[train_val_mask]
        y_train_val = y[train_val_mask]
        meta_train_val = metadata[train_val_mask].reset_index(drop=True)

        # split remaining subjects into train/val
        train_val_subjects = meta_train_val['subject_id'].unique()

        if stratify:

            subject_class_counts = {}
            for subject in train_val_subjects:
                subj_mask = meta_train_val['subject_id'] == subject
                counts = meta_train_val.loc[subj_mask, 'epoch_type_class'].value_counts().to_dict()
                subject_class_counts[subject] = counts

            subjects_with_seizure = [s for s in train_val_subjects 
                                    if any(c in subject_class_counts[s] 
                                          for c in ['preictal', 'ictal', 'onset'])]
            subjects_without_seizure = [s for s in train_val_subjects if s not in subjects_with_seizure]

            n_val_seizure = max(1, int(len(subjects_with_seizure) * val_size))
            n_val_non_seizure = max(1, int(len(subjects_without_seizure) * val_size))

            val_seizure_subjects = np.random.choice(subjects_with_seizure, n_val_seizure, replace=False)
            val_non_seizure_subjects = np.random.choice(subjects_without_seizure, n_val_non_seizure, replace=False)
            val_subjects = np.concatenate([val_seizure_subjects, val_non_seizure_subjects])

            train_subjects = [s for s in train_val_subjects if s not in val_subjects]
        else:

            n_val_subjects = max(1, int(len(train_val_subjects) * val_size))
            val_subjects = np.random.choice(train_val_subjects, n_val_subjects, replace=False)
            train_subjects = [s for s in train_val_subjects if s not in val_subjects]

        train_mask = meta_train_val['subject_id'].isin(train_subjects)
        val_mask = meta_train_val['subject_id'].isin(val_subjects)

        X_train_eeg = X_train_val_eeg[train_mask]
        X_train_ecg = X_train_val_ecg[train_mask]
        y_train = y_train_val[train_mask]
        meta_train = meta_train_val[train_mask].reset_index(drop=True)

        X_val_eeg = X_train_val_eeg[val_mask]
        X_val_ecg = X_train_val_ecg[val_mask]
        y_val = y_train_val[val_mask]
        meta_val = meta_train_val[val_mask].reset_index(drop=True)

    else:

        logger.info("Using GroupShuffleSplit for subject-wise train/test split")

        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=args.seed)
        train_val_idx, test_idx = next(gss.split(np.zeros(len(metadata)), groups=metadata['subject_id']))

        X_test_eeg = eeg_data[test_idx]
        X_test_ecg = ecg_data[test_idx]
        y_test = y[test_idx]
        meta_test = metadata.iloc[test_idx].reset_index(drop=True)

        X_train_val_eeg = eeg_data[train_val_idx]
        X_train_val_ecg = ecg_data[train_val_idx]
        y_train_val = y[train_val_idx]
        meta_train_val = metadata.iloc[train_val_idx].reset_index(drop=True)

        gss_val = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=args.seed)
        train_idx, val_idx = next(gss_val.split(np.zeros(len(meta_train_val)), groups=meta_train_val['subject_id']))

        X_train_eeg = X_train_val_eeg[train_idx]
        X_train_ecg = X_train_val_ecg[train_idx]
        y_train = y_train_val[train_idx]
        meta_train = meta_train_val.iloc[train_idx].reset_index(drop=True)

        X_val_eeg = X_train_val_eeg[val_idx]
        X_val_ecg = X_train_val_ecg[val_idx]
        y_val = y_train_val[val_idx]
        meta_val = meta_train_val.iloc[val_idx].reset_index(drop=True)

    logger.info(f"Train set: {len(X_train_eeg)} epochs from {meta_train['subject_id'].nunique()} subjects")
    logger.info(f"Val set: {len(X_val_eeg)} epochs from {meta_val['subject_id'].nunique()} subjects")
    logger.info(f"Test set: {len(X_test_eeg)} epochs from {meta_test['subject_id'].nunique()} subjects")

    train_class_dist = meta_train['epoch_type_class'].value_counts().to_dict()
    val_class_dist = meta_val['epoch_type_class'].value_counts().to_dict()
    test_class_dist = meta_test['epoch_type_class'].value_counts().to_dict()

    logger.info(f"Train class distribution: {train_class_dist}")
    logger.info(f"Val class distribution: {val_class_dist}")
    logger.info(f"Test class distribution: {test_class_dist}")

    train_subjects = set(meta_train['subject_id'].unique())
    val_subjects = set(meta_val['subject_id'].unique())
    test_subjects = set(meta_test['subject_id'].unique())

    logger.info(f"Train subjects: {sorted(train_subjects)}")
    logger.info(f"Val subjects: {sorted(val_subjects)}")
    logger.info(f"Test subjects: {sorted(test_subjects)}")

    train_val_overlap = train_subjects.intersection(val_subjects)
    train_test_overlap = train_subjects.intersection(test_subjects)
    val_test_overlap = val_subjects.intersection(test_subjects)

    assert len(train_val_overlap) == 0, f"Train-val subject overlap: {train_val_overlap}"
    assert len(train_test_overlap) == 0, f"Train-test subject overlap: {train_test_overlap}"
    assert len(val_test_overlap) == 0, f"Val-test subject overlap: {val_test_overlap}"

    splits = {
        'train': {
            'eeg': X_train_eeg,
            'ecg': X_train_ecg,
            'y': y_train,
            'metadata': meta_train
        },
        'val': {
            'eeg': X_val_eeg,
            'ecg': X_val_ecg,
            'y': y_val,
            'metadata': meta_val
        },
        'test': {
            'eeg': X_test_eeg,
            'ecg': X_test_ecg,
            'y': y_test,
            'metadata': meta_test
        }
    }

    return splits


def fuse_data(eeg_data, ecg_data, fusion_method='concatenate'):
    """
    Fuse EEG and ECG data for intermediate fusion.

    Args:
        eeg_data (np.ndarray): EEG data with shape (n_epochs, n_channels, n_times)
        ecg_data (np.ndarray): ECG data with shape (n_epochs, n_channels, n_times)
        fusion_method (str): Method for fusion ('concatenate', 'average', etc.)

    Returns:
        np.ndarray: Fused data
    """
    if eeg_data is None or ecg_data is None:
        return None

    logger.info(f"Fusing EEG and ECG data using method: {fusion_method}")

    n_eeg_epochs, n_eeg_ch, n_eeg_times = eeg_data.shape
    n_ecg_epochs, n_ecg_ch, n_ecg_times = ecg_data.shape

    assert n_eeg_epochs == n_ecg_epochs, "Number of epochs must match for fusion"

    if n_eeg_times != n_ecg_times:
        logger.warning(f"Time dimensions don't match: EEG ({n_eeg_times}) vs ECG ({n_ecg_times})")

        if n_eeg_times > n_ecg_times:
            logger.info(f"Resampling ECG from {n_ecg_times} to {n_eeg_times} time points")
            resampled_ecg = np.zeros((n_ecg_epochs, n_ecg_ch, n_eeg_times))
            for i in range(n_ecg_epochs):
                for j in range(n_ecg_ch):
                    resampled_ecg[i, j] = mne.filter.resample(ecg_data[i, j], up=n_eeg_times, down=n_ecg_times)
            ecg_data = resampled_ecg
            n_ecg_times = n_eeg_times
        else:
            logger.info(f"Resampling EEG from {n_eeg_times} to {n_ecg_times} time points")
            resampled_eeg = np.zeros((n_eeg_epochs, n_eeg_ch, n_ecg_times))
            for i in range(n_eeg_epochs):
                for j in range(n_eeg_ch):
                    resampled_eeg[i, j] = mne.filter.resample(eeg_data[i, j], up=n_ecg_times, down=n_eeg_times)
            eeg_data = resampled_eeg
            n_eeg_times = n_ecg_times

    if fusion_method == 'concatenate':
        # Channel-wise concatenation
        fused_data = np.zeros((n_eeg_epochs, n_eeg_ch + n_ecg_ch, n_eeg_times))
        fused_data[:, :n_eeg_ch, :] = eeg_data
        fused_data[:, n_eeg_ch:, :] = ecg_data

    elif fusion_method == 'average':

        fused_data = np.zeros((n_eeg_epochs, max(n_eeg_ch, n_ecg_ch), n_eeg_times))

        for i in range(min(n_eeg_ch, n_ecg_ch)):
            fused_data[:, i, :] = (eeg_data[:, i, :] + ecg_data[:, i, :]) / 2

        if n_eeg_ch > n_ecg_ch:
            fused_data[:, n_ecg_ch:, :] = eeg_data[:, n_ecg_ch:, :]
        elif n_ecg_ch > n_eeg_ch:
            fused_data[:, n_eeg_ch:, :] = ecg_data[:, n_eeg_ch:, :]

    elif fusion_method == 'parallel':
        fused_data = {'eeg': eeg_data, 'ecg': ecg_data}

    else:
        logger.warning(f"Unknown fusion method: {fusion_method}. Using concatenation.")
        fused_data = np.zeros((n_eeg_epochs, n_eeg_ch + n_ecg_ch, n_eeg_times))
        fused_data[:, :n_eeg_ch, :] = eeg_data
        fused_data[:, n_eeg_ch:, :] = ecg_data

    return fused_data


def save_dataset(data_splits, output_path, save_format='npy', dataset_name='eeg_ecg_fusion'):
    """
    Save the processed dataset.

    Args:
        data_splits (dict): Dictionary with train/val/test splits
        output_path (str): Path to save data
        save_format (str): Format to save data ('npy', 'pickle', 'h5')
        dataset_name (str): Name for the dataset
    """
    logger.info(f"Saving dataset in {save_format} format to {output_path}")

    os.makedirs(output_path, exist_ok=True)

    dataset_dir = os.path.join(output_path, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    for split_name, split_data in data_splits.items():
        split_dir = os.path.join(dataset_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        if save_format == 'npy':
            for data_name, data in split_data.items():
                if data_name == 'metadata':

                    csv_path = os.path.join(split_dir, f"{data_name}.csv")
                    data.to_csv(csv_path, index=False)
                else:

                    npy_path = os.path.join(split_dir, f"{data_name}.npy")
                    np.save(npy_path, data)
        elif save_format == 'pickle':

            pickle_path = os.path.join(split_dir, f"{split_name}_data.pkl")
            with open(pickle_path, 'wb') as f:
                pickle.dump(split_data, f)

        elif save_format == 'h5':
            try:
                import h5py

                h5_path = os.path.join(split_dir, f"{split_name}_data.h5")
                with h5py.File(h5_path, 'w') as f:
                    for data_name, data in split_data.items():
                        if data_name == 'metadata':

                            meta_group = f.create_group('metadata')
                            for col in data.columns:
                                col_data = data[col].values

                                if col_data.dtype.kind == 'O':

                                    str_data = np.array(col_data, dtype=h5py.special_dtype(vlen=str))
                                    meta_group.create_dataset(col, data=str_data)
                                else:
                                    meta_group.create_dataset(col, data=col_data)
                        else:

                            f.create_dataset(data_name, data=data)
            except ImportError:
                logger.warning("h5py not available. Falling back to npy format.")

                for data_name, data in split_data.items():
                    if data_name == 'metadata':
                        csv_path = os.path.join(split_dir, f"{data_name}.csv")
                        data.to_csv(csv_path, index=False)
                    else:
                        npy_path = os.path.join(split_dir, f"{data_name}.npy")
                        np.save(npy_path, data)
        else:
            logger.warning(f"Unknown save format: {save_format}. Using npy.")

            for data_name, data in split_data.items():
                if data_name == 'metadata':
                    csv_path = os.path.join(split_dir, f"{data_name}.csv")
                    data.to_csv(csv_path, index=False)
                else:
                    npy_path = os.path.join(split_dir, f"{data_name}.npy")
                    np.save(npy_path, data)

    info = {
        'dataset_name': dataset_name,
        'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'args': vars(args),
        'splits': {
            split: {
                'samples': len(data['metadata']),
                'subjects': data['metadata']['subject_id'].nunique(),
                'class_distribution': data['metadata']['epoch_type_class'].value_counts().to_dict()
            } for split, data in data_splits.items()
        }
    }

    info_path = os.path.join(dataset_dir, 'dataset_info.json')
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
        
    logger.info(f"Dataset saved to {dataset_dir}")
    return dataset_dir


def main():
    """Main function to run the EEG-ECG fusion pipeline."""
    logger.info("Starting EEG-ECG fusion pipeline")
    logger.info(f"Arguments: {vars(args)}")

    os.makedirs(args.output_path, exist_ok=True)

    eeg_files = {}
    ecg_files = {}

    for category in args.categories:
        eeg_files[category] = glob.glob(os.path.join(args.eeg_path, category, "*-epo.fif"))
        ecg_files[category] = glob.glob(os.path.join(args.ecg_path, category, "*-epo.fif"))

        logger.info(f"Category {category}: Found {len(eeg_files[category])} EEG files and {len(ecg_files[category])} ECG files")

    all_merged_data = {}

    for category in args.categories:
        logger.info(f"Processing category: {category}")

        if not eeg_files[category] or not ecg_files[category]:
            logger.warning(f"Skipping category {category}: Missing files")
            continue

        eeg_subjects = set()
        for eeg_file in eeg_files[category]:

            meta_file = eeg_file.replace("-epo.fif", "_metadata.csv")
            if os.path.exists(meta_file):
                try:
                    meta = pd.read_csv(meta_file)
                    if 'subject_id' in meta.columns:
                        eeg_subjects.update(meta['subject_id'].unique())
                except Exception as e:
                    logger.warning(f"Error reading {meta_file}: {e}")

        logger.info(f"Found {len(eeg_subjects)} subjects in category {category}")

        if args.skip_subjects:
            eeg_subjects = eeg_subjects - set(args.skip_subjects)
            logger.info(f"After filtering skip_subjects: {len(eeg_subjects)} subjects")

        if args.max_subjects and len(eeg_subjects) > args.max_subjects:
            eeg_subjects = set(list(eeg_subjects)[:args.max_subjects])
            logger.info(f"Limiting to {len(eeg_subjects)} subjects")

        subject_data = {}

        for subject_id in tqdm(eeg_subjects, desc=f"Processing subjects in {category}"):
            logger.info(f"Processing subject {subject_id}")

            subject_merged_data = {
                'eeg': [],
                'ecg': [],
                'metadata': []
            }

            for epoch_type in args.epoch_types:
                logger.info(f"Processing {subject_id}, {epoch_type}")

                matching_eeg_files = [f for f in eeg_files[category]
                                     if f"{epoch_type}_" in os.path.basename(f)]

                if not matching_eeg_files:
                    logger.warning(f"No matching EEG files for {subject_id}/{epoch_type}")
                    continue

                for eeg_file in matching_eeg_files:
                    meta_file = eeg_file.replace("-epo.fif", "_metadata.csv")
                    if not os.path.exists(meta_file):
                        continue

                    try:
                        eeg_meta = pd.read_csv(meta_file)

                        if 'subject_id' not in eeg_meta.columns or subject_id not in eeg_meta['subject_id'].values:
                            continue

                        try:
                            eeg_epochs = mne.read_epochs(eeg_file, verbose=False)

                            subject_mask = eeg_epochs.metadata['subject_id'] == subject_id
                            if np.any(subject_mask):
                                this_eeg = eeg_epochs[subject_mask]
                                this_meta = eeg_meta[eeg_meta['subject_id'] == subject_id].reset_index(drop=True)

                                ecg_epochs, ecg_meta = find_matching_ecg_epochs(
                                    this_meta, ecg_files[category], subject_id, epoch_type
                                )

                                if ecg_epochs is not None:

                                    matched_eeg, matched_ecg, matched_meta = match_eeg_ecg_epochs(
                                        this_eeg, ecg_epochs, this_meta, ecg_meta
                                    )

                                    if matched_eeg is not None:

                                        eeg_data = matched_eeg.get_data()
                                        ecg_data = matched_ecg.get_data()


                                        subject_merged_data['eeg'].append(eeg_data)
                                        subject_merged_data['ecg'].append(ecg_data)
                                        subject_merged_data['metadata'].append(matched_meta)
                        except Exception as e:
                            logger.warning(f"Error loading EEG file {eeg_file}: {e}")
                            continue
                    except Exception as e:
                        logger.warning(f"Error reading metadata {meta_file}: {e}")
                        continue

            if subject_merged_data['eeg']:
                subject_data[subject_id] = {
                    'eeg': np.vstack(subject_merged_data['eeg']),
                    'ecg': np.vstack(subject_merged_data['ecg']),
                    'metadata': pd.concat(subject_merged_data['metadata'], ignore_index=True)
                }
                logger.info(f"Subject {subject_id}: {len(subject_data[subject_id]['metadata'])} matched epochs")

        if subject_data:
            cat_eeg = np.vstack([data['eeg'] for data in subject_data.values()])
            cat_ecg = np.vstack([data['ecg'] for data in subject_data.values()])
            cat_meta = pd.concat([data['metadata'] for data in subject_data.values()], ignore_index=True)

            logger.info(f"Category {category}: Combined {len(cat_meta)} epochs from {len(subject_data)} subjects")

            if args.normalize:
                logger.info("Applying normalization")
                cat_eeg, cat_ecg, scalers = apply_normalization(
                    cat_eeg, cat_ecg, 
                    method='robust', 
                    per_subject=args.normalize_per_subject,
                    metadata=cat_meta
                )

                scaler_dir = os.path.join(args.output_path, 'scalers', category)
                os.makedirs(scaler_dir, exist_ok=True)
                with open(os.path.join(scaler_dir, 'scalers.pkl'), 'wb') as f:
                    pickle.dump(scalers, f)

            if args.augment:
                logger.info("Applying data augmentation")
                cat_eeg, cat_ecg, cat_meta = augment_data(
                    cat_eeg, cat_ecg, cat_meta, 
                    augment_factor=args.augment_factor,
                    skip_non_seizure=args.skip_augmentation_for_non_seizure
                )

            if args.balance_classes:
                logger.info("Balancing classes")
                cat_eeg, cat_ecg, cat_meta = balance_classes(
                    cat_eeg, cat_ecg, cat_meta,
                    method=args.balance_method
                )

            all_merged_data[category] = {
                'eeg': cat_eeg,
                'ecg': cat_ecg,
                'metadata': cat_meta
            }

    for category, data in all_merged_data.items():
        logger.info(f"Creating splits for category: {category}")

        splits = create_data_splits(
            data['eeg'], data['ecg'], data['metadata'],
            test_size=args.test_split,
            val_size=args.val_split,
            stratify=args.stratify,
            test_subjects=args.test_subjects
        )

        if not args.no_fusion:
            logger.info("Creating fused versions of data")
            for split_name, split_data in splits.items():
                split_data['fused'] = fuse_data(
                    split_data['eeg'],
                    split_data['ecg'],
                    fusion_method='concatenate'
                )

        dataset_name = f"eeg_ecg_{category}"
        dataset_dir = save_dataset(
            splits,
            args.output_path,
            save_format=args.save_format,
            dataset_name=dataset_name
        )

        summary_file = os.path.join(dataset_dir, 'dataset_summary.txt')
        with open(summary_file, 'w') as f:
            f.write(f"EEG-ECG Fusion Dataset: {dataset_name}\n")
            f.write(f"Created at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("Dataset Overview:\n")
            for split_name, split_data in splits.items():
                f.write(f"\n{split_name.capitalize()} Set:\n")
                f.write(f"  Samples: {len(split_data['metadata'])}\n")
                f.write(f"  Subjects: {split_data['metadata']['subject_id'].nunique()}\n")
                f.write(f"  Class distribution:\n")
                for cls, count in split_data['metadata']['epoch_type_class'].value_counts().items():
                    f.write(f"    {cls}: {count} ({count/len(split_data['metadata'])*100:.1f}%)\n")

                f.write(f"  EEG shape: {split_data['eeg'].shape}\n")
                f.write(f"  ECG shape: {split_data['ecg'].shape}\n")
                if 'fused' in split_data:
                    f.write(f"  Fused shape: {split_data['fused'].shape}\n")

            f.write("\nProcessing Steps Applied:\n")
            if args.normalize:
                f.write(f"  • Normalization: {'per-subject' if args.normalize_per_subject else 'global'}\n")
            if args.augment:
                f.write(f"  • Data Augmentation: factor={args.augment_factor}\n")
            if args.balance_classes:
                f.write(f"  • Class Balancing: method={args.balance_method}\n")
            f.write(f"  • Train/Val/Test Split: {(1-args.test_split-args.val_split):.0%}/{args.val_split:.0%}/{args.test_split:.0%}\n")
            if not args.no_fusion:
                f.write("  • Modality Fusion: channel concatenation\n")

        logger.info(f"Dataset summary saved to {summary_file}")

    logger.info("EEG-ECG fusion pipeline completed successfully")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"Error in main execution: {e}")
        sys.exit(1)
