import mne
import os
import numpy as np
import pandas as pd

def verify_epoch_time_alignment(eeg_path, ecg_path):
    """
    Verifies the time alignment between EEG and ECG epochs.

    Args:
        eeg_path (str): Path to the EEG epochs file (*-epo.fif).
        ecg_path (str): Path to the ECG epochs file (*-ecg-clean-epo.fif).
    """

    try:
        eeg_epochs = mne.read_epochs(eeg_path, preload=False)
        ecg_epochs = mne.read_epochs(ecg_path, preload=False)
    except FileNotFoundError:
        print(f"Error: One or both files not found. EEG: {eeg_path}, ECG: {ecg_path}")
        return

    eeg_times = eeg_epochs.times
    ecg_times = ecg_epochs.times

    # 1. Verify Time Vector Consistency
    if not np.allclose(eeg_times, ecg_times):
        print(f"WARNING: Time vector mismatch between {eeg_path} and {ecg_path}!")
        print(f"  - EEG times shape: {eeg_times.shape}")
        print(f"  - ECG times shape: {ecg_times.shape}")
        print(f"  - Max time difference: {np.max(np.abs(eeg_times - ecg_times))}")
    else:
        print(f"Time vectors are consistent between {eeg_path} and {ecg_path}.")

    # 2. Metadata Comparison (Optional)
    if hasattr(eeg_epochs, 'metadata') and eeg_epochs.metadata is not None and \
       hasattr(ecg_epochs, 'metadata') and ecg_epochs.metadata is not None:
        if 'onset' in eeg_epochs.metadata.columns and 'onset' in ecg_epochs.metadata.columns:
            eeg_onset = eeg_epochs.metadata['onset'].values
            ecg_onset = ecg_epochs.metadata['onset'].values
            if not np.allclose(eeg_onset, ecg_onset):
                print(f"WARNING: Onset time mismatch between {eeg_path} and {ecg_path}!")
                print(f"  - Max onset difference: {np.max(np.abs(eeg_onset - ecg_onset))}")
            else:
                print("Onset times in metadata are consistent.")
        else:
            print("Onset times not found in metadata. Skipping comparison.")
    else:
        print("Metadata not found in one or both files. Skipping onset comparison.")

if __name__ == '__main__':
    # Example Usage
    base_path = "/scratch/s5107318/BP/ds005873"  # Replace with your base path

    for epoch_type in ['preictal', 'ictal', 'onset', 'non_seizure']:
        eeg_file = os.path.join(base_path, f"{epoch_type}_epochs-clean-epo.fif")
        ecg_file = os.path.join(base_path, f"{epoch_type}_ecg-clean-epo.fif")

        if os.path.exists(eeg_file) and os.path.exists(ecg_file):
            print(f"\n--- Verifying {epoch_type} ---")
            verify_epoch_time_alignment(eeg_file, ecg_file)
        else:
            print(f"\n--- Skipping {epoch_type} ---")
            print(f"  - EEG file exists: {os.path.exists(eeg_file)}")
            print(f"  - ECG file exists: {os.path.exists(ecg_file)}")