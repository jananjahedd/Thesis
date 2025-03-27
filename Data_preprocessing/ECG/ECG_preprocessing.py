import os
import glob
import numpy as np
import mne
import pandas as pd
import matplotlib.pyplot as plt

# --- Parameters ---
base_path = "/Users/jananjahed/Desktop/Bachelor's project/ds005873"
subjects = ["sub-001", "sub-002"]
tmin, tmax = 0, 25  # seconds
epoch_length = int((tmax - tmin) * 256)
step_size = int(5 * 256)
sfreq = 256
save_path = os.path.join(base_path, "ecg_preprocessed")
os.makedirs(save_path, exist_ok=True)

# --- Helper functions ---
def overlapping_windows(data, window_size, step_size):
    windows = []
    indices = []
    for start in range(0, data.shape[1] - window_size + 1, step_size):
        window = data[:, start:start+window_size]
        if window.shape[1] == window_size:
            windows.append(window)
            indices.append(start)
    return windows, indices

def zscore(signal):
    mean = np.mean(signal, axis=-1, keepdims=True)
    std = np.std(signal, axis=-1, keepdims=True)
    return (signal - mean) / std

# --- Storage ---
all_windows = []
all_labels = []

# --- Process each ECG EDF file ---
for subj in subjects:
    ecg_folder = os.path.join(base_path, subj, "ses-01", "ecg")
    edf_files = glob.glob(os.path.join(ecg_folder, f"{subj}_ses-01_task-szMonitoring_run-*_ecg.edf"))

    for edf_file in edf_files:
        print(f"\n📁 Processing {os.path.basename(edf_file)}")
        try:
            raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
            print("Available channels:", raw.ch_names)
            raw.pick(picks=raw.ch_names)  # Pick all available ECG channels
        except Exception as e:
            print(f"Could not load {edf_file}: {e}")
            continue

        # --- HRV-optimized filtering ---
        raw.filter(l_freq=1.0, h_freq=25.0, fir_design='firwin')  # Adjusted bandpass for better R-peak detection
        raw.notch_filter(freqs=50)
        raw.resample(sfreq)

        # --- Normalization & Windowing ---
        ecg_data = raw.get_data()
        norm_data = zscore(ecg_data)
        windows, indices = overlapping_windows(norm_data, epoch_length, step_size)

        # --- Label using EEG events ---
        eeg_events_path = edf_file.replace("/ecg/", "/eeg/").replace("_ecg.edf", "_events.tsv")
        if not os.path.exists(eeg_events_path):
            print("⚠️ No EEG events file found; skipping labeling for this run.")
            continue

        try:
            events_df = pd.read_csv(eeg_events_path, sep="\t")
            seizure_events = events_df[events_df['eventType'].str.startswith("sz_", na=False)]
        except Exception as e:
            print(f"⚠️ Could not read events TSV: {e}")
            continue

        labels = []
        for start_idx in indices:
            win_start_sec = start_idx / sfreq
            win_end_sec = (start_idx + epoch_length) / sfreq
            is_seizure = False

            for _, row in seizure_events.iterrows():
                sz_onset = row['onset']
                sz_end = sz_onset + row['duration']
                if (win_start_sec < sz_end) and (win_end_sec > sz_onset):
                    is_seizure = True
                    break

            labels.append(1 if is_seizure else 0)

        all_windows.extend(windows)
        all_labels.extend(labels)

print(f"\n Extracted {len(all_windows)} ECG windows.")

# --- Save data ---
X = np.array(all_windows)
y = np.array(all_labels)

np.save(os.path.join(save_path, "X_ecg.npy"), X)
np.save(os.path.join(save_path, "y_ecg_labels.npy"), y)

# --- Summary + Plot ---
import random
print("\n ECG Preprocessing Complete!")
print(f"Total ECG windows saved: {len(all_windows)}")
if len(X) > 0:
    print(f"Shape of each window: {X[0].shape} (channels, samples)")
    print(f"Label distribution: {np.bincount(y)}")

    # Plot a few random windows
    plot_samples = min(5, len(X))
    indices = random.sample(range(len(X)), plot_samples)
    fig, axs = plt.subplots(plot_samples, 1, figsize=(10, 8), sharex=True)

    for i, idx in enumerate(indices):
        axs[i].plot(X[idx][0])
        axs[i].set_title(f"ECG Window #{idx} - Label: {y[idx]}")
        axs[i].set_ylabel("Z-scored Amplitude")
    axs[-1].set_xlabel("Time (samples)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "sample_ecg_windows.png"))
    plt.close()

    print(f"Saved plot of {plot_samples} ECG windows to: {os.path.join(save_path, 'sample_ecg_windows.png')}")
