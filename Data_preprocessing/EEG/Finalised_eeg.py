import os
import numpy as np
import mne
import matplotlib.pyplot as plt
from sklearn.utils import resample

# --- Parameters ---
data_path = "/Users/jananjahed/Desktop/Bachelor's project/ds005873"
epoch_sz_path = os.path.join(data_path, 'combined_seizure_epochs-epo.fif')
epoch_nsz_path = os.path.join(data_path, 'combined_nonseizure_epochs-epo.fif')
out_path = os.path.join(data_path, 'DL_ready')
os.makedirs(out_path, exist_ok=True)

# --- Load Epochs ---
epochs_sz = mne.read_epochs(epoch_sz_path, preload=True)
epochs_nsz = mne.read_epochs(epoch_nsz_path, preload=True)

# --- Overlapping Window Extraction ---
def overlapping_windows(data, window_size, step_size):
    n_epochs, n_channels, n_samples = data.shape
    new_epochs = []
    for i in range(n_epochs):
        for start in range(0, n_samples - window_size + 1, step_size):
            segment = data[i, :, start:start+window_size]
            new_epochs.append(segment)
    return np.array(new_epochs)

# --- Z-score Normalization (baseline-based) ---
def zscore_epochs(epochs, baseline_samples=1280):  # 5s at 256Hz
    data = epochs.get_data()
    baseline = data[:, :, :baseline_samples]
    mean = baseline.mean(axis=-1, keepdims=True)
    std = baseline.std(axis=-1, keepdims=True)
    norm_data = (data - mean) / std
    return norm_data

X_sz = zscore_epochs(epochs_sz)
X_nsz = zscore_epochs(epochs_nsz)
y_sz = np.ones(len(X_sz))
y_nsz = np.zeros(len(X_nsz))

# --- Apply overlapping window slicing ---
window_size = 6400  # 25 seconds at 256Hz
step_size = 1280    # 5 seconds
X_sz_win = overlapping_windows(X_sz, window_size, step_size)
X_nsz_win = overlapping_windows(X_nsz, window_size, step_size)
y_sz_win = np.ones(len(X_sz_win))
y_nsz_win = np.zeros(len(X_nsz_win))

# --- Data Augmentation ---
def add_gaussian_noise(data, noise_level=0.05):
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise

def fourier_surrogate(data):
    surrogate = []
    for trial in data:
        trial_fft = np.fft.rfft(trial, axis=-1)
        amplitude = np.abs(trial_fft)
        phase = np.angle(trial_fft)
        new_phase = np.random.uniform(-np.pi, np.pi, phase.shape)
        new_fft = amplitude * np.exp(1j * new_phase)
        new_data = np.fft.irfft(new_fft, n=trial.shape[-1], axis=-1)
        surrogate.append(new_data)
    return np.array(surrogate)

X_sz_aug = np.concatenate([X_sz_win, add_gaussian_noise(X_sz_win), fourier_surrogate(X_sz_win)], axis=0)
y_sz_aug = np.concatenate([y_sz_win]*3)

X_nsz_aug = np.concatenate([X_nsz_win, add_gaussian_noise(X_nsz_win)], axis=0)
y_nsz_aug = np.concatenate([y_nsz_win]*2)

# --- Balance Classes ---
if len(X_nsz_aug) > len(X_sz_aug):
    X_sz_aug, y_sz_aug = resample(X_sz_aug, y_sz_aug, replace=True, n_samples=len(X_nsz_aug), random_state=42)
elif len(X_sz_aug) > len(X_nsz_aug):
    X_nsz_aug, y_nsz_aug = resample(X_nsz_aug, y_nsz_aug, replace=True, n_samples=len(X_sz_aug), random_state=42)

# --- Combine and Shuffle ---
X = np.concatenate([X_sz_aug, X_nsz_aug], axis=0)
y = np.concatenate([y_sz_aug, y_nsz_aug], axis=0)

perm = np.random.permutation(len(X))
X, y = X[perm], y[perm]

# --- Save numpy arrays ---
np.save(os.path.join(out_path, 'X_eeg.npy'), X)
np.save(os.path.join(out_path, 'y_labels.npy'), y)

print(f"Saved normalized, windowed, and augmented dataset to: {out_path}")
print(f"Shape X: {X.shape}, Shape y: {y.shape}")

# --- Plot example epoch ---
plt.figure(figsize=(10, 4))
plt.plot(X[0][0])
plt.title(f"Example normalized EEG epoch (Label: {int(y[0])})")
plt.xlabel("Time (samples)")
plt.ylabel("Z-scored Amplitude")
plt.tight_layout()
plt.savefig(os.path.join(out_path, "sample_epoch.png"))
plt.close()