import mne
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from scipy import stats
from mne.time_frequency import psd_array_welch

base_path   = "/Users/jananjahed/Desktop/Bachelor's project/ds005873"
for epoch_type in ['preictal', 'ictal', 'onset', 'non_seizure']:
    eeg_file = os.path.join(base_path, f"{epoch_type}_epochs-clean-epo.fif")
    if not os.path.exists(eeg_file):
        continue

    epochs = mne.read_epochs(eeg_file, preload=True)
    features = []

    for i in tqdm(range(len(epochs)), desc=f"Processing {epoch_type}"):
        epoch = epochs[i]
        data = epoch.get_data()[0]
        times = epoch.times

        meta = {
            'epoch_type': epoch_type,
            'epoch_idx': i,
            'file': epochs.metadata['file'][i] if hasattr(epochs, 'metadata') and epochs.metadata is not None and 'file' in epochs.metadata else '',
            'onset': epochs.metadata['onset'][i] if hasattr(epochs, 'metadata') and epochs.metadata is not None and 'onset' in epochs.metadata else 0,
        }

        # Features for each channel
        for ch_idx, ch_name in enumerate(epochs.ch_names):
            ch_data = data[ch_idx, :]

            # Time domain features
            time_feats = {
                f'{ch_name}_mean': np.mean(ch_data),
                f'{ch_name}_std': np.std(ch_data),
                f'{ch_name}_skew': stats.skew(ch_data),
                f'{ch_name}_kurtosis': stats.kurtosis(ch_data),
                f'{ch_name}_min': np.min(ch_data),
                f'{ch_name}_max': np.max(ch_data),
                f'{ch_name}_ptp': np.ptp(ch_data),  # peak-to-peak
                f'{ch_name}_var': np.var(ch_data),
                f'{ch_name}_rms': np.sqrt(np.mean(ch_data**2)),  # root mean square
            }

            # Hjorth parameters
            diff1 = np.diff(ch_data)
            diff2 = np.diff(diff1)

            hjorth_feats = {
                f'{ch_name}_hjorth_activity': np.var(ch_data),
                f'{ch_name}_hjorth_mobility': np.sqrt(np.var(diff1) / np.var(ch_data)),
                f'{ch_name}_hjorth_complexity': np.sqrt(np.var(diff2) / np.var(diff1)) /
                                                np.sqrt(np.var(diff1) / np.var(ch_data))
            }

            # Frequency domain features (using MNE's built-in PSD function)
            sfreq = epochs.info['sfreq']
            psd, freqs = psd_array_welch(ch_data, sfreq=sfreq, fmin=0.5, fmax=40, n_fft=256)

            # Define frequency bands
            bands = {
                'delta': (0.5, 4),
                'theta': (4, 8),
                'alpha': (8, 13),
                'beta': (13, 30),
                'gamma': (30, 40)
            }

            # Calculate band powers
            band_powers = {}
            for band, (fmin, fmax) in bands.items():
                idx = np.logical_and(freqs >= fmin, freqs <= fmax)
                band_powers[f'{ch_name}_{band}_power'] = np.mean(psd[idx]) if np.any(idx) else 0

            # Spectral entropy
            psd_norm = psd / np.sum(psd)
            spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-16))

            freq_feats = {
                **band_powers,
                f'{ch_name}_spectral_entropy': spectral_entropy,
                f'{ch_name}_peak_freq': freqs[np.argmax(psd)],
                f'{ch_name}_mean_freq': np.sum(freqs * psd) / np.sum(psd),
            }

            # Combine all features for this channel
            channel_feats = {**time_feats, **hjorth_feats, **freq_feats}
            meta.update(channel_feats)

        features.append(meta)

    df = pd.DataFrame(features)
    df.to_csv(os.path.join(base_path, f"{epoch_type}_eeg-features.csv"), index=False)
    print(f"Saved {len(df)} records to {epoch_type}_eeg-features.csv")
