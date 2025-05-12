import argparse
import os
import numpy as np
import pandas as pd
import mne
import gc
from tqdm import tqdm
import time

np.random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('--base', default="/scratch/s5107318/BP/ds005873")
parser.add_argument('--ratio', type=float, default=1.0,
                   help='target minority/majority size (<=1)')
parser.add_argument('--no-ecg', action='store_true', help='ignore ECG modality')
parser.add_argument('--outdir', default='dl_ready')
parser.add_argument('--batch-size', type=int, default=50,
                   help='process epochs in batches to reduce memory usage')
args = parser.parse_args()

EPOCH_TYPES = ['preictal', 'ictal', 'onset', 'non_seizure']
MINORITY = ['preictal', 'ictal', 'onset']

eos_path = lambda typ: os.path.join(args.base, f"{typ}_epochs-clean-epo.fif")
ecg_path = lambda typ: os.path.join(args.base, f"{typ}_ecg-clean-epo.fif")

# Function to Extract Epoch Identifiers
def get_epoch_id(epoch, typ, i):
    return f"{typ}_{i}"


def aug_eeg(x, sf):
    # x shape C×T
    # 1/f noise (optimized)
    freqs = np.fft.rfftfreq(x.shape[1], 1 / sf)
    amp = 1 / np.maximum(freqs, 0.1)
    noise_shape = x.shape

    # Generate noise directly with amplitude
    noise = np.random.randn(*noise_shape).astype('float32')
    noise_fft = np.fft.rfft(noise)
    noise_fft *= amp
    noise = np.fft.irfft(noise_fft, n=noise_shape[1])

    x = x + 3e-6 * noise

    # time-shift (in-place)
    shift = np.random.randint(0, x.shape[1])
    x = np.roll(x, shift, axis=1)

    # channel dropout
    if np.random.rand() < 0.1:
        x[np.random.randint(x.shape[0])] = 0

    # polarity flip
    if np.random.rand() < 0.5:
        x *= -1

    return x

def aug_ecg(ecg, sf, target_len=2561):
    t = np.arange(ecg.size) / sf
    # Add sine wave directly instead of creating temporary arrays
    phase = 2 * np.pi * np.random.rand()
    ecg = ecg + 0.05 * np.sin(2 * np.pi * 0.15 * t + phase)

    # More efficient resampling
    factor = np.random.uniform(0.97, 1.03)
    resampled_ecg = mne.filter.resample(ecg, int(ecg.size * factor), n_jobs=1)

    # Handle target length
    if resampled_ecg.size > target_len:
        ecg = resampled_ecg[:target_len]
    elif resampled_ecg.size < target_len:
        ecg = np.pad(resampled_ecg, (0, target_len - resampled_ecg.size), 'constant')
    else:
        ecg = resampled_ecg

    # Add noise and scale
    std_ecg = np.std(ecg)
    ecg += np.random.normal(0, 0.01 * std_ecg, size=ecg.shape)
    ecg *= np.random.uniform(0.8, 1.2)

    return ecg.reshape(1, -1)

# Generate dummy ECG data when real data is unavailable
def generate_dummy_ecg(target_len=2561):
    # Create a simple placeholder ECG with some noise
    dummy = np.zeros((1, target_len), dtype='float32')
    # Add very small random noise to avoid all-zero data
    dummy += np.random.normal(0, 1e-5, size=dummy.shape)
    return dummy


def process_in_batches(output_dir, label_map, counts, target_per_min, major):
    # Initialize output arrays and metadata
    Xeeg_file = os.path.join(output_dir, 'Xeeg.npy')
    Xecg_file = os.path.join(output_dir, 'Xecg.npy')
    y_file = os.path.join(output_dir, 'y.npy')
    meta_file = os.path.join(output_dir, 'meta.csv')

    all_Xeeg = []
    all_Xecg = []
    all_y = []
    meta_rows = []

    print("\n--- Processing Epoch Types ---")
    for typ in EPOCH_TYPES:
        if counts[typ] == 0:
            print(f" - Skipping {typ}: No epochs found.")
            continue

        print(f" - Processing {typ} epochs:")
        start_time = time.time()

        eeg_info = mne.read_epochs(eos_path(typ), preload=False)
        sf = eeg_info.info['sfreq']
        n_orig = len(eeg_info)
        target_count = major if typ == 'non_seizure' else target_per_min
        rep_factor = int(np.ceil(target_count / n_orig)) if n_orig > 0 else 1

        batch_size = min(args.batch_size, n_orig)

        for batch_start in range(0, n_orig, batch_size):
            batch_end = min(batch_start + batch_size, n_orig)
            batch_indices = list(range(batch_start, batch_end))

            try:
                eeg_batch = mne.read_epochs(eos_path(typ), preload=True)[batch_indices]
            except Exception as e:
                print(f" - Error loading EEG batch: {e}")
                continue

            has_ecg = not args.no_ecg and os.path.exists(ecg_path(typ))
            ecg_batch = None
            if has_ecg:
                try:
                    ecg_batch = mne.read_epochs(ecg_path(typ), preload=True)[batch_indices]
                except Exception as e:
                    print(f" - Error loading ECG batch: {e}")

            for i in tqdm(range(len(eeg_batch)), desc=f" - Batch {batch_start//batch_size+1}", leave=False):
                for r in range(rep_factor):
                    if len(all_y) >= target_count:
                        break

                    eeg = eeg_batch[i].get_data().squeeze()

                    ecg = None
                    if has_ecg and ecg_batch is not None and i < len(ecg_batch):
                        ecg = ecg_batch[i].get_data().squeeze()

                    if typ in MINORITY:
                        eeg = aug_eeg(eeg, sf)
                        if ecg is not None:
                            ecg = aug_ecg(ecg, sf)

                    all_Xeeg.append(eeg)
                    if not args.no_ecg:
                        if ecg is not None:
                            all_Xecg.append(aug_ecg(ecg, sf) if ecg.ndim == 1 else ecg)
                        else:
                            all_Xecg.append(generate_dummy_ecg())
                    else:
                        all_Xecg.append(np.zeros((1, 2561), dtype='float32'))

                    all_y.append(label_map[typ])
                    meta_rows.append({
                        "epoch_type": typ,
                        "orig_idx": batch_start + i,
                        "replication": r + 1,
                        "file": eeg_info.filename,
                        "has_ecg": ecg is not None
                    })
                if len(all_y) >= target_count:
                    break

            del eeg_batch
            if ecg_batch is not None:
                del ecg_batch
            gc.collect()

        print(f" - Finished processing {typ} epochs. Total Time: {time.time() - start_time:.2f}s\n")

    np.save(Xeeg_file, np.array(all_Xeeg, dtype='float32'), allow_pickle=False)
    np.save(Xecg_file, np.array(all_Xecg, dtype='float32'), allow_pickle=False)
    np.save(y_file, np.array(all_y, dtype='int64'), allow_pickle=False)
    pd.DataFrame(meta_rows).to_csv(meta_file, index=False)

    return len(all_y)

def main():
    # Create output directory
    output_dir = os.path.join(args.base, args.outdir)
    os.makedirs(output_dir, exist_ok=True)

    # Map labels to indices
    label_map = {k: i for i, k in enumerate(EPOCH_TYPES)}

    # First pass: count epochs per class
    counts = {}
    print("--- Counting Epochs ---")
    for typ in EPOCH_TYPES:
        print(f" - Counting {typ} epochs...", end=" ", flush=True)
        start_time = time.time()
        eeg_file_path = eos_path(typ)
        if os.path.exists(eeg_file_path):
            try:
                epochs = mne.read_epochs(eeg_file_path, preload=False)
                counts[typ] = len(epochs)
            except Exception as e:
                print(f"\nError counting {typ} epochs: {e}")
                counts[typ] = 0
        else:
            counts[typ] = 0
        print(f"Count: {counts[typ]}, Time: {time.time() - start_time:.2f}s")

    # Calculate target counts
    major = counts['non_seizure']
    target_per_min = int(major * args.ratio)
    print("Target per minority class:", target_per_min)

    # Process data in batches to manage memory
    total_processed = process_in_batches(output_dir, label_map, counts, target_per_min, major)

    print(f"\n--- Data Processing Complete: {total_processed} total samples ---")

if __name__ == "__main__":
    # Set MNE to be more memory efficient
    mne.set_log_level('WARNING')  # Reduce log verbosity

    # Run the main function
    main()
