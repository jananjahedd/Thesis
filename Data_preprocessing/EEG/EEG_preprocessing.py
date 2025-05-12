import mne
import pandas as pd
import numpy as np
import os
import glob
import logging
from tqdm import tqdm
from autoreject import AutoReject
from itertools import zip_longest

logging.basicConfig(
    filename='preprocessing_log.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

base_path = "/Users/jananjahed/Desktop/BP/ds005873"
desired_channels = ['BTEleft SD', 'CROSStop SD']
sfreq = 256
buffer_sec = 5

epoch_config = {
    'preictal': (-10, 0),
    'ictal':    (0, 10),
    'onset':    (-5, 5),
    'non_seizure': (0, 10)
}

times_reference = {key: None for key in epoch_config}
containers = {key: [] for key in epoch_config}


def fallback_rejection(epochs, threshold_uV=150, min_samples=None):
    data = epochs.get_data()
    bad_amp = np.any(np.abs(data) > threshold_uV * 1e-6, axis=(1, 2))
    var = np.var(data, axis=(1, 2))
    m, s = np.mean(var), np.std(var)
    bad_var = np.abs(var - m) > 3 * s
    bad = bad_amp | bad_var

    keep = ~bad
    if min_samples is not None and keep.sum() < min_samples:
        idx = np.argsort(np.abs(var - m))[:min_samples]
        keep = np.zeros_like(keep)
        keep[idx] = True

    logger.info(f"Fallback rejection: {keep.sum()}/{len(keep)} epochs kept.")
    return epochs[keep]


def create_epoch(raw, onset_time, tmin, tmax, key):
    start_samp = int((onset_time + tmin) * sfreq)
    end_samp = int((onset_time + tmax) * sfreq)
    if start_samp < 0 or end_samp > raw.n_times:
        logger.warning(f"Epoch ({key}) out-of-bounds [{start_samp},"
                       f"{end_samp}].")
        return None

    try:
        events = np.array([[int(onset_time * sfreq), 0, 1]])
        ep = mne.Epochs(raw, events, tmin=tmin, tmax=tmax,
                        baseline=None, preload=True, verbose=False)
        if times_reference[key] is None:
            times_reference[key] = ep.times
        elif not np.allclose(times_reference[key], ep.times):
            raise ValueError("Time mismatch across epochs.")
        return ep
    except Exception as e:
        logger.warning(f"create_epoch failed ({key}): {e}")
        return None


def process_non_seizure(raw, sz_intervals, sample_limit=None):
    mask = np.zeros(raw.n_times, dtype=bool)
    for start, end in sz_intervals:
        buf_start = max(0, start - buffer_sec * sfreq)
        buf_end = min(raw.n_times, end + buffer_sec * sfreq)
        mask[buf_start:buf_end] = True

    diff = np.diff(mask.astype(int))
    seg_starts = (np.where(diff == -1)[0] + 1).tolist()
    seg_ends = (np.where(diff == 1)[0] + 1).tolist()
    if not mask[0]:
        seg_starts.insert(0, 0)
    if not mask[-1]:
        seg_ends.append(raw.n_times)

    n_samps = int((epoch_config['non_seizure'][1] -
                   epoch_config['non_seizure'][0]) * sfreq)
    events = []

    for st, en in zip_longest(seg_starts, seg_ends, fillvalue=None):
        if st is None or en is None:
            continue
        n_epochs = (en - st) // n_samps
        for i in range(n_epochs):
            s = st + i * n_samps
            e = s + n_samps
            if e > en or mask[s:e].any():
                continue
            events.append([s, 0, 2])
            if sample_limit and len(events) >= sample_limit:
                break
        if sample_limit and len(events) >= sample_limit:
            break

    if events:
        return mne.Epochs(raw, np.array(events),
                          tmin=epoch_config['non_seizure'][0],
                          tmax=epoch_config['non_seizure'][1],
                          baseline=None, preload=True, verbose=False)
    return None


for edf_path in tqdm(glob.glob(os.path.join(base_path, "sub-*", "ses-*", "eeg",
                                            "*_eeg.edf")), desc="Processing"):

    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True)
        missing = [ch for ch in desired_channels if ch not in raw.ch_names]
        if missing:
            logger.error(f"Missing channels {missing}; "
                         f"skipping {os.path.basename(edf_path)}.")
            continue
        raw.pick_channels(desired_channels)
        raw.set_montage('standard_1020', on_missing='ignore')
        raw.set_eeg_reference('average')

        raw.filter(0.5, 40)
        raw.notch_filter(50)
        raw.resample(sfreq)

        tsv = edf_path.replace('_eeg.edf', '_events.tsv')
        df = pd.read_csv(tsv, sep='\t')
        sz_intervals = [
            (int(r.onset * sfreq), int((r.onset + r.duration) * sfreq))
            for _, r in df[df.eventType.str.startswith('sz_')].iterrows()
        ]

        for onset, _ in [(r.onset, r.duration) for _, r in df.iterrows()
                         if r.eventType.startswith('sz_')]:
            for key in ['preictal', 'ictal', 'onset']:
                tmin, tmax = epoch_config[key]
                ep = create_epoch(raw, onset, tmin, tmax, key)
                if ep is not None:
                    containers[key].append(ep)

        # non‑seizure epochs
        non_ep = process_non_seizure(raw, sz_intervals, sample_limit=500)
        if non_ep is not None:
            containers['non_seizure'].append(non_ep)

        logger.info(f"Processed {os.path.basename(edf_path)}: "
                    f"seizures={len(sz_intervals)},"
                    f"non_seizure={len(non_ep) if non_ep else 0}")
    except Exception as e:
        logger.error(f"Error processing {os.path.basename(edf_path)}: {e}")


for key, eps_list in containers.items():
    if not eps_list:
        continue
    try:
        all_epochs = mne.concatenate_epochs(eps_list)
        logger.info(f"Combined {len(all_epochs)} {key} epochs.")

        # subsample interictal to balance with seizure counts
        if key == 'non_seizure':
            sz_counts = [len(mne.concatenate_epochs(containers[k]))
                         for k in ['preictal', 'ictal'] if containers[k]]
            if sz_counts:
                target = min(len(all_epochs), 10 * max(sz_counts))
                if len(all_epochs) > target:
                    idx = np.random.choice(len(all_epochs), target,
                                           replace=False)
                    all_epochs = all_epochs[idx]
                    logger.info(f"Subsampled non_seizure to {target} epochs.")

        # artifact rejection
        try:
            ar = AutoReject()
            clean_epochs = ar.fit_transform(all_epochs)
            logger.info(f"AutoReject {len(clean_epochs)}/{len(all_epochs)} "
                        f"{key} epochs.")
        except Exception:
            clean_epochs = fallback_rejection(all_epochs, min_samples=20
                                              if key != 'non_seizure' else 100)

        out_path = os.path.join(base_path, f"{key}_epochs-clean-epo.fif")
        clean_epochs.save(out_path, overwrite=True)
        logger.info(f"Saved cleaned {key} epochs to {out_path}.")
    except Exception as e:
        logger.error(f"Saving {key} failed: {e}")

logger.info("Preprocessing complete with robust rejection.")