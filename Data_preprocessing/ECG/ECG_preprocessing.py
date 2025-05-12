import mne
import pandas as pd
import numpy as np
import os
import glob
import logging
from tqdm import tqdm
import neurokit2 as nk
from itertools import zip_longest
from autoreject import AutoReject
from joblib import Parallel, delayed

base_path = "/Users/jananjahed/Desktop/BP/ds005873"
sfreq = 256
buffer_sec = 5
np.random.seed(42)

epoch_config = {
    "preictal":    (-10, 0),
    "ictal":       (0, 10),
    "onset":       (-5, 5),
    "non_seizure": (0, 10),
}

logging.basicConfig(
    filename="ecg_preprocessing_log.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()
logger.info("==== ECG preprocessing started ====")

times_ref = {k: None for k in epoch_config}
containers = {k: [] for k in epoch_config}
feature_containers = {k: [] for k in epoch_config}

def create_epoch(raw, onset, tmin, tmax, key):
    beg = int(onset * sfreq + tmin * sfreq)
    end = int(onset * sfreq + tmax * sfreq)
    if beg < 0 or end > raw.n_times:
        return None
    ep = mne.Epochs(raw, np.array([[int(onset * sfreq), 0, 1]]),
                    tmin=tmin, tmax=tmax, baseline=None,
                    picks="all", preload=True, verbose=False)
    if times_ref[key] is None:
        times_ref[key] = ep.times
    elif not np.allclose(times_ref[key], ep.times):
        raise ValueError("times mismatch")
    return ep

def process_non_seizure(raw, sz_intervals, *, sample_limit=None):
    mask = np.zeros(raw.n_times, bool)
    for s, e in sz_intervals:
        mask[max(0, s - buffer_sec * sfreq):min(raw.n_times, e + buffer_sec * sfreq)] = True
    diff = np.diff(mask.astype(int))
    starts = (np.where(diff == -1)[0] + 1).tolist(); ends = (np.where(diff == 1)[0] + 1).tolist()
    if not mask[0]: starts.insert(0, 0)
    if not mask[-1]: ends.append(raw.n_times)
    win = int((epoch_config['non_seizure'][1] - epoch_config['non_seizure'][0]) * sfreq)
    events = []
    for st, en in zip_longest(starts, ends):
        if st is None or en is None: continue
        for off in range(0, en - st - win + 1, win):
            s0, e0 = st + off, st + off + win
            if mask[s0:e0].any():
                continue
            events.append([s0, 0, 2])
            if sample_limit and len(events) >= sample_limit:
                break
        if sample_limit and len(events) >= sample_limit:
            break
    if not events:
        return None
    return mne.Epochs(raw, np.array(events), tmin=epoch_config['non_seizure'][0], tmax=epoch_config['non_seizure'][1], baseline=None, picks='all', preload=True, verbose=False)

def find_events_file(ecg_path: str) -> str | None:
    base = os.path.basename(ecg_path).replace("_ecg.edf", "")
    eeg_ts = ecg_path.replace(os.sep + "ecg" + os.sep, os.sep + "eeg" + os.sep).replace("_ecg.edf", "_events.tsv")
    if os.path.exists(eeg_ts):
        return eeg_ts
    folder = os.path.dirname(eeg_ts)
    alts = glob.glob(os.path.join(folder, f"{base.split('_run')[0]}*_events.tsv"))
    return alts[0] if alts else None

def extract_ecg_features(epoch: mne.Epochs) -> dict | None:
    try:
        data = epoch.get_data().squeeze()
        ecg_signals, info = nk.ecg_process(data, sampling_rate=sfreq)
        rpeaks = np.asarray(info.get("ECG_R_Peaks", []), dtype=int)
        if rpeaks.size < 2:
            return None
        rr_ms = np.diff(rpeaks) / sfreq * 1000
        hr = 60000 / rr_ms.mean() if rr_ms.size else np.nan
        hrv = {
            "SDNN": np.std(rr_ms) if rr_ms.size >= 2 else np.nan,
            "RMSSD": np.sqrt(np.mean(np.diff(rr_ms) ** 2)) if rr_ms.size >= 2 else np.nan,
            "MEAN_RR": rr_ms.mean() if rr_ms.size else np.nan,
            "MEDIAN_RR": np.median(rr_ms) if rr_ms.size else np.nan,
            "pNN50": 100 * np.sum(np.abs(np.diff(rr_ms)) > 50) / (rr_ms.size - 1) if rr_ms.size > 1 else np.nan,
            "LF": np.nan, "HF": np.nan, "LF_HF_RATIO": np.nan
        }
        try:
            freq_df = nk.hrv_frequency(rpeaks, sampling_rate=sfreq, show=False)
            for col in ("LF", "HF", "LF/HF"):
                alt = f"HRV_{col}"
                if col in freq_df.columns:
                    hrv[col if col != "LF/HF" else "LF_HF_RATIO"] = freq_df[col].iloc[0]
                elif alt in freq_df.columns:
                    hrv[col if col != "LF/HF" else "LF_HF_RATIO"] = freq_df[alt].iloc[0]
        except Exception as e:
            logger.warning(f"freq‑HRV failed: {e}")

        qrs = {"QRS_DURATION": np.nan, "QT_INTERVAL": np.nan, "QTc": np.nan}
        try:
            waves = info.get("ECG_Waves", {})
            q_peaks = np.asarray(waves.get("ECG_Q_Peaks", []), dtype=int)
            s_peaks = np.asarray(waves.get("ECG_S_Peaks", []), dtype=int)
            t_peaks = np.asarray(waves.get("ECG_T_Peaks", []), dtype=int)
            qrs_dur, qt_int = [], []
            for r in rpeaks:
                q_prev = q_peaks[q_peaks < r]
                if q_prev.size:
                    q = q_prev[-1]
                    s_next = s_peaks[s_peaks > r]
                    if s_next.size:
                        qrs_dur.append((s_next[0] - q) / sfreq * 1000)
                    t_next = t_peaks[t_peaks > r]
                    if t_next.size:
                        qt_int.append((t_next[0] - q) / sfreq * 1000)
            if qrs_dur:
                qrs["QRS_DURATION"] = np.mean(qrs_dur)
            if qt_int:
                qt = np.mean(qt_int)
                qrs["QT_INTERVAL"] = qt
                rr_sec = hrv["MEAN_RR"] / 1000 if not np.isnan(hrv["MEAN_RR"]) else np.nan
                qrs["QTc"] = qt / np.sqrt(rr_sec) if not np.isnan(rr_sec) else np.nan
        except Exception as e:
            logger.warning(f"QRS/QT extraction failed: {e}")

        return {"HR": hr, **hrv, **qrs}
    except Exception as e:
        logger.warning(f"nk.ecg_process failed: {e}")
        return None

def process_single_file(ecg_path):
    try:
        raw = mne.io.read_raw_edf(ecg_path, preload=True, verbose=False)
        ecg_chs = [ch for ch in raw.ch_names if "ECG" in ch.upper()]
        if not ecg_chs:
            logger.error(f"No ECG channel in {os.path.basename(ecg_path)} → skip")
            return
        raw.set_channel_types({ecg_chs[0]: "ecg"})
        raw.pick_channels([ecg_chs[0]])
        raw.filter(0.5, 40, picks="all", verbose=False)
        raw.notch_filter(50, picks="all", verbose=False)
        if raw.info["sfreq"] != sfreq:
            raw.resample(sfreq, verbose=False)
        ev_path = find_events_file(ecg_path)
        if not ev_path or not os.path.exists(ev_path):
            logger.error(f"No events file for {os.path.basename(ecg_path)} → skip")
            return
        ev_df = pd.read_csv(ev_path, sep="\t")
        sz_df = ev_df[ev_df.eventType.str.startswith("sz_")]
        sz_intervals = [(int(r.onset * sfreq), int((r.onset + r.duration) * sfreq)) for _, r in sz_df.iterrows()]
        for _, row in sz_df.iterrows():
            for key in ("preictal", "ictal", "onset"):
                ep = create_epoch(raw, row.onset, *epoch_config[key], key)
                if ep is None:
                    continue
                feats = extract_ecg_features(ep)
                if feats is None:
                    continue
                feats.update({"file": os.path.basename(ecg_path), "epoch_type": key, "onset": row.onset})
                feature_containers[key].append(feats)
                containers[key].append(ep)
        non_ep = process_non_seizure(raw, sz_intervals, sample_limit=500)
        if non_ep is not None:
            for idx in range(len(non_ep)):
                feats = extract_ecg_features(non_ep[idx:idx+1])
                if feats:
                    feats.update({"file": os.path.basename(ecg_path), "epoch_type": "non_seizure", "onset": non_ep.events[idx,0] / sfreq})
                    feature_containers["non_seizure"].append(feats)
            containers["non_seizure"].append(non_ep)
        logger.info(f"{os.path.basename(ecg_path)} → sz={len(sz_df)}, non={len(non_ep) if non_ep else 0}")
    except Exception as e:
        logger.error(f"{os.path.basename(ecg_path)} failed: {e}")

all_files = glob.glob(os.path.join(base_path, "sub-*", "ses-*", "ecg", "*_ecg.edf"))
Parallel(n_jobs=16)(delayed(process_single_file)(f) for f in tqdm(all_files, desc="ECG Files"))

for key, rows in feature_containers.items():
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(base_path, f"{key}_ecg-features.csv"), index=False)
        logger.info(f"Wrote {len(rows)} rows → {key}_ecg-features.csv")

for key, eps_list in containers.items():
    if not eps_list:
        continue
    epochs_all = mne.concatenate_epochs(eps_list)
    if key == "non_seizure":
        sz_counts = [len(mne.concatenate_epochs(containers[k])) for k in ("preictal", "ictal") if containers[k]]
        if sz_counts:
            max_sz = max(sz_counts)
            target = min(len(epochs_all), 10 * max_sz)
            if len(epochs_all) > target:
                idx = np.random.choice(len(epochs_all), target, replace=False)
                epochs_all = epochs_all[idx]
    try:
        epochs_clean = AutoReject().fit_transform(epochs_all)
    except Exception:
        epochs_clean = fallback_ecg_rejection(epochs_all, min_samples=(20 if key != "non_seizure" else 100))
    epochs_clean.save(os.path.join(base_path, f"{key}_ecg-clean-epo.fif"), overwrite=True)
    logger.info(f"Saved {key}_ecg-clean-epo.fif → {len(epochs_clean)} epochs")

logger.info("==== ECG preprocessing complete ====")