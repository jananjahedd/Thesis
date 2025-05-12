import mne
import pandas as pd
import os
import glob
import logging

# Setup Logger
logging.basicConfig(filename='preprocessing_log.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Parameters
base_path = "/Users/jananjahed/Desktop/Bachelor's project/ds005873"
desired_channels = ['BTEleft SD', 'CROSStop SD']

# Find all EDF files across all subjects in the directory (recursive search)
all_edf_files = glob.glob(os.path.join(base_path, "**", "*_eeg.edf"), recursive=True)
logger.info(f"Found {len(all_edf_files)} EDF files across all subjects.")

# Counters & containers
seizure_durations = []
seizures_under_20s = 0

# Process each EDF file
for edf_path in all_edf_files:
    tsv_path = edf_path.replace("_eeg.edf", "_events.tsv")
    logger.info(f"Processing: {os.path.basename(edf_path)}")

    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    except Exception as e:
        logger.error(f"Error loading EDF: {e}")
        continue

    # Pick desired channels
    available = [ch for ch in desired_channels if ch in raw.ch_names]
    if set(available) != set(desired_channels):
        logger.warning(f"Missing required channels in {os.path.basename(edf_path)}; skipping.")
        continue
    raw.pick_channels(available)

    # Load events
    try:
        events_df = pd.read_csv(tsv_path, sep='\t')
    except Exception as e:
        logger.error(f"Error loading TSV: {e}")
        continue

    # Process seizure events
    seiz_df = events_df[events_df['eventType'].str.startswith('sz_', na=False)]
    if not seiz_df.empty:
        seiz_df['onset_end'] = seiz_df['onset'] + seiz_df['duration']  # Assuming 'duration' column exists
        seizure_durations.extend(seiz_df['duration'].values)

        # Count seizures under 20s
        seizures_under_20s += sum(1 for duration in seiz_df['duration'] if duration < 12)

# Calculate minimum seizure duration
min_seizure_duration = min(seizure_durations) if seizure_durations else None

# Log the results
logger.info(f"Minimum seizure duration: {min_seizure_duration}s")
logger.info(f"Seizures under 20s: {seizures_under_20s} out of {len(seizure_durations)} seizures.")
