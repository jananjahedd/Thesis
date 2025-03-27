import mne
import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

# --- Parameters ---
base_path = "/Users/jananjahed/Desktop/Bachelor's project/ds005873"
subject_pattern = "sub-00[1-5]"
tmin, tmax = -5, 20  # shorter epoch window
reject_criteria = dict(eeg=600e-6)  # adjusted for artifact leniency
flat_criteria = dict(eeg=1e-6)
plot_duration = 60
desired_channels = ['BTEleft SD', 'CROSStop SD']

# --- Setup for saving plots ---
plots_folder = os.path.join(base_path, "compact_plots")
os.makedirs(plots_folder, exist_ok=True)

# --- Find all EDF files ---
edf_files = glob.glob(os.path.join(base_path, "sub-00[1-5]", "ses-*", "eeg", "*_eeg.edf"))
print(f"Found {len(edf_files)} EDF files for the first 5 subjects.")

# --- Counters & containers ---
total_seizure_attempted = total_seizure_accepted = 0
total_nonseizure_attempted = total_nonseizure_accepted = 0
seizure_epochs_list, nonseizure_epochs_list = [], []
reference_channels = None

for edf_path in edf_files:
    tsv_path = edf_path.replace("_eeg.edf", "_events.tsv")
    print(f"Processing: {os.path.basename(edf_path)}")

    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    except Exception as e:
        print("Error loading EDF:", e)
        continue

    # Pick and reference channels
    available = [ch for ch in desired_channels if ch in raw.ch_names]
    if set(available) != set(desired_channels):
        print("Missing required channels; skipping.")
        continue
    raw.pick_channels(available)
    raw.set_eeg_reference('average', projection=True)

    # Filter + notch
    raw.filter(l_freq=0.5, h_freq=40, fir_design='firwin')
    raw.notch_filter(freqs=50)

    # Resample
    raw.resample(256, npad='auto')

    # Skip ICA: not effective with only 2 channels
    print("Skipping ICA: only 2 channels available.")

    # Load events
    try:
        events_df = pd.read_csv(tsv_path, sep='\t')
    except Exception as e:
        print("Error loading TSV:", e)
        continue

    sfreq = raw.info['sfreq']
    def create_events(df, eid):
        if df.empty:
            return None
        samp = np.round(df['onset'] * sfreq).astype(int)
        return np.column_stack((samp, np.zeros_like(samp), np.full_like(samp, eid)))

    # Seizure epochs
    seiz_df = events_df[events_df['eventType'].str.startswith('sz_', na=False)]
    total_seizure_attempted += len(seiz_df)
    seiz_events = create_events(seiz_df, 1)
    if seiz_events is not None:
        ep_sz = mne.Epochs(raw, events=seiz_events, event_id={'seizure':1}, tmin=tmin, tmax=tmax,
                           baseline=(None,0), reject=reject_criteria, flat=flat_criteria, preload=True)
        total_seizure_accepted += len(ep_sz)
        if len(ep_sz) > 0:
            seizure_epochs_list.append(ep_sz)

            # Visual inspection: plot a few epochs
            fig_epochs = ep_sz.plot(n_epochs=5, n_channels=2, scalings='auto', show=False)
            fig_epochs.savefig(os.path.join(plots_folder, f"{os.path.basename(edf_path)}_seizure_preview.png"))
            plt.close(fig_epochs)

            # Save drop log plot
            fig = ep_sz.plot_drop_log(show=False)
            fig.savefig(os.path.join(plots_folder, f"{os.path.basename(edf_path)}_seizure_drop_log.png"))
            plt.close(fig)

    # Non-seizure epochs
    non_df = events_df[~events_df['eventType'].str.startswith('sz_', na=False)]
    total_nonseizure_attempted += len(non_df)
    non_events = create_events(non_df, 2)
    if non_events is not None:
        ep_nsz = mne.Epochs(raw, events=non_events, event_id={'nonseizure':2}, tmin=tmin, tmax=tmax,
                             baseline=(None,0), reject=reject_criteria, flat=flat_criteria, preload=True)
        total_nonseizure_accepted += len(ep_nsz)
        if len(ep_nsz) > 0:
            nonseizure_epochs_list.append(ep_nsz)


print(f"Seizure accepted: {total_seizure_accepted}/{total_seizure_attempted}")
print(f"Non-seizure accepted: {total_nonseizure_accepted}/{total_nonseizure_attempted}")


if seizure_epochs_list:
    all_sz = mne.concatenate_epochs(seizure_epochs_list)
    all_sz.save(os.path.join(base_path, 'combined_seizure_epochs-epo.fif'), overwrite=True)
if nonseizure_epochs_list:
    all_nsz = mne.concatenate_epochs(nonseizure_epochs_list)
    all_nsz.save(os.path.join(base_path, 'combined_nonseizure_epochs-epo.fif'), overwrite=True)
if seizure_epochs_list and nonseizure_epochs_list:
    all_epochs = mne.concatenate_epochs(seizure_epochs_list + nonseizure_epochs_list)
    all_epochs.save(os.path.join(base_path, 'combined_all_epochs-epo.fif'), overwrite=True)