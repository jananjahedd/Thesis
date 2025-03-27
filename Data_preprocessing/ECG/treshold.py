import mne
import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

# --- Parameters ---
base_path = "/Users/jananjahed/Desktop/Bachelor's project/ds005873"
subject_pattern = "sub-00[1-5]"  # Process subjects sub-001 to sub-005
tmin, tmax = -10, 40             # Fixed epoch window: from -10 s to +40 s (total 50 s)
desired_channels = ['BTEleft SD', 'CROSStop SD']  # Required channel names
plot_duration = 30              # Duration for raw plot (seconds)
# Base rejection criteria in µV (will be adjusted in the scan)
reject_criteria_base = dict(eeg=1000)  
# Range of rejection thresholds (in µV) to scan:
threshold_range = range(500, 2100, 100)  # 500,600,...,2000 µV

# --- Setup for Saving Plots ---
plots_folder = os.path.join(base_path, "compact_plots")
os.makedirs(plots_folder, exist_ok=True)

# --- Find all EDF Files for Subjects sub-001 to sub-005 ---
edf_files = glob.glob(os.path.join(base_path, subject_pattern, "ses-*", "eeg", "*_eeg.edf"), recursive=True)
print(f"Found {len(edf_files)} EDF files for the first 5 subjects.")

# --- Helper Function: Create Events Array ---
def create_events(df, event_id, sfreq):
    if df.empty:
        return None
    samples = np.round(df["onset"] * sfreq).astype(int)
    return np.column_stack((samples, np.zeros_like(samples), np.full_like(samples, event_id, dtype=int)))

# --- Helper Function: Count Accepted Epochs at a Given Threshold ---
def count_accepted_epochs(raw, events_array, event_dict, tmin, tmax, threshold, reject_criteria_base):
    # Convert threshold (µV) to V in reject criteria.
    reject_criteria = {k: threshold * 1e-6 for k in reject_criteria_base}
    epochs = mne.Epochs(raw, events=events_array, event_id=event_dict,
                        tmin=tmin, tmax=tmax, baseline=(None, 0),
                        reject=reject_criteria, preload=True, verbose=False)
    return len(epochs)

# --- Initialize Lists and Counters ---
seizure_epochs_list = []
nonseizure_epochs_list = []
total_seizure_attempted = 0
total_seizure_accepted = 0
total_nonseizure_attempted = 0
total_nonseizure_accepted = 0
results = []  # To store threshold scan results
reference_channels = None  # For consistent channel order

# --- Process Each EDF File ---
for edf_path in edf_files:
    tsv_path = edf_path.replace("_eeg.edf", "_events.tsv")
    print(f"\nProcessing file: {os.path.basename(edf_path)}")
    
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    except Exception as e:
        print("Error loading EDF file:", edf_path, e)
        continue
    print(raw.info)
    
    # Enforce consistent channel order
    available_channels = [ch for ch in desired_channels if ch in raw.info["ch_names"]]
    if set(available_channels) != set(desired_channels):
        print("Missing required channels - skipping this file.")
        continue
    if reference_channels is None:
        reference_channels = available_channels
        print(f"Reference channels locked to: {reference_channels}")
    raw.pick_channels(reference_channels)
    
    # Save raw data plot for subject sub-001
    if "sub-001" in edf_path:
        fig_raw = raw.plot(duration=plot_duration, n_channels=len(available_channels),
                           title=f"Raw EEG: {os.path.basename(edf_path)}", show=False)
        raw_plot_path = os.path.join(plots_folder, os.path.basename(edf_path).replace(".edf", "_raw.png"))
        fig_raw.savefig(raw_plot_path, dpi=100, bbox_inches='tight')
        plt.close(fig_raw)
    
    # --- Filtering ---
    raw.filter(l_freq=0.5, h_freq=70, fir_design='firwin')
    raw.notch_filter(freqs=[50])
    
    # --- Load Events ---
    try:
        events_df = pd.read_csv(tsv_path, sep="\t")
    except Exception as e:
        print("Error loading TSV file:", tsv_path, e)
        continue
    print("First few rows of events TSV:")
    print(events_df.head())
    
    sfreq = raw.info["sfreq"]
    
    # --- Extract Seizure and Non-Seizure Events ---
    seizure_df = events_df[events_df["eventType"].str.startswith("sz_", na=False)]
    nonseizure_df = events_df[~events_df["eventType"].str.startswith("sz_", na=False)]
    
    n_seizure_attempted = len(seizure_df)
    n_nonseizure_attempted = len(nonseizure_df)
    total_seizure_attempted += n_seizure_attempted
    total_nonseizure_attempted += n_nonseizure_attempted
    
    seizure_events = create_events(seizure_df, 1, sfreq)
    nonseizure_events = create_events(nonseizure_df, 2, sfreq)
    
    # --- Threshold Scan for This File ---
    for thr in threshold_range:
        seizure_count = 0
        nonseizure_count = 0
        if seizure_events is not None:
            seizure_count = count_accepted_epochs(raw, seizure_events, {'seizure': 1},
                                                  tmin, tmax, thr, reject_criteria_base)
        if nonseizure_events is not None:
            nonseizure_count = count_accepted_epochs(raw, nonseizure_events, {'nonseizure': 2},
                                                     tmin, tmax, thr, reject_criteria_base)
        results.append({
            'file': os.path.basename(edf_path),
            'threshold': thr,
            'seizure_attempted': n_seizure_attempted,
            'seizure_accepted': seizure_count,
            'nonseizure_attempted': n_nonseizure_attempted,
            'nonseizure_accepted': nonseizure_count
        })
    
    # --- Create Epochs with Chosen Threshold for Final Processing ---
    chosen_threshold = 1500  # Adjust based on threshold scan results
    if seizure_events is not None:
        seizure_epochs = mne.Epochs(raw, events=seizure_events, event_id={'seizure': 1},
                                    tmin=tmin, tmax=tmax, baseline=(None, 0),
                                    reject={k: chosen_threshold * 1e-6 for k in reject_criteria_base},
                                    preload=True, verbose=False)
        n_seizure = len(seizure_epochs)
        total_seizure_accepted += n_seizure
        print(f"Using threshold {chosen_threshold} µV: Created {n_seizure} seizure epochs from file.")
        if n_seizure > 0:
            seizure_epochs.metadata = pd.DataFrame({"label": ["seizure"] * n_seizure})
            seizure_epochs_list.append(seizure_epochs)
            
            # For subject sub-001, save the drop log plot.
            if "sub-001" in edf_path:
                fig_drop = seizure_epochs.plot_drop_log(show=False)
                drop_log_path = os.path.join(plots_folder, os.path.basename(edf_path).replace(".edf", "_drop_log.png"))
                fig_drop.savefig(drop_log_path, dpi=100, bbox_inches='tight')
                plt.close(fig_drop)
    if nonseizure_events is not None:
        nonseizure_epochs = mne.Epochs(raw, events=nonseizure_events, event_id={'nonseizure': 2},
                                       tmin=tmin, tmax=tmax, baseline=(None, 0),
                                       reject={k: chosen_threshold * 1e-6 for k in reject_criteria_base},
                                       preload=True, verbose=False)
        n_nonseizure = len(nonseizure_epochs)
        total_nonseizure_accepted += n_nonseizure
        print(f"Using threshold {chosen_threshold} µV: Created {n_nonseizure} non-seizure epochs from file.")
        if n_nonseizure > 0:
            nonseizure_epochs.metadata = pd.DataFrame({"label": ["nonseizure"] * n_nonseizure})
            nonseizure_epochs_list.append(nonseizure_epochs)

# --- Save Threshold Scan Results ---
df_results = pd.DataFrame(results)
csv_path = os.path.join(base_path, "threshold_scan_results.csv")
df_results.to_csv(csv_path, index=False)
print(f"\nThreshold scan complete. Results saved to {csv_path}.")

# --- Overall Summary ---
print("\n--- Overall Summary ---")
print(f"Total seizure events attempted: {total_seizure_attempted}")
print(f"Total seizure epochs accepted: {total_seizure_accepted}")
print(f"Total seizure epochs dropped: {total_seizure_attempted - total_seizure_accepted}")
print(f"Total non-seizure events attempted: {total_nonseizure_attempted}")
print(f"Total non-seizure epochs accepted: {total_nonseizure_accepted}")
print(f"Total non-seizure epochs dropped: {total_nonseizure_attempted - total_nonseizure_accepted}")

# --- Concatenate and Save Epochs ---
if seizure_epochs_list:
    all_seizure_epochs = mne.concatenate_epochs(seizure_epochs_list)
    seizure_save_path = os.path.join(base_path, "combined_seizure_epochs-epo.fif")
    all_seizure_epochs.save(seizure_save_path, overwrite=True)
    print(f"Saved combined seizure epochs: {len(all_seizure_epochs)}")
    # Interactive visualization of combined seizure epochs:
    print("Displaying combined seizure epochs (interactive plot)...")
    all_seizure_epochs.plot(n_epochs=5, n_channels=len(desired_channels), title="Combined Seizure Epochs", block=True)
    all_seizure_epochs.plot_drop_log()
else:
    print("No seizure epochs to combine.")

if nonseizure_epochs_list:
    all_nonseizure_epochs = mne.concatenate_epochs(nonseizure_epochs_list)
    nonseizure_save_path = os.path.join(base_path, "combined_nonseizure_epochs-epo.fif")
    all_nonseizure_epochs.save(nonseizure_save_path, overwrite=True)
    print(f"Saved combined non-seizure epochs: {len(all_nonseizure_epochs)}")
    # Interactive visualization of combined non-seizure epochs:
    print("Displaying combined non-seizure epochs (interactive plot)...")
    all_nonseizure_epochs.plot(n_epochs=5, n_channels=len(desired_channels), title="Combined Non-Seizure Epochs", block=True)
    all_nonseizure_epochs.plot_drop_log()
else:
    print("No non-seizure epochs to combine.")

if seizure_epochs_list and nonseizure_epochs_list:
    all_epochs = mne.concatenate_epochs(seizure_epochs_list + nonseizure_epochs_list)
    combined_save_path = os.path.join(base_path, "combined_all_epochs-epo.fif")
    all_epochs.save(combined_save_path, overwrite=True)
    print(f"Saved combined epochs (both classes): {len(all_epochs)}")
