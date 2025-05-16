import os
import glob
import argparse
import pandas as pd
import numpy as np


def analyze_seizure_durations(base_path):
    """
    Analyzes seizure durations from _events.tsv files in a BIDS-like dataset.

    Args:
        base_path (str): The root folder containing sub-*/ses-*/eeg data
                         and corresponding _events.tsv files.
    """
    event_files = glob.glob(os.path.join(base_path, "sub-*", "ses-*", "eeg",
                                         "*_events.tsv"))

    if not event_files:
        print(f"No event files found in the specified base path: {base_path}")
        return

    all_seizure_durations = []
    print(f"Found {len(event_files)} event files. Processing...")

    for i, event_file_path in enumerate(event_files):
        if (i + 1) % 10 == 0 or i == len(event_files) - 1:
            print(f"Processing file {i + 1}/{len(event_files)}: "
                  f"{os.path.basename(event_file_path)}")

        try:
            df = pd.read_csv(event_file_path, sep='\t')

            seizure_events = df[df['eventType'].str.startswith(
                'sz_', na=False)]

            if not seizure_events.empty:
                if 'duration' in seizure_events.columns:
                    durations = pd.to_numeric(seizure_events['duration'],
                                              errors='coerce').dropna()
                    all_seizure_durations.extend(durations.tolist())
                else:
                    print(f"Warning: 'duration' column not found in "
                          f"{event_file_path}. Skipping file for duration "
                          "analysis.")

        except Exception as e:
            print(f"Error processing file {event_file_path}: {e}")
            continue

    if not all_seizure_durations:
        print("No seizure events with valid durations found across all files.")
        return

    durations_array = np.array(all_seizure_durations)
    total_seizure_events = len(durations_array)

    average_duration = np.mean(durations_array) if total_seizure_events > 0 else 0
    min_duration = np.min(durations_array) if total_seizure_events > 0 else 0
    max_duration = np.max(durations_array) if total_seizure_events > 0 else 0

    count_less_than_average = np.sum(durations_array < average_duration) if total_seizure_events > 0 else 0
    count_less_than_20_seconds = np.sum(durations_array < 10) if total_seizure_events > 0 else 0

    print("\n--- Seizure Duration Analysis ---")
    print(f"Total seizure events analyzed: {total_seizure_events}")
    if total_seizure_events > 0:
        print(f"Average seizure duration: {average_duration:.2f} seconds")
        print(f"Minimum seizure duration: {min_duration:.2f} seconds")
        print(f"Maximum seizure duration: {max_duration:.2f} seconds")
        print(f"Number of seizure events with duration less than average "
              f"({average_duration:.2f}s): {count_less_than_average}")
        percentage_less_than_average = (count_less_than_average /
                                        total_seizure_events) * 100
        print(f"Percentage of seizure events with duration less than average: "
              f"{percentage_less_than_average:.2f}%")
        print(f"Number of seizure events with duration less than 20 seconds: "
              f"{count_less_than_20_seconds}")
        percentage_less_than_20_seconds = (count_less_than_20_seconds /
                                           total_seizure_events) * 100
        print(f"Percentage of seizure events with duration less than 20 s: "
              f"{percentage_less_than_20_seconds:.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Analyze seizure durations from BIDS event files.")
    parser.add_argument('--base_path', required=True,
                        help='Root folder containing sub-*/ses-*/eeg'
                        'data and corresponding _events.tsv files '
                        '(e.g., /Users/jananjahed/Desktop/BP/ds005873)')

    args = parser.parse_args()

    analyze_seizure_durations(args.base_path)
