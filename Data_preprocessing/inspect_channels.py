import os
import glob
import argparse
import mne


def inspect_specific_subject_channels(base_path, subject_ids):
    """
    Reads EDF files for specified subjects and prints their channel names,
    sampling frequency, and duration.

    Args:
        base_path (str): The root folder containing sub-*/ses-*/eeg data.
        subject_ids (list of str): A list of subject IDs to inspect
        (e.g., ['019', '010']).
    """
    all_found_files_for_subjects = []
    for subj_id in subject_ids:
        glob_pattern = os.path.join(base_path, f"sub-{subj_id}", "ses-*",
                                    "eeg", "*_eeg.edf")
        edf_files_for_subject = glob.glob(glob_pattern)
        if not edf_files_for_subject:
            print(f"No EDF found for subject sub-{subj_id} with pattern: "
                  f"{glob_pattern}")
        all_found_files_for_subjects.extend(edf_files_for_subject)

    if not all_found_files_for_subjects:
        print(f"No EDF  found for the specified subjects: {subject_ids} in "
              f"base path: {base_path}")
        return

    print(f"Found {len(all_found_files_for_subjects)} EDF for subjects "
          f"{subject_ids}. Inspecting all...")

    for i, edf_path in enumerate(all_found_files_for_subjects):
        print(f"\n--- File {i+1}/{len(all_found_files_for_subjects)} ---")
        print(f"Path: {edf_path}")
        try:
            raw = mne.io.read_raw_edf(edf_path, preload=False,
                                      verbose='WARNING')
            print(f"Channels: {raw.ch_names}")
            print(f"Sampling Frequency: {raw.info['sfreq']} Hz")
            if raw.n_times > 0:
                print(f"Duration: {raw.times[-1]:.2f} seconds")
            else:
                print("Duration: 0.00 seconds")
        except FileNotFoundError:
            print(f"Error: File not found at {edf_path}.")
        except Exception as e:
            print(f"Error reading or processing file {edf_path}: {e}")
        print("--------------------")

    print(f"\nInspection complete. {len(all_found_files_for_subjects)} "
          f"files for subjects {subject_ids}.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Inspect channel names, sampling frequency, and duration "
        "in EDF files for specific subjects."
    )
    parser.add_argument(
        '--base_path',
        required=True,
        help='Root folder containing sub-*/ses-*/eeg data '
        '(e.g., /Users/jananjahed/Desktop/BP/ds005873)'
    )

    args = parser.parse_args()

    subjects_to_inspect = ['019', '010']

    print(f"Targeting subjects: {subjects_to_inspect}")
    inspect_specific_subject_channels(args.base_path, subjects_to_inspect)
