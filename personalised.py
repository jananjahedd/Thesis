"""
Filename: personalised.py
Author: Janan Jahed
Description: This file includes the Random Forest model trained for a patient
specific framework
"""

import argparse
import gc
import glob
import logging
import os
import time
import warnings

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.signal import welch
from scipy.stats import kurtosis, skew
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import (RepeatedStratifiedKFold, learning_curve)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings(
    "ignore", message="`trapz` is deprecated.*", category=DeprecationWarning
)


def plot_roc_curves_for_subject(results_df, subject_id, save_path):
    """Plot the mean ROC curve for each model with standard deviation

    Args:
        results_df (pd.DataFrame): DataFrame containing fold results,
                                   including true labels and predictions
        subject_id (str): The identifier for the subject
        save_path (str): Directory path to save the plot
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 8))

    for model_name, group in results_df.groupby('model_name'):
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        for _, row in group.iterrows():
            fpr, tpr, _ = roc_curve(row['y_true'], row['y_pred_proba'])
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            aucs.append(roc_auc_score(row['y_true'], row['y_pred_proba']))

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        ax.plot(mean_fpr, mean_tpr,
                label=f'{model_name} (AUC = {mean_auc:.2f} '
                      f'$\\pm$ {std_auc:.2f})',
                lw=2, alpha=0.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, alpha=0.2)

    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           xlabel='False Positive Rate', ylabel='True Positive Rate',
           title=f'Mean ROC Curves for {subject_id}')
    ax.legend(loc='lower right')
    plt.savefig(os.path.join(save_path, f'{subject_id}_mean_roc_curves.png'))
    plt.close(fig)


def plot_confusion_matrices_for_subject(results_df, subject_id, save_path):
    """Plot the mean normalised confusion matrix for each model type

    Args:
        results_df (pd.DataFrame): DataFrame with fold results
        subject_id (str): The identifier for the subject
        save_path (str): Directory path to save the plot
    """
    model_types = results_df['model_name'].unique()
    fig, axes = plt.subplots(1, len(model_types),
                             figsize=(6 * len(model_types), 5), squeeze=False)
    title = f'Mean Normalized Confusion Matrices for {subject_id}'
    fig.suptitle(title, fontsize=16)

    for i, model_name in enumerate(model_types):
        group = results_df[results_df['model_name'] == model_name]
        all_cms = []
        for _, row in group.iterrows():
            if len(np.unique(row['y_true'])) < 2:
                continue
            y_pred = (row['y_pred_proba'] > 0.5).astype(int)
            cm = confusion_matrix(row['y_true'], y_pred, labels=[0, 1])
            all_cms.append(cm)

        if not all_cms:
            continue

        mean_cm = np.mean(all_cms, axis=0)
        cm_sum = mean_cm.sum(axis=1)[:, np.newaxis]
        mean_cm_norm = np.divide(mean_cm.astype('float'), cm_sum,
                                 out=np.zeros_like(mean_cm),
                                 where=cm_sum != 0)

        sns.heatmap(mean_cm_norm, annot=True, fmt='.2f', cmap='Blues',
                    ax=axes[0, i], cbar=False)
        axes[0, i].set_title(model_name)
        axes[0, i].set_xlabel('Predicted Label')
        axes[0, i].set_ylabel('True Label')
        axes[0, i].set_xticklabels(['Non-Seizure', 'Seizure'])
        axes[0, i].set_yticklabels(['Non-Seizure', 'Seizure'], va='center')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_file = f'{subject_id}_mean_confusion_matrices.png'
    plt.savefig(os.path.join(save_path, save_file))
    plt.close(fig)


def plot_performance_bars_for_subject(summary_df, subject_id, save_path):
    """Plot a bar chart comparing model performance with error bars.

    Args:
        summary_df (pd.DataFrame): DataFrame with mean/std performance.
        subject_id (str): The identifier for the subject.
        save_path (str): Directory path to save the plot.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    title = f'Model Performance Comparison for {subject_id}'
    fig.suptitle(title, fontsize=16)

    metrics = ['roc_auc', 'f1_seizure']
    titles = ['Mean ROC AUC', 'Mean Seizure F1-Score']

    for i, metric in enumerate(metrics):
        sns.barplot(x='model_name', y=f'{metric}_mean', data=summary_df,
                    ax=axes[i], capsize=0.1)
        axes[i].errorbar(x=summary_df['model_name'],
                         y=summary_df[f'{metric}_mean'],
                         yerr=summary_df[f'{metric}_std'],
                         fmt='none', c='black', capsize=5)
        axes[i].set_title(titles[i])
        axes[i].set_xlabel('Model Type')
        axes[i].set_ylabel('Score')
        axes[i].set_ylim(0, 1.05)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_file = f'{subject_id}_performance_bar_plot.png'
    plt.savefig(os.path.join(save_path, save_file))
    plt.close(fig)


def plot_feature_importances(model, x_cols, subject_id, model_name,
                             save_path, top_n=20):
    """Plot the top N feature importances for a given model

    Args:
        model (RandomForestClassifier): The trained model.
        x_cols (dict): Dictionary with channel counts for 'eeg' and 'ecg'
        subject_id (str): The identifier for the subject
        model_name (str): The name of the model being evaluated
        save_path (str): Directory path to save the plot
        top_n (int, optional): Number of top features to display
                               Defaults to 20
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    feature_names = []
    eeg_channels = x_cols.get('eeg', 0)
    ecg_channels = x_cols.get('ecg', 0)
    feat_types = ['mean', 'std', 'skew', 'kurtosis', 'rms', 'delta', 'theta',
                  'alpha', 'beta']

    if 'EEG' in model_name:
        for i in range(eeg_channels):
            for f in feat_types:
                feature_names.append(f'EEG-Ch{i+1}-{f}')
    if 'ECG' in model_name:
        for i in range(ecg_channels):
            for f in feat_types:
                feature_names.append(f'ECG-Ch{i+1}-{f}')

    feature_names = np.array(feature_names)
    num_features = min(top_n, len(importances))

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))
    title = (f'Top {num_features} Feature Importances for '
             f'{subject_id} ({model_name})')
    ax.set_title(title)
    ax.bar(range(num_features), importances[indices][:num_features],
           align='center')
    ax.set_xticks(range(num_features))
    ax.set_xticklabels(feature_names[indices][:num_features], rotation=90)
    ax.set_xlim([-1, num_features])
    plt.tight_layout()
    save_file = f'{subject_id}_{model_name}_feature_importances.png'
    plt.savefig(os.path.join(save_path, save_file))
    plt.close(fig)


def plot_learning_curves_for_subject(model, X, y, cv, subject_id,
                                     model_name, save_path):
    """Generate and save learning curves for a given model

    Args:
        model (RandomForestClassifier): The classifier instance
        X (np.ndarray): The feature matrix
        y (np.ndarray): The target labels
        cv (object): The cross-validation splitting strategy
        subject_id (str): The identifier for the subject
        model_name (str): The name of the model
        save_path (str): Directory path to save the plot
    """
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='f1_macro'
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(f'Learning Curve for {subject_id} ({model_name})')
    ax.set_xlabel("Training examples")
    ax.set_ylabel("F1 Macro Score")
    ax.grid(True)

    ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.1,
                    color="r")
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.1,
                    color="g")
    ax.plot(train_sizes, train_scores_mean, 'o-', color="r",
            label="Training score")
    ax.plot(train_sizes, test_scores_mean, 'o-', color="g",
            label="Cross-validation score")
    ax.legend(loc="best")
    save_file = f'{subject_id}_{model_name}_learning_curve.png'
    plt.savefig(os.path.join(save_path, save_file))
    plt.close(fig)


def get_args():
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: An object containing the script's arguments.
    """
    parser = argparse.ArgumentParser(
        description="Patient-Specific EEG/ECG Seizure Detection using RF"
    )
    parser.add_argument('--base_path', required=True,
                        help='Root folder for the BIDS dataset')
    parser.add_argument('--eeg_deriv_dir', type=str,
                        default='eeg_paired_by_subject',
                        help='Name of the EEG derivatives folder')
    parser.add_argument('--ecg_deriv_dir', type=str,
                        default='ecg_paired_by_subject',
                        help='Name of the ECG derivatives folder')
    parser.add_argument('--results_dir', type=str,
                        default='patient_specific_results_rf',
                        help='Folder to save Random Forest results')
    parser.add_argument('--n_splits', type=int, default=5,
                        help='Number of splits for K-Fold CV')
    parser.add_argument('--n_repeats', type=int, default=3,
                        help='Number of repeats for K-Fold CV')
    return parser.parse_args()


def extract_features(data, sfreq):
    """Extract time and frequency domain features from epoch data

    Args:
        data (np.ndarray): Epoch data of shape
                           (n_epochs, n_channels, n_times)
        sfreq (float): The sampling frequency of the data

    Returns:
        np.ndarray: A feature matrix of shape
                    (n_epochs, n_channels * n_features)
    """
    n_epochs, n_channels, _ = data.shape
    bands = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13),
             'beta': (13, 30)}
    n_features_per_channel = 5 + len(bands)
    features = np.zeros((n_epochs, n_channels * n_features_per_channel))

    for i in range(n_epochs):
        epoch_features = []
        for j in range(n_channels):
            channel_data = data[i, j, :]
            #time-domain features
            epoch_features.extend([
                np.mean(channel_data),
                np.std(channel_data),
                skew(channel_data),
                kurtosis(channel_data),
                np.sqrt(np.mean(channel_data**2))
            ])
            #frequency-domain features
            freqs, psd = welch(channel_data, sfreq, nperseg=int(sfreq*2))
            psd_sum = np.sum(psd)
            psd_norm = psd / psd_sum if psd_sum > 0 else psd
            for low, high in bands.values():
                band_mask = (freqs >= low) & (freqs <= high)
                band_power = np.trapz(psd_norm[band_mask], freqs[band_mask])
                epoch_features.append(band_power)
        features[i, :] = epoch_features
    return features


def load_subject_data(subject_id, args):
    """Load, align, and extract features for a single subject

    Args:
        subject_id (str): The identifier for the subject (e.g., 'sub-55')
        args (argparse.Namespace): Parsed command-line arguments

    Returns:
        tuple: A tuple containing:
               - X_eeg_features (np.ndarray): EEG feature matrix
               - X_ecg_features (np.ndarray): ECG feature matrix
               - y (np.ndarray): Target labels
               - X_columns (dict): Dictionary with channel counts
               Returns (None, None, None, None) if data is missing or
               cannot be aligned
    """
    logging.info(f"--- Loading and processing data for {subject_id} ---")
    eeg_path = os.path.join(args.base_path, "derivatives",
                            args.eeg_deriv_dir, subject_id)
    ecg_path = os.path.join(args.base_path, "derivatives",
                            args.ecg_deriv_dir, subject_id)

    eeg_files = glob.glob(os.path.join(eeg_path, "**", "*.fif"),
                          recursive=True)
    ecg_files = glob.glob(os.path.join(ecg_path, "**", "*.fif"),
                          recursive=True)

    if not eeg_files or not ecg_files:
        logging.warning(f"Missing preprocessed files for {subject_id}.")
        return None, None, None, None

    try:
        eeg_epochs = mne.concatenate_epochs(
            [mne.read_epochs(f, preload=True, verbose=False)
             for f in eeg_files]
        )
        ecg_epochs = mne.concatenate_epochs(
            [mne.read_epochs(f, preload=True, verbose=False)
             for f in ecg_files]
        )
        sfreq = eeg_epochs.info['sfreq']
    except Exception as e:
        logging.error(f"Error loading epoch files for {subject_id}: {e}",
                      exc_info=True)
        return None, None, None, None

    eeg_meta = eeg_epochs.metadata.copy()
    eeg_meta['original_index'] = np.arange(len(eeg_meta))
    ecg_meta = ecg_epochs.metadata.copy()
    ecg_meta['original_index'] = np.arange(len(ecg_meta))

    aligned_meta = pd.merge(eeg_meta, ecg_meta, on='unique_epoch_id',
                            how='inner', suffixes=('_eeg', '_ecg'))

    if aligned_meta.empty:
        logging.warning(f"No aligned epochs found for {subject_id}.")
        return None, None, None, None

    eeg_indices = aligned_meta['original_index_eeg'].values
    ecg_indices = aligned_meta['original_index_ecg'].values

    aligned_eeg_data = eeg_epochs[eeg_indices].get_data(copy=False)
    aligned_ecg_data = ecg_epochs[ecg_indices].get_data(copy=False)
    y = aligned_meta['seizure_label_eeg'].values

    logging.info("Extracting features from EEG data...")
    X_eeg = extract_features(aligned_eeg_data, sfreq)

    logging.info("Extracting features from ECG data...")
    X_ecg = extract_features(aligned_ecg_data, sfreq)

    x_cols = {'eeg': aligned_eeg_data.shape[1],
              'ecg': aligned_ecg_data.shape[1]}

    logging.info(f"Loaded {len(y)} aligned windows for {subject_id}.")
    return X_eeg, X_ecg, y, x_cols


def train_and_evaluate_patient(X_eeg, X_ecg, y, x_cols, subject_id, args):
    """Train models, run CV, and generate reports for a subject.

    Args:
        X_eeg (np.ndarray): EEG feature matrix
        X_ecg (np.ndarray): ECG feature matrix
        y (np.ndarray): Target labels
        x_cols (dict): Dictionary with channel counts
        subject_id (str): The identifier for the subject
        args (argparse.Namespace): Parsed command-line arguments

    Returns:
        pd.DataFrame: A DataFrame containing the CV results, excluding
                      raw predictions. Returns None if training fails
    """
    path = os.path.join(args.base_path, "derivatives", args.results_dir,
                        subject_id)
    os.makedirs(path, exist_ok=True)

    experiments = {"EEG_Only": X_eeg, "ECG_Only": X_ecg,
                   "Fused_EEG_ECG": np.hstack([X_eeg, X_ecg])}
    all_results = []

    for model_name, X in experiments.items():
        msg = f"--- Running CV for {subject_id} - {model_name} ---"
        logging.info(msg)

        if np.sum(y == 1) < args.n_splits:
            logging.warning("Not enough seizure samples for CV. Skipping.")
            continue

        cv = RepeatedStratifiedKFold(n_splits=args.n_splits,
                                     n_repeats=args.n_repeats,
                                     random_state=42)
        fold_metrics = []
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            rf = RandomForestClassifier(n_estimators=150,
                                        class_weight='balanced',
                                        random_state=42, n_jobs=-1,
                                        max_depth=15, min_samples_leaf=5)
            rf.fit(X_train, y_train)

            y_pred_proba = rf.predict_proba(X_test)[:, 1]
            try:
                auc = roc_auc_score(y_test, y_pred_proba)
            except ValueError:
                auc = np.nan
            y_pred = (y_pred_proba > 0.5).astype(int)
            p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred,
                                                          labels=[0, 1],
                                                          average=None)
            acc = accuracy_score(y_test, y_pred)
            fold_metrics.append({'model_name': model_name,
                                 'fold': fold_idx, 'accuracy': acc,
                                 'roc_auc': auc, 'f1_seizure': f1[1],
                                 'recall_seizure': r[1],
                                 'precision_seizure': p[1],
                                 'y_true': y_test,
                                 'y_pred_proba': y_pred_proba})

        if fold_metrics:
            df_fold = pd.DataFrame(fold_metrics)
            all_results.extend(df_fold.to_dict('records'))
            summary = df_fold.drop(columns=['y_true', 'y_pred_proba'])
            logging.info(f"--- CV Summary: {model_name} ---\n"
                         f"{summary.describe().to_string()}")

            logging.info(f"Generating learning curve for {model_name}...")
            plot_learning_curves_for_subject(rf, X, y, cv, subject_id,
                                             model_name, path)

    if not all_results:
        return None

    final_df = pd.DataFrame(all_results)
    logging.info("Generating plots and reports...")
    plot_roc_curves_for_subject(final_df, subject_id, path)
    plot_confusion_matrices_for_subject(final_df, subject_id, path)

    summary_df = final_df.groupby('model_name').agg(
        roc_auc_mean=('roc_auc', 'mean'),
        roc_auc_std=('roc_auc', 'std'),
        f1_seizure_mean=('f1_seizure', 'mean'),
        f1_seizure_std=('f1_seizure', 'std')
    ).reset_index()
    plot_performance_bars_for_subject(summary_df, subject_id, path)

    best_model_name = summary_df.loc[
        summary_df['f1_seizure_mean'].idxmax()]['model_name']
    logging.info(f"Best model for {subject_id} is '{best_model_name}'. "
                 "Generating feature importances")
    X_best = experiments[best_model_name]
    scaler = StandardScaler()
    X_best_scaled = scaler.fit_transform(X_best)
    final_model = RandomForestClassifier(n_estimators=150,
                                         class_weight='balanced',
                                         random_state=42, n_jobs=-1,
                                         max_depth=15, min_samples_leaf=5)
    final_model.fit(X_best_scaled, y)
    plot_feature_importances(final_model, x_cols, subject_id,
                             best_model_name, path)

    return final_df.drop(columns=['y_true', 'y_pred_proba'])


def main():
    """Main seizure detection pipeline."""
    args = get_args()
    results_path = os.path.join(args.base_path, "derivatives",
                                args.results_dir)
    os.makedirs(results_path, exist_ok=True)

    log_file = os.path.join(results_path, 'pipeline_run.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ]
    )

    logging.info("RF Modeling Pipeline Initialized. Results -> %s",
                 results_path)
    logging.info("Args: %s", args)

    eeg_deriv_path = os.path.join(args.base_path, "derivatives",
                                  args.eeg_deriv_dir, "sub-*")
    all_subject_ids = sorted([os.path.basename(d) for d
                              in glob.glob(eeg_deriv_path)])

    overall_summary_list = []
    start_time = time.time()

    for subject_id in all_subject_ids:
        X_eeg_feat, X_ecg_feat, y, X_cols = load_subject_data(subject_id,
                                                              args)

        if X_eeg_feat is None or len(y) < 20:
            logging.warning(f"Skipping {subject_id} due to insufficient "
                            "or inconsistent data.")
            continue

        subject_cv_results = train_and_evaluate_patient(X_eeg_feat,
                                                        X_ecg_feat, y,
                                                        X_cols, subject_id,
                                                        args)

        if subject_cv_results is not None:
            summary = subject_cv_results.groupby('model_name').agg(
                accuracy_mean=('accuracy', 'mean'),
                accuracy_std=('accuracy', 'std'),
                roc_auc_mean=('roc_auc', 'mean'),
                roc_auc_std=('roc_auc', 'std'),
                f1_seizure_mean=('f1_seizure', 'mean'),
                f1_seizure_std=('f1_seizure', 'std'),
                recall_seizure_mean=('recall_seizure', 'mean'),
                recall_seizure_std=('recall_seizure', 'std'),
                precision_seizure_mean=('precision_seizure', 'mean'),
                precision_seizure_std=('precision_seizure', 'std')
            ).reset_index()
            summary['subject_id'] = subject_id
            overall_summary_list.append(summary)

        gc.collect()

    end_time = time.time()
    total_time = end_time - start_time
    logging.info("\nTotal processing time: %.2f minutes.", total_time / 60)

    if overall_summary_list:
        logging.info("\n--- Overall Performance Summary ---")
        overall_df = pd.concat(overall_summary_list, ignore_index=True)
        cols_order = ['subject_id', 'model_name'] + sorted(
            [c for c in overall_df.columns if c not in
             ['subject_id', 'model_name']]
        )
        overall_df = overall_df[cols_order]

        agg_summary = overall_df.drop(
            columns='subject_id').groupby('model_name').mean()
        logging.info("--- Mean Performance Across All Subjects ---\n%s",
                     agg_summary.to_string())

        csv_path = os.path.join(results_path,
                                'overall_summary_metrics.csv')
        overall_df.to_csv(csv_path, index=False)
        logging.info("\nSaved overall summary to %s", csv_path)
    else:
        logging.info("\n--- No patients processed successfully. ---")

    logging.info("\nPipeline complete.")


if __name__ == "__main__":
    main()
