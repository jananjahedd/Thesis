"""
Filename: generalised.py
Author: Janan Jahed
Description: This file includes the Random Forest model trained for a patient
independant framework
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
from joblib import dump
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, welch
from scipy.stats import kurtosis, skew
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, precision_recall_curve,
                             precision_recall_fscore_support, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import StratifiedGroupKFold, learning_curve
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)


def plot_roc_curves(results_list, subject_id, save_path):
    """Plot the ROC curve for each model

    Args:
        results_list (list): A list of result dictionaries, one per model
        subject_id (str): An identifier for the plot title and filename
        save_path (str): The directory where the plot will be saved
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 8))
    for result in results_list:
        y_true = np.concatenate(result['y_true_all_folds'])
        y_pred = np.concatenate(result['y_pred_proba_all_folds'])
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        auc = result['roc_auc']
        label = f"{result['model_name']} (AUC = {auc:.2f})"
        ax.plot(fpr, tpr, label=label, lw=2, alpha=0.8)

    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           xlabel='False Positive Rate', ylabel='True Positive Rate',
           title=f'ROC Curves for {subject_id}')
    ax.legend(loc='lower right')
    plt.savefig(os.path.join(save_path, f'{subject_id}_roc_curves.png'))
    plt.close(fig)


def plot_precision_recall_curves(results_list, subject_id, save_path):
    """Plot Precision-Recall curves and mark the optimal operating points.

    Args:
        results_list (list): A list of result dictionaries for each model.
        subject_id (str): An identifier for the plot title and filename.
        save_path (str): The directory where the plot will be saved.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 8))
    for result in results_list:
        pr_curve = result['pr_curve']
        ax.plot(pr_curve['recall'], pr_curve['precision'],
                label=f"{result['model_name']}", lw=2, alpha=0.8)
        op_point_label = (f"{result['model_name']} Op Point "
                          f"(Thresh={result['optimal_threshold']:.2f})")
        ax.plot(result['recall_seizure'], result['precision_seizure'], 'o',
                markersize=10, label=op_point_label)

    ax.set(xlabel='Recall (Sensitivity)', ylabel='Precision',
           title=f'Precision-Recall Curves for {subject_id}')
    ax.legend(loc='best')
    ax.grid(True)
    f_name = f'{subject_id}_precision_recall_curves.png'
    plt.savefig(os.path.join(save_path, f_name))
    plt.close(fig)


def plot_optimized_confusion_matrices(results_list, subject_id, save_path):
    """Plot confusion matrices based on each model's optimal threshold

    Args:
        results_list (list): A list of result dictionaries for each model
        subject_id (str): identifier for the plot title and filename
        save_path (str): directory where the plot will be saved
    """
    n_models = len(results_list)
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 6),
                             squeeze=False)
    title = f'Optimized Confusion Matrices for {subject_id}'
    fig.suptitle(title, fontsize=16)

    for i, result in enumerate(results_list):
        y_true = np.concatenate(result['y_true_all_folds'])
        y_pred_proba = np.concatenate(result['y_pred_proba_all_folds'])
        y_pred = (y_pred_proba > result['optimal_threshold']).astype(int)

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        cm_sum = cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.divide(cm.astype('float'), cm_sum,
                            out=np.zeros_like(cm.astype('float')),
                            where=cm_sum != 0)

        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                    ax=axes[0, i], cbar=False, annot_kws={"size": 14})
        axes[0, i].set_title(result['model_name'])
        axes[0, i].set_xlabel('Predicted Label')
        axes[0, i].set_ylabel('True Label')
        axes[0, i].set_xticklabels(['Non-Seizure', 'Seizure'])
        axes[0, i].set_yticklabels(['Non-Seizure', 'Seizure'], va='center')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    f_name = f'{subject_id}_optimized_confusion_matrices.png'
    plt.savefig(os.path.join(save_path, f_name))
    plt.close(fig)


def plot_learning_curves(model, X, y, groups, cv_splits, subject_id,
                         model_name, save_path):
    """Generate and save learning curves with group-aware CV

    Args:
        model: The classifier instance
        X (np.ndarray): The feature matrix
        y (np.ndarray): The target labels
        groups (np.ndarray): Group labels for each sample
        cv_splits: The cross-validation splitting strategy
        subject_id (str): Identifier for the plot
        model_name (str): Name of the model being evaluated
        save_path (str): Directory to save the plot
    """
    logging.info(f"Generating learning curve for {model_name}...")
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, groups=groups, cv=cv_splits, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10), scoring='f1_macro'
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(f'Learning Curve for {subject_id} ({model_name})')
    ax.set(xlabel="Training examples", ylabel="F1 Macro Score")
    ax.grid(True)

    ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std,
                    alpha=0.1, color="r")
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std,
                    alpha=0.1, color="g")
    ax.plot(train_sizes, train_scores_mean, 'o-', color="r",
            label="Training score")
    ax.plot(train_sizes, test_scores_mean, 'o-', color="g",
            label="Cross-validation score")

    ax.set_ylim(0.4, 1.01)

    ax.legend(loc="best")
    f_name = f'{subject_id}_{model_name}_learning_curve.png'
    plt.savefig(os.path.join(save_path, f_name))
    plt.close(fig)


def plot_feature_importances(model, feature_names, subject_id,
                             model_name, save_path, top_n=20):
    """Plot the top N feature importances for a given model

    Args:
        model: trained classifier with feature_importances_
        feature_names (list): The names of the features
        subject_id (str): identifier for the plot
        model_name (str): name of the model
        save_path (str): directory to save the plot
        top_n (int): number of top features to display
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    num_to_plot = min(top_n, len(importances))

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 10))
    title = (f'Top {num_to_plot} Feature Importances for '
             f'{subject_id} ({model_name})')
    ax.set_title(title)

    sns.barplot(x=importances[indices][:num_to_plot],
                y=np.array(feature_names)[indices][:num_to_plot],
                orient='h', ax=ax)

    ax.set(xlabel="Feature Importance", ylabel="Feature Name")
    plt.tight_layout()
    f_name = f'{subject_id}_{model_name}_feature_importances.png'
    plt.savefig(os.path.join(save_path, f_name))
    plt.close(fig)


def plot_summary_metrics_bar_chart(summary_df, save_path):
    """Create a bar chart comparing key metrics across all models

    Args:
        summary_df (pd.DataFrame): dataFrame with final model metrics
        save_path (str): directory to save the plot
    """
    logging.info("Generating summary metrics bar chart")
    plt.style.use('seaborn-v0_8-whitegrid')

    df_melted = summary_df.melt(
        id_vars='model_name',
        value_vars=['f1_seizure', 'precision_seizure', 'recall_seizure',
                    'roc_auc'],
        var_name='metric', value_name='score'
    )

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.barplot(data=df_melted, x='model_name', y='score', hue='metric',
                ax=ax)

    ax.set_title('Model Performance (at Optimized Thresholds)', fontsize=16)
    ax.set_xlabel('Model Type', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(title='Metric', loc='upper right')

    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.2f'),
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 9),
                    textcoords='offset points')

    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'summary_metrics_bar_chart.png'))
    plt.close(fig)


def plot_prediction_distribution(results_list, save_path):
    """Plot the distribution of predicted probabilities for each class

    Args:
        results_list (list): A list of result dictionaries for each model
        save_path (str): Directory to save the plot
    """
    logging.info("Generating prediction distribution plots")
    plt.style.use('seaborn-v0_8-whitegrid')

    n_models = len(results_list)
    fig, axes = plt.subplots(n_models, 1, figsize=(10, 6 * n_models),
                             squeeze=False)
    fig.suptitle('Distribution of Predicted Probabilities', fontsize=16)

    for i, result in enumerate(results_list):
        ax = axes[i, 0]
        y_true = np.concatenate(result['y_true_all_folds'])
        y_pred = np.concatenate(result['y_pred_proba_all_folds'])
        opt_thresh = result['optimal_threshold']

        plot_df = pd.DataFrame({'probability': y_pred, 'True Label': y_true})
        plot_df['True Label'] = plot_df['True Label'].map(
            {0: 'Non-Seizure', 1: 'Seizure'}
        )

        sns.histplot(data=plot_df, x='probability', hue='True Label',
                     kde=True, ax=ax, bins=50)
        ax.axvline(opt_thresh, color='red', linestyle='--', lw=2,
                   label=(f"Op. Threshold ({opt_thresh:.2f})\n"
                           f"Target: {result['optimization_target']}"))

        ax.set_title(result['model_name'])
        ax.set_xlabel('Predicted Probability of Seizure')
        ax.set_ylabel('Count')
        ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(save_path, 'prediction_distributions.png'))
    plt.close(fig)


def get_hrv_features(rr_intervals, sfreq):
    """Calculate Heart Rate Variability (HRV) features from RR intervals

    Args:
        rr_intervals (np.ndarray): Array of RR intervals in samples
        sfreq (float): The sampling frequency

    Returns:
        list: A list of HRV features. Includes a flag indicating if
              HRV was successfully computed
    """
    if len(rr_intervals) < 5:
        return [0] * 7
    sdnn = np.std(rr_intervals)
    rmssd = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))
    nn50 = np.sum(np.abs(np.diff(rr_intervals)) > (50 / 1000 * sfreq))
    pnn50 = (nn50 / len(rr_intervals)) * 100 if len(rr_intervals) > 0 else 0

    rr_times = np.cumsum(rr_intervals) / sfreq
    if len(rr_times) < 2:
        return [sdnn, rmssd, pnn50, 0, 0, 0, 1.0]
    f = interp1d(rr_times, rr_intervals, kind='cubic',
                 fill_value='extrapolate')
    t_new = np.arange(rr_times[0], rr_times[-1], 1/4)
    if len(t_new) < 2:
        return [sdnn, rmssd, pnn50, 0, 0, 0, 1.0]

    rr_interp = f(t_new)
    nperseg = min(len(rr_interp), 256)
    freqs, psd = welch(rr_interp, fs=4, nperseg=nperseg)

    lf_mask = (freqs >= 0.04) & (freqs < 0.15)
    hf_mask = (freqs >= 0.15) & (freqs < 0.4)
    lf_power = np.trapz(psd[lf_mask], freqs[lf_mask])
    hf_power = np.trapz(psd[hf_mask], freqs[hf_mask])
    lf_hf_ratio = lf_power / hf_power if hf_power > 0 else 0
    return [sdnn, rmssd, pnn50, lf_power, hf_power, lf_hf_ratio, 1.0]


def extract_advanced_features(eeg_data, ecg_data, sfreq):
    """Extract advanced time, frequency, and interaction features.

    Args:
        eeg_data (np.ndarray): EEG data (epochs, channels, times).
        ecg_data (np.ndarray): ECG data (epochs, channels, times).
        sfreq (float): Sampling frequency.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The complete feature matrix.
            - list: The names of all features.
    """
    n_epochs, eeg_channels = eeg_data.shape[0], eeg_data.shape[1]

    basic_eeg_names = ['mean', 'std', 'skew', 'kurtosis', 'rms', 'delta',
                       'theta', 'alpha', 'beta', 'high_beta']
    hjorth_names = ['hjorth_mobility', 'hjorth_complexity']
    eeg_feat_names = [f"EEG-Ch{i+1}-{name}"
                      for i in range(eeg_channels)
                      for name in basic_eeg_names + hjorth_names]
    hrv_feat_names = ['ECG-sdnn', 'ECG-rmssd', 'ECG-pnn50', 'ECG-lf_power',
                      'ECG-hf_power', 'ECG-lf_hf_ratio', 'ECG-hrv_computed']
    interaction_names = ['eeg_power_div_hf', 'eeg_power_div_lf']
    final_feature_names = eeg_feat_names + hrv_feat_names + interaction_names

    all_features = np.zeros((n_epochs, len(final_feature_names)))
    bands = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13),
             'beta': (13, 30), 'high_beta': (30, 40)}

    for i in range(n_epochs):
        epoch_eeg_features = []
        for j in range(eeg_channels):
            ch_data = eeg_data[i, j, :]
            epoch_eeg_features.extend([np.mean(ch_data), np.std(ch_data),
                                       skew(ch_data), kurtosis(ch_data),
                                       np.sqrt(np.mean(ch_data**2))])
            freqs, psd = welch(ch_data, sfreq, nperseg=int(sfreq*2))
            psd_norm = psd / np.sum(psd) if np.sum(psd) > 0 else psd
            for low, high in bands.values():
                mask = (freqs >= low) & (freqs <= high)
                epoch_eeg_features.append(np.trapz(psd_norm[mask],
                                                   freqs[mask]))
            diff1 = np.diff(ch_data)
            diff2 = np.diff(diff1)
            var_raw, var_d1, var_d2 = np.var(ch_data), np.var(diff1),
            np.var(diff2)
            mob = np.sqrt(var_d1 / var_raw) if var_raw > 0 else 0
            comp = np.sqrt(var_d2 / var_d1) / mob if var_d1 > 0 and mob > 0 else 0
            epoch_eeg_features.extend([mob, comp])

        # HRV Features
        ecg_signal = ecg_data[i, 0, :]
        ecg_height = np.mean(ecg_signal) + np.std(ecg_signal)
        peaks, _ = find_peaks(ecg_signal, height=ecg_height,
                              distance=0.4*sfreq)
        epoch_hrv_features = get_hrv_features(np.diff(peaks), sfreq)

        total_eeg_power = np.sum([f**2 for f in epoch_eeg_features])
        hf_power, lf_power = epoch_hrv_features[4], epoch_hrv_features[3]
        interaction_feats = [
            total_eeg_power / hf_power if hf_power > 0 else 0,
            total_eeg_power / lf_power if lf_power > 0 else 0
        ]
        all_features[i, :] = np.concatenate([epoch_eeg_features,
                                             epoch_hrv_features,
                                             interaction_feats])
    return all_features, final_feature_names


def get_args():
    """Parse command-line arguments

    Returns:
        argparse.Namespace: An object containing the script's arguments
    """
    desc = "Final Comparative Pipeline - Advanced Features & High Recall"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--base_path', required=True,
                        help='Root folder for the BIDS dataset')
    parser.add_argument('--results_dir', type=str,
                        default='final_comparison_model_zoomed',
                        help='Folder to save final model results')
    parser.add_argument('--n_splits', type=int, default=5,
                        help='Number of splits for Group K-Fold')
    parser.add_argument('--target_recall', type=float, default=0.90,
                        help='Minimum recall (0.0-1.0) to optimize for.')
    return parser.parse_args()


def load_all_subject_data(args):
    """Load, align, and extract features for all subjects.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Combined feature matrix (X).
            - np.ndarray: Combined labels (y).
            - np.ndarray: Combined group identifiers.
            - list: The names of the features.
            Returns (None, None, None, None) if no data is loaded.
    """
    logging.info("--- Loading and processing data for all subjects ---")
    eeg_path = os.path.join(args.base_path, "derivatives",
                            'eeg_paired_by_subject')
    ecg_path = os.path.join(args.base_path, "derivatives",
                            'ecg_paired_by_subject')
    subject_dirs = sorted([d for d in glob.glob(os.path.join(eeg_path,
                           "sub-*")) if os.path.isdir(d)])
    master_X, master_y, master_groups = [], [], []
    feature_names = None

    for subject_dir in subject_dirs:
        subject_id = os.path.basename(subject_dir)
        logging.info(f"Processing {subject_id}...")
        eeg_files = glob.glob(os.path.join(eeg_path, subject_id, "**",
                              "*.fif"), recursive=True)
        ecg_files = glob.glob(os.path.join(ecg_path, subject_id, "**",
                              "*.fif"), recursive=True)
        if not eeg_files or not ecg_files:
            continue
        try:
            eeg_epochs = mne.concatenate_epochs(
                [mne.read_epochs(f, preload=True, verbose=False)
                 for f in eeg_files])
            ecg_epochs = mne.concatenate_epochs(
                [mne.read_epochs(f, preload=True, verbose=False)
                 for f in ecg_files])
        except Exception as e:
            logging.warning(f"Skipping {subject_id} due to error: {e}")
            continue

        eeg_meta = eeg_epochs.metadata.assign(original_index=np.arange(
            len(eeg_epochs)))
        ecg_meta = ecg_epochs.metadata.assign(original_index=np.arange(
            len(ecg_epochs)))
        aligned_meta = pd.merge(eeg_meta, ecg_meta, on='unique_epoch_id',
                                how='inner', suffixes=('_eeg', '_ecg'))
        if aligned_meta.empty:
            continue

        eeg_idx = aligned_meta['original_index_eeg'].values
        ecg_idx = aligned_meta['original_index_ecg'].values
        aligned_eeg = eeg_epochs[eeg_idx].get_data(copy=True)
        aligned_ecg = ecg_epochs[ecg_idx].get_data(copy=True)

        X_sub, current_names = extract_advanced_features(
            aligned_eeg, aligned_ecg, eeg_epochs.info['sfreq'])
        if feature_names is None:
            feature_names = current_names

        master_X.append(X_sub)
        master_y.append(aligned_meta['seizure_label_eeg'].values)
        master_groups.extend([subject_id] * len(aligned_meta))
        logging.info(f"Added {len(aligned_meta)} windows from {subject_id}.")
        gc.collect()

    if not master_y:
        return None, None, None, None

    X_all = np.vstack(master_X)
    y_all = np.concatenate(master_y)
    groups_all = np.array(master_groups)
    X_all = np.nan_to_num(X_all)
    n_subjects = len(np.unique(groups_all))
    logging.info(f"Total loaded windows: {len(y_all)} from {n_subjects} subs.")
    return X_all, y_all, groups_all, feature_names


def main():
    """Main training and evaluation pipeline"""
    args = get_args()
    results_path = os.path.join(args.base_path, "derivatives",
                                args.results_dir)
    os.makedirs(results_path, exist_ok=True)

    log_file = os.path.join(results_path,
                            'final_comparison_pipeline_run.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file, mode='w'),
                  logging.StreamHandler()]
    )

    msg = f"Final Pipeline | Target Recall: {args.target_recall*100:.1f}"
    logging.info(msg)
    X_all, y, groups, feature_names = load_all_subject_data(args)

    if y is None:
        logging.info("Could not proceed.")
        return

    eeg_indices = [i for i, name in enumerate(feature_names)
                   if name.startswith('EEG')]
    ecg_indices = [i for i, name in enumerate(feature_names)
                   if name.startswith('ECG')]
    experiments = {
        "EEG_Only (Advanced)": X_all[:, eeg_indices],
        "ECG_Only (HRV)": X_all[:, ecg_indices],
        "Fused_EEG_ECG (Advanced)": X_all
    }
    final_results_list = []

    for model_name, X in experiments.items():
        logging.info(f"\n Starting Experiment: {model_name}")
        cv = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True,
                                  random_state=42)
        cv_splits_list = list(cv.split(X, y, groups))

        lc_model = RandomForestClassifier(
            n_estimators=150, class_weight='balanced', random_state=42,
            n_jobs=-1, max_depth=15, min_samples_leaf=5
        )
        plot_learning_curves(lc_model, StandardScaler().fit_transform(X), y,
                             groups, cv_splits_list,
                             'All_Subjects_Generalized', model_name,
                             results_path)

        y_true_all, y_pred_proba_all = [], []
        for fold, (train_idx, test_idx) in enumerate(cv_splits_list):
            n_subs = len(np.unique(groups[test_idx]))
            logging.info(f"Fold {fold+1}/{args.n_splits}: "
                         f"Testing on {n_subs} subjects")
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            scaler = StandardScaler().fit(X_train)
            X_train_s = scaler.transform(X_train)
            X_test_s = scaler.transform(X_test)

            rf = RandomForestClassifier(
                n_estimators=150, class_weight='balanced', random_state=42,
                n_jobs=-1, max_depth=15, min_samples_leaf=5
            )
            rf.fit(X_train_s, y_train)
            y_pred_proba_all.append(rf.predict_proba(X_test_s)[:, 1])
            y_true_all.append(y_test)

        y_true_flat = np.concatenate(y_true_all)
        y_pred_flat = np.concatenate(y_pred_proba_all)
        precisions, recalls, thresholds = precision_recall_curve(
            y_true_flat, y_pred_flat)

        possible_indices = np.where(recalls[:-1] >= args.target_recall)[0]
        if len(possible_indices) == 0:
            logging.warning(f"Target recall {args.target_recall} not "
                            f"achievable for {model_name}. "
                            "Falling back to max F1-score.")
            f1_scores = ((2 * recalls[:-1] * precisions[:-1]) /
                         (recalls[:-1] + precisions[:-1]))
            optimal_idx = np.nanargmax(f1_scores)
            opt_target = "Max F1"
        else:
            best_idx_for_recall = np.argmax(precisions[possible_indices])
            optimal_idx = possible_indices[best_idx_for_recall]
            opt_target = f"Recall>={args.target_recall}"

        optimal_thresh = thresholds[min(optimal_idx, len(thresholds)-1)]

        y_pred_opt = (y_pred_flat > optimal_thresh).astype(int)
        prec, recall, f1, _ = precision_recall_fscore_support(
            y_true_flat, y_pred_opt, average='binary', zero_division=0)
        final_auc = roc_auc_score(y_true_flat, y_pred_flat)

        final_results_list.append({
            'model_name': model_name, 'roc_auc': final_auc,
            'f1_seizure': f1, 'recall_seizure': recall,
            'precision_seizure': prec, 'optimal_threshold': optimal_thresh,
            'optimization_target': opt_target,
            'y_true_all_folds': y_true_all,
            'y_pred_proba_all_folds': y_pred_proba_all,
            'pr_curve': {'precision': precisions, 'recall': recalls}
        })
        logging.info(f"Optimized Metrics for {model_name}: F1={f1:.3f}, "
                     f"Recall={recall:.3f}, Precision={prec:.3f}")

    logging.info("\n--- Generating Final Comparison Plots ---")
    plot_id = 'All_Subjects_HighRecall_Comparison'
    plot_roc_curves(final_results_list, plot_id, results_path)
    plot_precision_recall_curves(final_results_list, plot_id, results_path)
    plot_optimized_confusion_matrices(final_results_list, plot_id,
                                      results_path)
    plot_prediction_distribution(final_results_list, results_path)

    summary_df = pd.DataFrame(final_results_list).drop(
        columns=['y_true_all_folds', 'y_pred_proba_all_folds', 'pr_curve'])
    summary_df.to_csv(
        os.path.join(results_path, 'high_recall_summary_metrics.csv'),
        index=False)

    plot_summary_metrics_bar_chart(summary_df, results_path)
    logging.info("\n--- Final Model Performance Summary ---")
    logging.info(summary_df.to_string())

    logging.info("\n--- Training and Saving Final Generalized Models ---")
    for model_name, X_final in experiments.items():
        logging.info(f"Training final version of: {model_name}")
        current_names = np.array(feature_names)
        if "EEG_Only" in model_name:
            current_feature_names = list(current_names[eeg_indices])
        elif "ECG_Only" in model_name:
            current_feature_names = list(current_names[ecg_indices])
        else:
            current_feature_names = feature_names

        scaler_final = StandardScaler().fit(X_final)
        X_final_scaled = scaler_final.transform(X_final)

        rf_final = RandomForestClassifier(
            n_estimators=150, class_weight='balanced', random_state=42,
            n_jobs=-1, max_depth=15, min_samples_leaf=5
        )
        rf_final.fit(X_final_scaled, y)

        logging.info(f"Plotting feature importances for {model_name}")
        plot_feature_importances(rf_final, current_feature_names,
                                 'Generalized_Model', model_name,
                                 results_path)

        dump(rf_final, os.path.join(results_path,
                                    f'final_model_{model_name}.joblib'))
        dump(scaler_final, os.path.join(results_path,
                                        f'final_scaler_{model_name}.joblib'))
        logging.info(f"Saved final model and scaler for {model_name}.")

    logging.info("\nComplete")


if __name__ == "__main__":
    main()

