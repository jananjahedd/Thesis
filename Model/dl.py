#!/usr/bin/env python3
import os
import sys
import gc
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import mixed_precision
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, LSTM, Dense, concatenate, Flatten, Dropout
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

LOG_FILE = 'training_log.txt'
sys.stdout = open(LOG_FILE, 'w')

# -------------------------------
# 1) GPU / Mixed-Precision Setup
# -------------------------------
print("=== GPU / Mixed-Precision Setup ===")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"Detected GPU(s): {[g.name for g in gpus]}")
    mixed_precision.set_global_policy('mixed_float16')
    print("Mixed precision enabled (mixed_float16).")
    strategy = tf.distribute.MirroredStrategy()
    print(f"Using MirroredStrategy with {strategy.num_replicas_in_sync} replicas.")
else:
    print("No GPU found; using CPU.")
    strategy = tf.distribute.get_strategy()

# -------------------------------
# 2) Model Definition
# -------------------------------
def create_model(input_shape_eeg, input_shape_ecg, use_ecg=True):
    with strategy.scope():
        input_eeg = Input(shape=input_shape_eeg, name='eeg_input')
        x_eeg = Conv1D(64, 3, activation='relu')(input_eeg)
        x_eeg = Conv1D(128, 3, activation='relu')(x_eeg)
        x_eeg = LSTM(64, return_sequences=True)(x_eeg)
        x_eeg = Flatten()(x_eeg)
        x_eeg = Dense(32, activation='relu')(x_eeg)
        x_eeg = Dropout(0.2)(x_eeg)

        if use_ecg:
            input_ecg = Input(shape=input_shape_ecg, name='ecg_input')
            x_ecg = Conv1D(32, 3, activation='relu')(input_ecg)
            x_ecg = Flatten()(x_ecg)
            x_ecg = Dense(32, activation='relu')(x_ecg)
            x_ecg = Dropout(0.2)(x_ecg)
            x = concatenate([x_eeg, x_ecg], name='fusion')
        else:
            x = x_eeg

        x = Dense(64, activation='relu')(x)
        x = Dropout(0.3)(x)
        output = Dense(4, activation='softmax', name='output')(x)

        model = Model(
            inputs=[input_eeg, input_ecg] if use_ecg else input_eeg,
            outputs=output
        )
        model.compile(
            optimizer=Adam(1e-3),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    return model

# -------------------------------
# 3) Training Utilities
# -------------------------------
def plot_history(history, fold):
    plt.figure(figsize=(12, 5))
    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title(f'Fold {fold+1} Accuracy')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'Fold {fold+1} Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.tight_layout()
    plt.savefig(f'fold_{fold+1}_performance.png')
    plt.close()

def train_and_evaluate(Xeeg, Xecg, y, groups, use_ecg=True, num_folds=5):
    print("\n=== Starting GroupKFold Training ===")
    print(f"Total samples: {Xeeg.shape[0]}, EEG shape: {Xeeg.shape[1:]}{' + ECG ' + str(Xecg.shape[1:]) if use_ecg else ''}")
    kf = GroupKFold(n_splits=num_folds)
    all_metrics = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(Xeeg, y, groups)):
        print(f"\n--- Fold {fold+1}/{num_folds} ---")
        # Slice
        Xeeg_tr, Xeeg_val = Xeeg[train_idx], Xeeg[val_idx]
        y_tr, y_val       = y[train_idx], y[val_idx]

        # Transpose to (batch, time, channels)
        Xeeg_tr = Xeeg_tr.transpose(0, 2, 1)
        Xeeg_val = Xeeg_val.transpose(0, 2, 1)
        seq_len, n_chan_eeg = Xeeg_tr.shape[1], Xeeg_tr.shape[2]

        if use_ecg:
            Xecg_tr = Xecg[train_idx].transpose(0, 2, 1)
            Xecg_val = Xecg[val_idx].transpose(0, 2, 1)
            _, n_chan_ecg = Xecg_tr.shape[1], Xecg_tr.shape[2]

        # Scale per‐channel: flatten (samples*time, channels), scale, reshape back
        print("Scaling EEG data...")
        scaler = StandardScaler()
        flat = Xeeg_tr.reshape(-1, n_chan_eeg)
        flat = scaler.fit_transform(flat)
        Xeeg_tr = flat.reshape(len(train_idx), seq_len, n_chan_eeg)

        flat_val = Xeeg_val.reshape(-1, n_chan_eeg)
        flat_val = scaler.transform(flat_val)
        Xeeg_val = flat_val.reshape(len(val_idx), seq_len, n_chan_eeg)

        if use_ecg:
            print("Scaling ECG data...")
            scaler_ecg = StandardScaler()
            flat_ecg = Xecg_tr.reshape(-1, n_chan_ecg)
            flat_ecg = scaler_ecg.fit_transform(flat_ecg)
            Xecg_tr = flat_ecg.reshape(len(train_idx), seq_len, n_chan_ecg)

            flat_ecg_val = Xecg_val.reshape(-1, n_chan_ecg)
            flat_ecg_val = scaler_ecg.transform(flat_ecg_val)
            Xecg_val = flat_ecg_val.reshape(len(val_idx), seq_len, n_chan_ecg)

        # Build model
        print("Building model...")
        model = create_model(
            input_shape_eeg=(seq_len, n_chan_eeg),
            input_shape_ecg=(seq_len, n_chan_ecg) if use_ecg else None,
            use_ecg=use_ecg
        )

        # Callbacks
        cb = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5)
        ]

        # Fit
        print("Training...")
        if use_ecg:
            history = model.fit(
                [Xeeg_tr, Xecg_tr], y_tr,
                validation_data=([Xeeg_val, Xecg_val], y_val),
                epochs=50, batch_size=32, callbacks=cb, verbose=1
            )
        else:
            history = model.fit(
                Xeeg_tr, y_tr,
                validation_data=(Xeeg_val, y_val),
                epochs=50, batch_size=32, callbacks=cb, verbose=1
            )

        # Evaluate
        print("Evaluating...")
        preds = model.predict([Xeeg_val, Xecg_val] if use_ecg else Xeeg_val)
        pred_classes = np.argmax(preds, axis=1)
        metrics = {
            'accuracy': accuracy_score(y_val, pred_classes),
            'precision': precision_score(y_val, pred_classes, average='macro'),
            'recall': recall_score(y_val, pred_classes, average='macro'),
            'f1': f1_score(y_val, pred_classes, average='macro')
        }
        print(f"Fold {fold+1} metrics: {metrics}")
        if metrics['accuracy'] > 0.99:
            print("⚠️ Possible overfitting or leakage detected!")
        all_metrics.append(metrics)

        plot_history(history, fold)

        # Cleanup
        tf.keras.backend.clear_session()
        del model, Xeeg_tr, Xeeg_val, y_tr, y_val, preds
        if use_ecg:
            del Xecg_tr, Xecg_val
        gc.collect()

    # Aggregate
    avg = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
    print(f"\nAverage across folds: {avg}")

# -------------------------------
# 4) Main Execution
# -------------------------------
if __name__ == '__main__':
    print("=== Loading data ===")
    Xeeg = np.nan_to_num(np.load('/Users/jananjahed/Desktop/BP/ds005873/dl_ready/Xeeg.npy'))
    Xecg = np.nan_to_num(np.load('/Users/jananjahed/Desktop/BP/ds005873/dl_ready/Xecg.npy'))
    y    = np.load('/Users/jananjahed/Desktop/BP/ds005873/dl_ready/y.npy')
    meta = pd.read_csv('/Users/jananjahed/Desktop/BP/ds005873/dl_ready/meta.csv')
    groups = meta['epoch_type'] + '_' + meta['orig_idx'].astype(str)
    print(f"Loaded: Xeeg {Xeeg.shape}, Xecg {Xecg.shape}, y {y.shape}, groups {groups.shape}")

    choice = input("Run EEG-only (e) or EEG+ECG (b)? ").strip().lower()
    use_ecg = (choice == 'b')
    print(f"\n--- Starting {'EEG+ECG' if use_ecg else 'EEG-only'} training ---")

    train_and_evaluate(Xeeg, Xecg, y, groups, use_ecg=use_ecg)

    sys.stdout.close()
    sys.stdout = sys.__stdout__
    print(f"Logs written to {LOG_FILE}")
