import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, concatenate, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import gc
import matplotlib.pyplot as plt
import os
import sys

LOG_FILE = 'training_log.txt'
sys.stdout = open(LOG_FILE, 'w')


def create_model(input_shape_eeg, input_shape_ecg, use_ecg=True):
    input_eeg = Input(shape=input_shape_eeg, name='eeg_input')
    conv1_eeg = Conv1D(filters=64, kernel_size=3, activation='relu')(input_eeg)
    conv2_eeg = Conv1D(filters=128, kernel_size=3, activation='relu')(conv1_eeg)
    lstm_eeg = LSTM(64, return_sequences=True)(conv2_eeg)
    flatten_eeg = Flatten()(lstm_eeg)
    dense_eeg = Dense(32, activation='relu')(flatten_eeg)
    dropout_eeg = Dropout(0.2)(dense_eeg)

    if use_ecg:
        input_ecg = Input(shape=input_shape_ecg, name='ecg_input')
        conv1_ecg = Conv1D(filters=32, kernel_size=3, activation='relu')(input_ecg)
        flatten_ecg = Flatten()(conv1_ecg)
        dense_ecg = Dense(32, activation='relu')(flatten_ecg)
        dropout_ecg = Dropout(0.2)(dense_ecg)

        merged = concatenate([dropout_eeg, dropout_ecg], name='fusion')
        dense_merged = Dense(64, activation='relu')(merged)
        dropout_merged = Dropout(0.3)(dense_merged)
        output = Dense(4, activation='softmax', name='output')(dropout_merged)
        model = Model(inputs=[input_eeg, input_ecg], outputs=output)
    else:
        output = Dense(4, activation='softmax', name='output')(dropout_eeg)
        model = Model(inputs=input_eeg, outputs=output)

    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def plot_history(history, fold):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'Fold {fold+1} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Fold {fold+1} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'fold_{fold+1}_performance.png')
    plt.close()


def train_and_evaluate(Xeeg, Xecg, y, groups, use_ecg=True, num_folds=5):
    kf = GroupKFold(n_splits=num_folds)
    all_metrics = []

    for fold, (train_index, val_index) in enumerate(kf.split(Xeeg, y, groups)):
        print(f"Fold {fold + 1}/{num_folds}")

        Xeeg_train, Xeeg_val = Xeeg[train_index], Xeeg[val_index]
        y_train, y_val = y[train_index], y[val_index]

        Xeeg_train = Xeeg_train.transpose(0, 2, 1)
        Xeeg_val = Xeeg_val.transpose(0, 2, 1)

        if use_ecg:
            Xecg_train, Xecg_val = Xecg[train_index], Xecg[val_index]
            Xecg_train = Xecg_train.transpose(0, 2, 1)
            Xecg_val = Xecg_val.transpose(0, 2, 1)

        scaler_eeg = StandardScaler()
        Xeeg_train_flat = Xeeg_train.reshape(-1, Xeeg_train.shape[-1])
        Xeeg_val_flat = Xeeg_val.reshape(-1, Xeeg_val.shape[-1])
        scaler_eeg.fit(Xeeg_train_flat)
        Xeeg_train = scaler_eeg.transform(Xeeg_train_flat).reshape(Xeeg_train.shape)
        Xeeg_val = scaler_eeg.transform(Xeeg_val_flat).reshape(Xeeg_val.shape)

        if use_ecg:
            scaler_ecg = StandardScaler()
            Xecg_train_flat = Xecg_train.reshape(-1, Xecg_train.shape[-1])
            Xecg_val_flat = Xecg_val.reshape(-1, Xecg_val.shape[-1])
            scaler_ecg.fit(Xecg_train_flat)
            Xecg_train = scaler_ecg.transform(Xecg_train_flat).reshape(Xecg_train.shape)
            Xecg_val = scaler_ecg.transform(Xecg_val_flat).reshape(Xecg_val.shape)

        input_shape_eeg = Xeeg_train.shape[1:]
        input_shape_ecg = Xecg_train.shape[1:] if use_ecg else (0, 0)

        model = create_model(input_shape_eeg, input_shape_ecg, use_ecg)

        early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(factor=0.5, patience=5)

        if use_ecg:
            history = model.fit([Xeeg_train, Xecg_train], y_train,
                                validation_data=([Xeeg_val, Xecg_val], y_val),
                                epochs=50, batch_size=32,
                                callbacks=[early_stopping, reduce_lr],
                                verbose=1)
        else:
            history = model.fit(Xeeg_train, y_train,
                                validation_data=(Xeeg_val, y_val),
                                epochs=50, batch_size=32,
                                callbacks=[early_stopping, reduce_lr],
                                verbose=1)

        y_pred = model.predict([Xeeg_val, Xecg_val] if use_ecg else Xeeg_val)
        y_pred_classes = np.argmax(y_pred, axis=1)

        fold_metrics = {
            'accuracy': accuracy_score(y_val, y_pred_classes),
            'precision': precision_score(y_val, y_pred_classes, average='macro'),
            'recall': recall_score(y_val, y_pred_classes, average='macro'),
            'f1': f1_score(y_val, y_pred_classes, average='macro')
        }
        print(f"Fold {fold + 1} Metrics: {fold_metrics}")

        if fold_metrics['accuracy'] > 0.99:
            print("⚠️  Possible overfitting or data leakage detected!")

        all_metrics.append(fold_metrics)
        plot_history(history, fold)

        tf.keras.backend.clear_session()
        del model, Xeeg_train, Xeeg_val, y_train, y_val, y_pred
        if use_ecg:
            del Xecg_train, Xecg_val
        gc.collect()

    avg_metrics = {key: np.mean([m[key] for m in all_metrics]) for key in all_metrics[0]}
    print(f"\nAverage Metrics across {num_folds} folds: {avg_metrics}")


sys.stdout.close()
sys.stdout = sys.__stdout__

if __name__ == '__main__':
    Xeeg = np.load('/scratch/s5107318/BP/dl_ready/Xeeg.npy')
    Xecg = np.load('/scratch/s5107318/BP/dl_ready/Xecg.npy')
    y = np.load('/scratch/s5107318/BP/dl_ready/y.npy')
    meta = pd.read_csv('/scratch/s5107318/BP/dl_ready/meta.csv')
    groups = meta['epoch_type'] + '_' + meta['orig_idx'].astype(str)

    Xeeg = np.nan_to_num(Xeeg)
    Xecg = np.nan_to_num(Xecg)

    model_choice = input("Run EEG-only (e) or EEG+ECG (b) model? ").strip().lower()
    use_ecg = model_choice == 'b'

    print(f"\n--- Running {'EEG+ECG' if use_ecg else 'EEG-only'} Model ---")
    train_and_evaluate(Xeeg, Xecg, y, groups, use_ecg=use_ecg)

    print(f"\nTraining logs saved to {LOG_FILE}")
