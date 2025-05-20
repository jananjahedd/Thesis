#!/usr/bin/env python3

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, roc_curve, auc,
                             precision_recall_curve, average_precision_score)
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Reshape, Lambda, Concatenate
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, AveragePooling1D
from tensorflow.keras.layers import LSTM, Bidirectional, GRU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
import random
import logging
from datetime import datetime
import sys
import json 
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils import class_weight
import gc 

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    print("Warning: imblearn not available. SMOTE will not be used for oversampling.")
    SMOTE_AVAILABLE = False

SEED = 42
args = None
logger = None
output_dir = None 

def set_seeds(seed_value=SEED):
    global logger 
    os.environ['PYTHONHASHSEED'] = str(seed_value)
 
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.compat.v1.set_random_seed(seed_value) 
    if logger: 
        logger.info(f"Seeds set to: {seed_value}")
    else:
        print(f"Seeds set to: {seed_value} (logger not yet initialized)")


def get_arg_parser():
    parser = argparse.ArgumentParser(description="Advanced multiscale CNN-LSTM model with HP Tuning & CV (TF1.x GPU compatible)")
    parser.add_argument('--data_path', type=str, default='/scratch/s5107318/BP/ds005873/derivatives/fused_dataset', help='Path to the dataset directory')
    parser.add_argument('--output_path', type=str, default='/scratch/s5107318/BP/Model', help='Base path for saving model results configurations')
    parser.add_argument('--model_type', type=str, choices=['eeg', 'fused'], default='fused', help='Model type: "eeg" or "fused"')
    parser.add_argument('--category', type=str, choices=['left_bte_crosstop', 'right_bte_crosstop', 'both_bte_no_crosstop', 'all'], default='all', help='BTE category (default: all)')
    parser.add_argument('--epochs', type=int, default=100, help='Max training epochs for main training/CV folds')
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
    parser.add_argument('--patience', type=int, default=15, help='Patience for early stopping for main training/CV folds')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--l1_conv', type=float, default=1e-5, help='L1 regularization for Conv layers')
    parser.add_argument('--l2_conv', type=float, default=1e-4, help='L2 regularization for Conv layers')
    parser.add_argument('--l1_dense', type=float, default=1e-5, help='L1 regularization for Dense layers')
    parser.add_argument('--l2_dense', type=float, default=1e-4, help='L2 regularization for Dense layers')
    parser.add_argument('--binary', action='store_true', help='Use binary classification')
    parser.add_argument('--focus_class', type=str, default='onset', help='Focus class for binary mode')
    parser.add_argument('--threshold', type=float, default=0.3, help='Classification threshold for binary prediction')
    parser.add_argument('--kfold', type=int, default=0, help='Number of K-Fold CV splits (0 or 1 to disable CV)')
    parser.add_argument('--tune_hyperparams', action='store_true', help='Perform hyperparameter tuning')
    parser.add_argument('--tune_epochs', type=int, default=15, help='Epochs for each HP tuning trial')
    parser.add_argument('--tune_patience', type=int, default=5, help='Patience for HP tuning trial early stopping')
    parser.add_argument('--gpu', type=str, default='0', help='Which GPU to use (e.g., "0", "0,1", or empty for CPU/"all visible")')
    parser.add_argument('--seed', type=int, default=SEED, help='Random seed for reproducibility')
    return parser


def get_simplified_category_name(category_arg):
    if category_arg == 'left_bte_crosstop': return 'bteL'
    if category_arg == 'right_bte_crosstop': return 'bteR'
    if category_arg == 'both_bte_no_crosstop': return 'bothBTE'
    if category_arg == 'all': return 'allCategories'
    return category_arg.replace("_", "")

def setup_logging_and_output_dir(cmd_args):
    global logger, output_dir
    simplified_cat_name = get_simplified_category_name(cmd_args.category)
    config_specific_base_folder_name = f"{cmd_args.model_type}_{simplified_cat_name}_results"
    config_specific_base_path = os.path.join(cmd_args.output_path, config_specific_base_folder_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_specific_subfolder_name = f"run_{timestamp}"
    if cmd_args.tune_hyperparams: run_specific_subfolder_name += "_hptuned"
    if cmd_args.kfold > 1: run_specific_subfolder_name += f"_{cmd_args.kfold}foldcv"
    output_dir = os.path.join(config_specific_base_path, run_specific_subfolder_name)
    os.makedirs(output_dir, exist_ok=True)
    log_filename = os.path.join(output_dir, "model_training_run.log")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', handlers=[logging.FileHandler(log_filename, mode='w'), logging.StreamHandler(sys.stdout)])
    logger = logging.getLogger()
    logger.info(f"Output directory for this run: {output_dir}")
    logger.info(f"Script arguments: {vars(cmd_args)}")

def safe_load_array(file_path):
    try: return np.load(file_path)
    except ValueError:
        logger.warning(f"Loading {file_path} with allow_pickle=True - security risk")
        data = np.load(file_path, allow_pickle=True)
        if data.dtype == object:
            try: return data.astype(float)
            except (ValueError, TypeError): return data
        return data

def load_dataset(data_path, category, modality='eeg'):
    global args, logger
    logger.info(f"Loading {modality} dataset for category: {category}")
    dataset_dir = os.path.join(data_path, f"eeg_ecg_{category}")
    if not os.path.exists(dataset_dir): raise FileNotFoundError(f"Dataset dir not found: {dataset_dir}")
    dataset = {}
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(dataset_dir, split)
        if not os.path.exists(split_dir):
            is_req = not (split == 'val' and (args.kfold > 1 or (not args.tune_hyperparams and args.kfold <= 1)))
            if is_req and split != 'test':
                 if not (split == 'test' and args.kfold > 1):
                    raise FileNotFoundError(f"Required split dir not found: {split_dir}")
            logger.warning(f"Split dir {split_dir} not found. Marked as None."); dataset[split] = None; continue
        dataset[split] = {}
        data_key_to_load = 'fused' if modality == 'fused' else 'eeg'
        if modality == 'eeg':
            eeg_path = os.path.join(split_dir, 'eeg.npy')
            if os.path.exists(eeg_path): dataset[split]['eeg'] = safe_load_array(eeg_path)
            else:
                fused_path = os.path.join(split_dir, 'fused.npy')
                if os.path.exists(fused_path):
                    fused_data = safe_load_array(fused_path); dataset[split]['eeg'] = fused_data[:, :2, :] 
                    logger.info(f"Extracted EEG from fused for {split} eeg shape: {dataset[split]['eeg'].shape}")
                else: raise FileNotFoundError(f"Neither eeg.npy nor fused.npy found in {split_dir} for EEG mode")
            if dataset[split].get('eeg') is not None: logger.info(f"{split} eeg shape: {dataset[split]['eeg'].shape}")
        elif modality == 'fused':
            fused_path = os.path.join(split_dir, 'fused.npy')
            if os.path.exists(fused_path): dataset[split]['fused'] = safe_load_array(fused_path)
            else:
                eeg_path, ecg_path = os.path.join(split_dir, 'eeg.npy'), os.path.join(split_dir, 'ecg.npy')
                if os.path.exists(eeg_path) and os.path.exists(ecg_path):
                    eeg_data, ecg_data = safe_load_array(eeg_path), safe_load_array(ecg_path)
                    if ecg_data.ndim == 2: ecg_data = np.expand_dims(ecg_data, axis=1)
                    if eeg_data.shape[0] == ecg_data.shape[0] and eeg_data.shape[2] == ecg_data.shape[2]:
                        dataset[split]['fused'] = np.concatenate([eeg_data, ecg_data], axis=1)
                        logger.info(f"Created fused data for {split} fused shape: {dataset[split]['fused'].shape}")
                    else: raise ValueError(f"Shape mismatch for fusion: EEG {eeg_data.shape}, ECG {ecg_data.shape} in {split_dir}")
                else: raise FileNotFoundError(f"Cannot create fused: eeg/ecg.npy missing in {split_dir}")
            if dataset[split].get('fused') is not None: logger.info(f"{split} fused shape: {dataset[split]['fused'].shape}")
        metadata_path = os.path.join(split_dir, 'metadata.csv')
        dataset[split]['metadata'] = pd.read_csv(metadata_path) if os.path.exists(metadata_path) else None
        y_path = os.path.join(split_dir, 'y.npy')
        if os.path.exists(y_path): dataset[split]['y'] = safe_load_array(y_path)
        elif dataset[split]['metadata'] is not None and 'epoch_type_class' in dataset[split]['metadata'].columns:
            class_mapping = {'non_seizure':0,'preictal':1,'ictal':2,'onset':3}
            dataset[split]['y'] = dataset[split]['metadata']['epoch_type_class'].map(class_mapping).values
        else: raise FileNotFoundError(f"Labels not found in {split_dir}")
        if dataset[split].get('y') is not None: logger.info(f"{split} y shape: {dataset[split]['y'].shape}")
    return dataset

def preprocess_labels_for_split(data_split_dict, binary_flag, focus_class_name_str):
    if data_split_dict is None or 'y' not in data_split_dict or data_split_dict['y'] is None:
        return data_split_dict
    class_mapping={'non_seizure':0,'preictal':1,'ictal':2,'onset':3}
    y_original=data_split_dict['y']
    y_processed=np.array([class_mapping.get(str(label).lower(),class_mapping['non_seizure']) for label in y_original]) if not np.issubdtype(y_original.dtype,np.number) else y_original.astype(int)
    data_split_dict['y_processed']=y_processed
    if binary_flag:
        focus_class_idx=class_mapping.get(focus_class_name_str,3)
        data_split_dict['y_binary_processed']=(y_processed==focus_class_idx).astype(int)
        counts=np.bincount(data_split_dict['y_binary_processed'],minlength=2)
        logger.info(f"Binary labels (focus: {focus_class_name_str}): Non-{focus_class_name_str} (0): {counts[0]}, {focus_class_name_str} (1): {counts[1]}")
    return data_split_dict

def create_multiscale_residual_model(input_shape, num_classes, current_run_args):
    inputs = Input(shape=input_shape)
    x = Lambda(lambda t: tf.transpose(t, perm=[0, 2, 1]))(inputs)
    dropout_rate_val,l1_conv_val,l2_conv_val,l1_dense_val,l2_dense_val=current_run_args.dropout_rate,current_run_args.l1_conv,current_run_args.l2_conv,current_run_args.l1_dense,current_run_args.l2_dense
    path1=Conv1D(64,3,padding='same',activation='relu',kernel_regularizer=l1_l2(l1_conv_val,l2_conv_val))(x)
    path1=BatchNormalization()(path1);path1=MaxPooling1D(2)(path1);path1=Dropout(dropout_rate_val/2)(path1)
    path1=Conv1D(128,3,padding='same',activation='relu',kernel_regularizer=l1_l2(l1_conv_val,l2_conv_val))(path1)
    path1=BatchNormalization()(path1);path1=MaxPooling1D(2)(path1);path1=Dropout(dropout_rate_val/2)(path1)
    path2=Conv1D(64,7,padding='same',activation='relu',kernel_regularizer=l1_l2(l1_conv_val,l2_conv_val))(x)
    path2=BatchNormalization()(path2);path2=MaxPooling1D(2)(path2);path2=Dropout(dropout_rate_val/2)(path2)
    path2=Conv1D(128,7,padding='same',activation='relu',kernel_regularizer=l1_l2(l1_conv_val,l2_conv_val))(path2)
    path2=BatchNormalization()(path2);path2=MaxPooling1D(2)(path2);path2=Dropout(dropout_rate_val/2)(path2)
    path3=Conv1D(64,15,padding='same',activation='relu',kernel_regularizer=l1_l2(l1_conv_val,l2_conv_val))(x)
    path3=BatchNormalization()(path3);path3=MaxPooling1D(2)(path3);path3=Dropout(dropout_rate_val/2)(path3)
    path3=Conv1D(128,15,padding='same',activation='relu',kernel_regularizer=l1_l2(l1_conv_val,l2_conv_val))(path3)
    path3=BatchNormalization()(path3);path3=MaxPooling1D(2)(path3);path3=Dropout(dropout_rate_val/2)(path3)
    global1=GlobalAveragePooling1D()(path1);global2=GlobalAveragePooling1D()(path2);global3=GlobalAveragePooling1D()(path3)
    merged_global=Concatenate()([global1,global2,global3])
    merged_shape=merged_global.get_shape().as_list();feature_dim=merged_shape[1]//3
    reshaped=Reshape((3,feature_dim))(merged_global)
    gru_units=64
    bidirectional=Bidirectional(GRU(gru_units,return_sequences=True,recurrent_dropout=0.2,implementation=1,recurrent_regularizer=l1_l2(l1_dense_val,l2_dense_val)))(reshaped)
    pooled=AveragePooling1D(pool_size=3)(bidirectional)
    flattened_dimension=gru_units*2;flattened=Reshape((flattened_dimension,))(pooled)
    dense=Dense(128,activation='relu',kernel_regularizer=l1_l2(l1_dense_val,l2_dense_val))(flattened)
    dense=BatchNormalization()(dense);dense=Dropout(dropout_rate_val)(dense)
    dense=Dense(64,activation='relu',kernel_regularizer=l1_l2(l1_dense_val,l2_dense_val))(dense)
    dense=BatchNormalization()(dense);dense=Dropout(dropout_rate_val/2)(dense)
    outputs=Dense(1 if num_classes==2 else num_classes,activation='sigmoid' if num_classes==2 else 'softmax')(dense)
    model=Model(inputs=inputs,outputs=outputs)
    model.compile(optimizer=Adam(lr=current_run_args.lr),loss='binary_crossentropy' if num_classes==2 else 'sparse_categorical_crossentropy',metrics=['accuracy'])
    return model

def train_model_instance(model, train_data, val_data, y_train_labels, y_val_labels, current_args_obj, context_output_dir, checkpoint_name_suffix='model', smote_on_train=True):
    global logger
    os.makedirs(context_output_dir, exist_ok=True)
    epochs_to_run=current_args_obj.epochs; patience_to_use=current_args_obj.patience
    if hasattr(current_args_obj,'is_hp_tuning_trial') and current_args_obj.is_hp_tuning_trial:
        epochs_to_run=current_args_obj.tune_epochs; patience_to_use=current_args_obj.tune_patience
    checkpoint_path=os.path.join(context_output_dir,f'best_{checkpoint_name_suffix}.h5')
    callbacks=[EarlyStopping(monitor='val_loss',patience=patience_to_use,restore_best_weights=True,verbose=1),
               ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=patience_to_use//2,min_lr=1e-7,verbose=1),
               ModelCheckpoint(checkpoint_path,monitor='val_loss',save_best_only=True,verbose=0),
               TensorBoard(log_dir=os.path.join(context_output_dir,f"tb_logs_{checkpoint_name_suffix.replace('.h5','')}"),histogram_freq=1)]
    X_train_current,y_train_current=train_data,y_train_labels; class_weight_dict_current=None
    if current_args_obj.binary:
        unique_train_labels,train_counts=np.unique(y_train_current,return_counts=True)
        if len(unique_train_labels)>1:
            class_weights_computed=class_weight.compute_class_weight(class_weight='balanced',classes=unique_train_labels,y=y_train_current)
            class_weight_dict_current=dict(zip(unique_train_labels,class_weights_computed))
            logger.info(f"Class weights: {class_weight_dict_current}")
        else: 
            logger.warning(f"Only one class in training. No class weights.")
        if SMOTE_AVAILABLE and smote_on_train and len(unique_train_labels)>1:
            min_samples_in_class = train_counts.min() if len(train_counts)>0 else 0
            k_neighbors_val = 5 if min_samples_in_class>5 else (min_samples_in_class-1 if min_samples_in_class>1 else 0)
            if k_neighbors_val>0:
                try:
                    n_s,n_c,n_t=X_train_current.shape;X_reshaped=X_train_current.reshape(n_s,n_c*n_t)
                    logger.info(f"SMOTE (k={k_neighbors_val}) on train (samples: {n_s})...")
                    sm=SMOTE(random_state=current_args_obj.seed,k_neighbors=k_neighbors_val)
                    X_res,y_res=sm.fit_resample(X_reshaped,y_train_current)
                    X_train_current,y_train_current=X_res.reshape(-1,n_c,n_t),y_res
                    logger.info(f"After SMOTE - Train: Pos: {np.sum(y_train_current==1)}, Neg: {np.sum(y_train_current==0)}")
                except Exception as e: logger.warning(f"SMOTE failed: {e}. No SMOTE.")
            else: logger.warning(f"SMOTE not applied: minority class has {min_samples_in_class} samples.")
    history=model.fit(X_train_current,y_train_current,
                      validation_data=(val_data,y_val_labels) if val_data is not None and y_val_labels is not None else None,
                      batch_size=current_args_obj.batch_size,epochs=epochs_to_run,callbacks=callbacks,
                      class_weight=class_weight_dict_current if current_args_obj.binary and class_weight_dict_current else None,verbose=1)
    logger.info("Training finished. EarlyStopping(restore_best_weights=True) should have restored best model state.")
    return history

def evaluate_model_instance(model, X_test_data, y_test_labels, current_run_args, num_model_outputs_eval):
    global logger
    if X_test_data is None or y_test_labels is None:
        logger.warning("Test data/labels not available for evaluation."); return {'metrics': None,'y_test':[],'y_pred':[],'y_pred_prob':[]}
    logger.info(f"Evaluating model on test set of size {X_test_data.shape[0]}...")
    test_loss,test_acc=model.evaluate(X_test_data,y_test_labels,verbose=0,batch_size=current_run_args.batch_size)
    logger.info(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")
    y_pred_prob=model.predict(X_test_data,batch_size=current_run_args.batch_size)
    y_pred=(y_pred_prob>=current_run_args.threshold).astype(int).flatten() if current_run_args.binary else np.argmax(y_pred_prob,axis=1)
    y_pred_prob_flat=y_pred_prob.flatten() if current_run_args.binary else y_pred_prob
    metrics={'test_loss':test_loss,'test_accuracy':test_acc,'accuracy':accuracy_score(y_test_labels,y_pred),
             'precision_weighted':precision_score(y_test_labels,y_pred,average='weighted',zero_division=0),
             'recall_weighted':recall_score(y_test_labels,y_pred,average='weighted',zero_division=0),
             'f1_weighted':f1_score(y_test_labels,y_pred,average='weighted',zero_division=0),
             'confusion_matrix':confusion_matrix(y_test_labels,y_pred,labels=range(num_model_outputs_eval)).tolist()}
    if current_run_args.binary:
        metrics['precision_class1']=precision_score(y_test_labels,y_pred,pos_label=1,zero_division=0)
        metrics['recall_class1']=recall_score(y_test_labels,y_pred,pos_label=1,zero_division=0)
        metrics['f1_class1']=f1_score(y_test_labels,y_pred,pos_label=1,zero_division=0)
        if len(np.unique(y_test_labels))>1:
            fpr,tpr,_=roc_curve(y_test_labels,y_pred_prob_flat)
            metrics['roc_auc']=auc(fpr,tpr);metrics['fpr']=fpr.tolist();metrics['tpr']=tpr.tolist()
            precision_pr,recall_pr,_=precision_recall_curve(y_test_labels,y_pred_prob_flat)
            metrics['pr_auc']=average_precision_score(y_test_labels,y_pred_prob_flat)
            metrics['precision_pr_curve']=precision_pr.tolist();metrics['recall_pr_curve']=recall_pr.tolist()
        else:
            metrics.update({'roc_auc':np.nan,'fpr':[],'tpr':[],'pr_auc':np.nan,'precision_pr_curve':[],'recall_pr_curve':[]})
            logger.warning("Only one class in y_test_labels. ROC/PR AUC not computed.")
    return {'metrics':metrics,'y_test':y_test_labels.tolist(),'y_pred':y_pred.tolist(),'y_pred_prob':y_pred_prob_flat.tolist()}


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj,np.integer):return int(obj)
        if isinstance(obj,np.floating):return float(obj)
        if isinstance(obj,np.ndarray):return obj.tolist()
        if hasattr(obj,'value') and type(obj).__name__=='Dimension':return obj.value
        return super(NpEncoder,self).default(obj)

def save_run_artefacts(results_dict, save_path_base, id_string):
    global logger, args 
    os.makedirs(save_path_base, exist_ok=True)
    if results_dict.get('metrics'):
        metrics_json_path = os.path.join(save_path_base, f'{id_string}_metrics_summary.json')
        args_serializable = {k: str(v) if not isinstance(v, (list, dict, int, float, bool, type(None))) else v for k, v in vars(args).items()}
        full_metrics_log = {'args_config': args_serializable, 'eval_metrics': results_dict['metrics']}
        with open(metrics_json_path, 'w') as f: json.dump(full_metrics_log, f, indent=4, cls=NpEncoder)
    
    metrics_text_path = os.path.join(save_path_base, f'{id_string}_metrics_summary.txt')
    with open(metrics_text_path, 'w') as f:
        f.write(f"Run ID: {id_string}\nArguments: {json.dumps(args_serializable, indent=2)}\n\n")
        if results_dict.get('metrics'):
            f.write("--- Evaluation Metrics ---\n"); m = results_dict['metrics']
            for key, value in m.items():
                if key not in ['fpr','tpr','confusion_matrix','precision_pr_curve','recall_pr_curve']:
                    if isinstance(value, float):
                        f.write(f"{key.replace('_',' ').capitalize()}: {value:.4f}\n")
                    else:
                        f.write(f"{key.replace('_',' ').capitalize()}: {value}\n")
            f.write("\nConfusion Matrix:\n" + str(np.array(m.get('confusion_matrix','N/A'))))
        else: f.write("No evaluation metrics available.\n")

    if 'history' in results_dict and results_dict['history']:
        pd.DataFrame(results_dict['history']).to_csv(os.path.join(save_path_base, f'{id_string}_training_history.csv'),index_label='epoch')
    for key in ['y_test','y_pred','y_pred_prob']:
        if key in results_dict: np.save(os.path.join(save_path_base,f'{id_string}_{key}.npy'),np.array(results_dict[key]))
    logger.info(f"Run artefacts for '{id_string}' saved to {save_path_base}")
    if 'history' in results_dict and results_dict['history'] and results_dict.get('metrics'):
        plot_results(results_dict,id_string,args.binary,args.focus_class,save_path=save_path_base)
        plot_detailed_overfitting_analysis(results_dict,id_string,save_path=save_path_base)

def plot_results(results, model_id_string, binary_flag, focus_class_name_str, save_path):
    history,metrics=results['history'],results['metrics']
    if not history: logger.warning(f"History empty for {model_id_string}, skipping history plots."); return
    plt.figure(figsize=(12,5)); acc_key='acc' if 'acc' in history else 'accuracy'; val_acc_key='val_acc' if 'val_acc' in history else 'val_accuracy'
    if not(acc_key in history and val_acc_key in history and history[acc_key] and history[val_acc_key]): logger.warning(f"Acc keys/data missing for {model_id_string}. Skipping history plot.")
    else:
        plt.subplot(1,2,1);plt.plot(history[acc_key],label='Train Acc');plt.plot(history[val_acc_key],label='Val Acc')
        plt.title('Model Accuracy');plt.ylabel('Accuracy');plt.xlabel('Epoch');plt.legend()
        plt.subplot(1,2,2);plt.plot(history['loss'],label='Train Loss');plt.plot(history['val_loss'],label='Val Loss')
        plt.title('Model Loss');plt.ylabel('Loss');plt.xlabel('Epoch');plt.legend()
        plt.tight_layout();plt.savefig(os.path.join(save_path,f'{model_id_string}_training_history.png'));plt.close()
    cm=np.array(metrics['confusion_matrix'])
    class_names=[f'Non-{focus_class_name_str.capitalize()}',focus_class_name_str.capitalize()] if binary_flag else [f'Class {i}' for i in range(cm.shape[0])]
    plt.figure(figsize=(10,4));plt.subplot(1,2,1);sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=class_names,yticklabels=class_names)
    plt.title(f'CM - {model_id_string}');plt.ylabel('True Label');plt.xlabel('Predicted Label')
    cm_sum_axis1=cm.sum(axis=1)[:,np.newaxis];cm_norm=cm.astype('float')/np.where(cm_sum_axis1==0,1,cm_sum_axis1)
    plt.subplot(1,2,2);sns.heatmap(cm_norm,annot=True,fmt='.2f',cmap='Blues',xticklabels=class_names,yticklabels=class_names)
    plt.title(f'Normalized CM - {model_id_string}');plt.ylabel('True Label');plt.xlabel('Predicted Label')
    plt.tight_layout();plt.savefig(os.path.join(save_path,f'{model_id_string}_confusion_matrix.png'));plt.close()
    if binary_flag and 'roc_auc' in metrics and metrics['roc_auc'] is not np.nan:
        plt.figure(figsize=(12,5));plt.subplot(1,2,1)
        fpr,tpr=np.array(metrics.get('fpr',[])),np.array(metrics.get('tpr',[]))
        if fpr.size>0 and tpr.size>0:
            plt.plot(fpr,tpr,label=f'ROC (AUC = {metrics["roc_auc"]:.2f})');plt.plot([0,1],[0,1],'k--')
            plt.xlim([0.0,1.0]);plt.ylim([0.0,1.05]);plt.xlabel('FPR');plt.ylabel('TPR');plt.title(f'ROC Curve - {model_id_string}');plt.legend()
        plt.subplot(1,2,2)
        pr_p,pr_r=np.array(metrics.get('precision_pr_curve',[])),np.array(metrics.get('recall_pr_curve',[]))
        pr_auc_val=metrics.get('pr_auc',np.nan)
        if pr_p.size>0 and pr_r.size>0:
            plt.plot(pr_r,pr_p,label=f'PR curve (AP = {pr_auc_val:.2f})')
            no_skill = np.sum(np.array(results['y_test'])==1)/len(results['y_test']) if len(results['y_test'])>0 else 0
            plt.plot([0,1],[no_skill,no_skill],linestyle='--',label=f'No Skill (AP={no_skill:.2f})' if no_skill>0 else 'No Skill')
            plt.xlabel('Recall');plt.ylabel('Precision');plt.title(f'PR Curve - {model_id_string}');plt.legend()
        plt.tight_layout();plt.savefig(os.path.join(save_path,f'{model_id_string}_roc_pr_curves.png'));plt.close()

def plot_detailed_overfitting_analysis(results, model_id_string, save_path):
    history = results['history']
    if not history: logger.warning(f"History empty for {model_id_string}, skipping overfitting plots."); return
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    acc_key='acc' if 'acc' in history else 'accuracy'; val_acc_key='val_acc' if 'val_acc' in history else 'val_accuracy'
    if not(acc_key in history and val_acc_key in history and history[acc_key] and history[val_acc_key]):
        logger.warning(f"Acc keys/data missing for {model_id_string}. Skipping overfitting plot.");plt.close(fig); return
    axes[0,0].plot(history[acc_key],label='Train Acc');axes[0,0].plot(history[val_acc_key],label='Val Acc')
    axes[0,0].set_title('Model Accuracy');axes[0,0].set_ylabel('Accuracy');axes[0,0].set_xlabel('Epoch');axes[0,0].legend()
    axes[0,1].plot(history['loss'],label='Train Loss');axes[0,1].plot(history['val_loss'],label='Val Loss')
    axes[0,1].set_title('Model Loss');axes[0,1].set_ylabel('Loss');axes[0,1].set_xlabel('Epoch');axes[0,1].legend()
    gen_gap_acc=[train-val for train,val in zip(history[acc_key],history[val_acc_key])]
    axes[1,0].plot(gen_gap_acc,label='Acc Gap (Train-Val)');axes[1,0].set_title('Generalization Gap (Acc)')
    axes[1,0].set_ylabel('Train Acc - Val Acc');axes[1,0].set_xlabel('Epoch');axes[1,0].axhline(0,c='r',ls='--',alpha=0.5);axes[1,0].legend()
    gen_gap_loss=[val-train for train,val in zip(history['loss'],history['val_loss'])]
    axes[1,1].plot(gen_gap_loss,label='Loss Gap (Val-Train)');axes[1,1].set_title('Generalization Gap (Loss)')
    axes[1,1].set_ylabel('Val Loss - Train Loss');axes[1,1].set_xlabel('Epoch');axes[1,1].axhline(0,c='r',ls='--',alpha=0.5);axes[1,1].legend()
    plt.tight_layout();plt.savefig(os.path.join(save_path,f'{model_id_string}_overfitting_analysis.png'));plt.close()

def hyperparameter_search(dataset_for_hp, input_shape_hp, num_model_outputs_hp, original_args_obj):
    global logger, output_dir 
    logger.info("--- Starting Hyperparameter Tuning ---")
    param_grid = { 
        'lr': [5e-4, 1e-4], 'dropout_rate': [0.3, 0.5],
        'l2_conv': [1e-4, 1e-5], 'l2_dense': [1e-4, 1e-5],
        'l1_conv': [original_args_obj.l1_conv], 'l1_dense': [original_args_obj.l1_dense],
    } 
    data_key_hp = 'fused' if original_args_obj.model_type == 'fused' else 'eeg'
    if not (dataset_for_hp.get('train') and data_key_hp in dataset_for_hp['train'] and \
            dataset_for_hp.get('val') and data_key_hp in dataset_for_hp['val']):
        logger.error("HP Tuning requires valid 'train' and 'val' data. Aborting HP tuning.")
        return vars(original_args_obj)
    X_hp_train = dataset_for_hp['train'][data_key_hp]
    y_hp_train = dataset_for_hp['train']['y_binary_processed' if original_args_obj.binary else 'y_processed']
    X_hp_val = dataset_for_hp['val'][data_key_hp]
    y_hp_val = dataset_for_hp['val']['y_binary_processed' if original_args_obj.binary else 'y_processed']

    best_val_metric_value,best_params_dict,tuning_results_log = -float('inf'),{},[]
    from itertools import product
    param_names,param_value_lists=list(param_grid.keys()),list(param_grid.values())
    total_combinations=np.prod([len(v) for v in param_value_lists])
    logger.info(f"Total HP combinations to test: {total_combinations}")
    hp_tuning_trials_base_dir=os.path.join(output_dir,"hp_tuning_trials");os.makedirs(hp_tuning_trials_base_dir,exist_ok=True)

    for i, combo_values in enumerate(product(*param_value_lists)):
        trial_num = i + 1; tf.keras.backend.clear_session(); gc.collect(); set_seeds(original_args_obj.seed + trial_num) 
        current_trial_params_combo = dict(zip(param_names, combo_values))
        logger.info(f"HP Tuning Trial {trial_num}/{total_combinations}: {current_trial_params_combo}")
        trial_args_namespace = argparse.Namespace(**vars(original_args_obj))
        for key, value in current_trial_params_combo.items(): setattr(trial_args_namespace, key, value)
        setattr(trial_args_namespace, 'is_hp_tuning_trial', True) 
        model_hp = create_multiscale_residual_model(input_shape_hp, num_model_outputs_hp, trial_args_namespace)
        trial_instance_output_dir = os.path.join(hp_tuning_trials_base_dir, f"trial_{trial_num}")
        history_hp = train_model_instance(model_hp, X_hp_train, X_hp_val, y_hp_train, y_hp_val, trial_args_namespace, trial_instance_output_dir, checkpoint_name_suffix=f'hp_trial_{trial_num}', smote_on_train=False)
        val_metric_to_optimize='val_accuracy'; metric_key='val_acc' if 'val_acc' in history_hp.history else 'val_accuracy'
        current_trial_metric_value=max(history_hp.history[metric_key]) if history_hp.history and history_hp.history.get(metric_key) else 0.0
        trial_summary={**current_trial_params_combo,val_metric_to_optimize:current_trial_metric_value};tuning_results_log.append(trial_summary)
        logger.info(f"Trial {trial_num} - {val_metric_to_optimize}: {current_trial_metric_value:.4f}")
        if current_trial_metric_value > best_val_metric_value:
            best_val_metric_value=current_trial_metric_value; best_params_dict=current_trial_params_combo
            logger.info(f"New best {val_metric_to_optimize}: {best_val_metric_value:.4f} with params: {best_params_dict}")
            if os.path.exists(os.path.join(trial_instance_output_dir,f'best_hp_trial_{trial_num}.h5')):
                 model_hp.save(os.path.join(hp_tuning_trials_base_dir, "best_overall_hp_trial_model.h5"))
        del model_hp, history_hp; gc.collect()
    logger.info(f"HP search complete. Best {val_metric_to_optimize}: {best_val_metric_value:.4f}\nBest params: {best_params_dict}")
    tuning_df=pd.DataFrame(tuning_results_log);tuning_df.sort_values(by=val_metric_to_optimize,ascending=False,inplace=True)
    tuning_df.to_csv(os.path.join(output_dir,"hp_tuning_summary.csv"),index=False)
    final_tuned_args_dict=vars(original_args_obj).copy()
    if best_params_dict: final_tuned_args_dict.update(best_params_dict)
    return final_tuned_args_dict

def perform_kfold_cross_validation(cv_data_dict, input_shape_cv, num_model_outputs_cv, current_run_args_obj):
    global logger, output_dir, dataset_global 
    main_run_output_dir = output_dir 
    logger.info(f"--- Starting {current_run_args_obj.kfold}-Fold CV (Base Output: {main_run_output_dir}) ---")
    data_key_cv='fused' if current_run_args_obj.model_type=='fused' else 'eeg'
    X_cv_combined, y_cv_combined = cv_data_dict[data_key_cv], cv_data_dict['y_binary_processed' if current_run_args_obj.binary else 'y_processed']
    skf = StratifiedKFold(n_splits=current_run_args_obj.kfold,shuffle=True,random_state=current_run_args_obj.seed)
    all_fold_results_list = []
    for fold_idx, (train_indices, val_indices) in enumerate(skf.split(X_cv_combined, y_cv_combined)):
        current_fold_num = fold_idx + 1
        fold_specific_output_dir = os.path.join(main_run_output_dir, f"fold_{current_fold_num}") 
        os.makedirs(fold_specific_output_dir, exist_ok=True)
        logger.info(f"===== Processing Fold {current_fold_num}/{current_run_args_obj.kfold} (Output: {fold_specific_output_dir}) =====")
        tf.keras.backend.clear_session(); gc.collect(); set_seeds(current_run_args_obj.seed + current_fold_num) 
        X_train_fold,X_val_fold=X_cv_combined[train_indices],X_cv_combined[val_indices]
        y_train_fold,y_val_fold=y_cv_combined[train_indices],y_cv_combined[val_indices]
        fold_args_obj = argparse.Namespace(**vars(current_run_args_obj)); setattr(fold_args_obj,'is_hp_tuning_trial',False)
        model_cv_fold = create_multiscale_residual_model(input_shape_cv, num_model_outputs_cv, fold_args_obj)
        if fold_idx == 0:
            with open(os.path.join(fold_specific_output_dir, "model_summary_fold_1.txt"), "w") as f_sum: model_cv_fold.summary(print_fn=lambda x:f_sum.write(x+'\n'))
        history_cv_fold = train_model_instance(model_cv_fold,X_train_fold,X_val_fold,y_train_fold,y_val_fold,fold_args_obj,fold_specific_output_dir,checkpoint_name_suffix=f'fold_{current_fold_num}_model')
        test_data,test_labels=None,None
        if dataset_global and dataset_global.get('test'):
            test_split_data = dataset_global['test']
            if test_split_data: 
                test_data=test_split_data.get(data_key_cv)
                test_labels=test_split_data.get('y_binary_processed' if args.binary else 'y_processed')
        eval_results_cv_fold=evaluate_model_instance(model_cv_fold,test_data,test_labels,fold_args_obj,num_model_outputs_cv)
        current_fold_full_results={'history':history_cv_fold.history,**eval_results_cv_fold};all_fold_results_list.append(current_fold_full_results)
        save_run_artefacts(current_fold_full_results,fold_specific_output_dir,f"fold_{current_fold_num}_eval_on_test")
        del model_cv_fold,history_cv_fold,eval_results_cv_fold;gc.collect()
    output_dir = main_run_output_dir 
    logger.info(f"===== K-Fold CV Summary (Results in: {output_dir}) =====")
    valid_metrics=[res['metrics'] for res in all_fold_results_list if res.get('metrics')]
    if valid_metrics:
        accs=[m['test_accuracy'] for m in valid_metrics if m and 'test_accuracy' in m]
        f1s_key='f1_class1' if current_run_args_obj.binary else 'f1_weighted'
        f1s=[m.get(f1s_key,np.nan) for m in valid_metrics if m]
        cv_summary={'mean_test_accuracy':np.nanmean(accs) if accs else np.nan,'std_test_accuracy':np.nanstd(accs) if accs else np.nan,
                    'mean_f1_score':np.nanmean(f1s) if f1s else np.nan,'std_f1_score':np.nanstd(f1s) if f1s else np.nan,
                    'all_fold_metrics':valid_metrics}
        with open(os.path.join(output_dir,"kfold_cv_overall_summary.json"),'w') as f:json.dump(cv_summary,f,indent=4,cls=NpEncoder)
        logger.info(f"Overall K-Fold CV summary saved.")
    else: logger.warning("No valid metrics from K-Fold CV folds to summarize.")
    return all_fold_results_list

dataset_global = None 

def main():
    global args, output_dir, logger, dataset_global 
    
    setup_logging_and_output_dir(args)
    set_seeds(args.seed) 
    
    try:
        from tensorflow.python.client import device_lib
        local_devices = device_lib.list_local_devices()
        gpus_detected = [x.name for x in local_devices if x.device_type == 'GPU']
        if gpus_detected:
            logger.info(f"TensorFlow (device_lib) detected GPUs: {gpus_detected}")
            if not args.gpu and len(gpus_detected) > 1:
                 logger.warning(f"Multiple GPUs ({len(gpus_detected)}) detected, but --gpu not specified. TF might use all or first by default.")
            elif args.gpu and args.gpu not in "".join(gpus_detected): 
                 logger.warning(f"User specified GPU '{args.gpu}' but it's not in the detected list: {gpus_detected}. Check TF/driver setup or CUDA_VISIBLE_DEVICES.")

        else:
            logger.warning("TensorFlow (device_lib) did not detect any GPUs. Running on CPU.")
            if args.gpu:
                 logger.warning(f"User specified GPU '{args.gpu}' but no GPUs were detected by TensorFlow.")
    except Exception as e_dev_list:
        logger.warning(f"Could not list TF devices via device_lib: {e_dev_list}")

    model_type = args.model_type
    category_to_process = args.category
    logger.info(f"Starting run. Model: {model_type}, Category: {category_to_process}, Binary: {args.binary}")
    
    try:
        final_processed_dataset = {}; model_id_string_main = ""
        data_key_model_input = 'fused' if model_type == 'fused' else 'eeg'
        y_ref_key = 'y_binary_processed' if args.binary else 'y_processed'

        if category_to_process == 'all':
            categories_list = ['left_bte_crosstop', 'right_bte_crosstop', 'both_bte_no_crosstop']
            for split in ['train', 'val', 'test']: final_processed_dataset[split] = {data_key_model_input: [], 'y_original_unprocessed': []} # Changed key
            any_data_loaded = False
            for cat_name in categories_list:
                try:
                    cat_data_loaded = load_dataset(args.data_path, cat_name, args.model_type)
                    for split in ['train', 'val', 'test']:
                        if cat_data_loaded.get(split) and cat_data_loaded[split].get(data_key_model_input):
                            final_processed_dataset[split][data_key_model_input].append(cat_data_loaded[split][data_key_model_input])
                            final_processed_dataset[split]['y_original_unprocessed'].append(cat_data_loaded[split]['y'])
                            any_data_loaded = True
                except FileNotFoundError: logger.warning(f"Data for cat {cat_name} (all mode) not found, skipping.")
            if not any_data_loaded: raise ValueError("No data loaded for any category in 'all' mode.")
            for split in ['train', 'val', 'test']:
                if final_processed_dataset[split][data_key_model_input]:
                    current_data_unprocessed = {
                        data_key_model_input: np.concatenate(final_processed_dataset[split][data_key_model_input], axis=0),
                        'y': np.concatenate(final_processed_dataset[split]['y_original_unprocessed'], axis=0)
                    }
                    final_processed_dataset[split] = preprocess_labels_for_split(current_data_unprocessed,args.binary,args.focus_class)
                else: final_processed_dataset[split] = None
            model_id_string_main = f"{model_type}_{get_simplified_category_name('all')}"
        else: 
            single_cat_loaded = load_dataset(args.data_path, category_to_process, model_type)
            for split_name in ['train', 'val', 'test']:
                final_processed_dataset[split_name] = preprocess_labels_for_split(single_cat_loaded.get(split_name),args.binary,args.focus_class)
            model_id_string_main = f"{model_type}_{get_simplified_category_name(category_to_process)}"
        
        dataset_global = final_processed_dataset

        if not (final_processed_dataset.get('train') and data_key_model_input in final_processed_dataset['train']):
            logger.error("No training data. Exiting."); return
        
        input_shape = final_processed_dataset['train'][data_key_model_input].shape[1:]
        y_train_labels_for_num_classes = final_processed_dataset['train'][y_ref_key]
        num_model_outputs = 2 if args.binary else len(np.unique(y_train_labels_for_num_classes))
        logger.info(f"Determined Model Input Shape: {input_shape}, Num Outputs: {num_model_outputs}")
        current_run_params_dict = vars(args).copy()

        if args.tune_hyperparams:
            logger.info("--- Hyperparameter Tuning Phase ---")
            hp_val_data_dict = final_processed_dataset.get('val')
            if not (hp_val_data_dict and data_key_model_input in hp_val_data_dict and y_ref_key in hp_val_data_dict):
                logger.warning("Val data for HP tuning insufficient/missing. Splitting 20% from train set (not ideal).")
                y_temp_train_hp = final_processed_dataset['train'][y_ref_key]
                stratify_hp = y_temp_train_hp if len(np.unique(y_temp_train_hp)) > 1 and args.binary else None # Stratify only if binary and multiple classes
                X_hp_t,X_hp_v,y_hp_t,y_hp_v = train_test_split(final_processed_dataset['train'][data_key_model_input],y_temp_train_hp,test_size=0.2,random_state=args.seed,stratify=stratify_hp)
                hp_data_source={'train':{data_key_model_input:X_hp_t,y_ref_key:y_hp_t},'val':{data_key_model_input:X_hp_v,y_ref_key:y_hp_v}}
            else: hp_data_source = {'train':final_processed_dataset['train'],'val':hp_val_data_dict}
            best_params=hyperparameter_search(hp_data_source,input_shape,num_model_outputs,args)
            current_run_params_dict.update(best_params); logger.info(f"Using tuned HPs: {best_params}")
        
        current_run_params_obj = argparse.Namespace(**current_run_params_dict)
        setattr(current_run_params_obj,'is_hp_tuning_trial',False) 

        if args.kfold > 1:
            logger.info(f"--- K-Fold CV Phase ({args.kfold} folds) ---")
            if not (final_processed_dataset.get('train') and final_processed_dataset.get('val')):
                 logger.error("Cannot perform K-Fold CV: Train or Val data missing."); return
            X_cv_all=np.concatenate((final_processed_dataset['train'][data_key_model_input],final_processed_dataset['val'][data_key_model_input]),axis=0)
            y_cv_all=np.concatenate((final_processed_dataset['train'][y_ref_key],final_processed_dataset['val'][y_ref_key]),axis=0)
            cv_prep_dataset={data_key_model_input:X_cv_all,y_ref_key:y_cv_all}
            perform_kfold_cross_validation(cv_prep_dataset,input_shape,num_model_outputs,current_run_params_obj)
            logger.info("--- Training Final Model on All Train+Val Data (Post-CV) ---")
            final_model=create_multiscale_residual_model(input_shape,num_model_outputs,current_run_params_obj)
            test_data_final,y_test_final=None,None
            if final_processed_dataset.get('test'):
                test_data_final=final_processed_dataset['test'].get(data_key_model_input)
                y_test_final=final_processed_dataset['test'].get(y_ref_key)
            history_final=train_model_instance(final_model,X_cv_all,None,y_cv_all,None,current_run_params_obj,output_dir,checkpoint_name_suffix=f'{model_id_string_main}_final_post_cv_no_val_in_fit')
            results_final_eval=evaluate_model_instance(final_model,test_data_final,y_test_final,current_run_params_obj,num_model_outputs)
            results_final_run={'history':history_final.history,**results_final_eval}
            save_run_artefacts(results_final_run,output_dir,f"{model_id_string_main}_final_post_cv")
            final_model.save(os.path.join(output_dir,f"{model_id_string_main}_final_model_post_cv.h5"))
        else: 
            logger.info("--- Standard Train/Val/Test Phase (No K-Fold CV) ---")
            model_single=create_multiscale_residual_model(input_shape,num_model_outputs,current_run_params_obj)
            with open(os.path.join(output_dir,f"{model_id_string_main}_model_summary.txt"),"w") as f_s:model_single.summary(print_fn=lambda x:f_s.write(x+'\n'))
            train_d,val_d,test_d=final_processed_dataset['train'],final_processed_dataset.get('val'),final_processed_dataset.get('test')
            history_single=train_model_instance(model_single,train_d[data_key_model_input],val_d.get(data_key_model_input) if val_d else None,train_d[y_ref_key],val_d.get(y_ref_key) if val_d else None,current_run_params_obj,output_dir,checkpoint_name_suffix=f'{model_id_string_main}_single_run')
            results_single_eval=evaluate_model_instance(model_single,test_d.get(data_key_model_input) if test_d else None,test_d.get(y_ref_key) if test_d else None,current_run_params_obj,num_model_outputs)
            results_single_run_full={'history':history_single.history,**results_single_eval}
            save_run_artefacts(results_single_run_full,output_dir,model_id_string_main)
            model_single.save(os.path.join(output_dir,f"{model_id_string_main}_final_model.h5"))
        logger.info(f"Script finished. Results in: {output_dir}")
    except FileNotFoundError as fnfe: logger.error(f"Data file/dir not found: {fnfe}",exc_info=False);sys.exit(1)
    except ValueError as ve: logger.error(f"ValueError: {ve}",exc_info=True);sys.exit(1)
    except Exception as e: logger.error(f"Unexpected error in main: {e}",exc_info=True);sys.exit(1)

if __name__ == "__main__":

    temp_parser_for_gpu = get_arg_parser()
    temp_pre_args = temp_parser_for_gpu.parse_args()

    if temp_pre_args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = temp_pre_args.gpu
        print(f"Attempting to set CUDA_VISIBLE_DEVICES to: '{temp_pre_args.gpu}' (early setup)")
    else:

        print(f"CUDA_VISIBLE_DEVICES not specified by --gpu arg. TensorFlow will use its default visibility.")

    try:
        if tf.__version__.startswith("1."):
            print(f"TensorFlow version {tf.__version__} detected. Attempting TF1.x GPU memory growth configuration.")
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.compat.v1.Session(config=config)
            tf.compat.v1.keras.backend.set_session(sess)
            print("TensorFlow 1.x Keras session configured for GPU memory growth.")
    except Exception as e_tf_config:
        print(f"Warning: Could not configure TF1.x session for GPU memory growth: {e_tf_config}")
    
    set_seeds(temp_pre_args.seed)
    
    args = temp_pre_args
    
    main()
