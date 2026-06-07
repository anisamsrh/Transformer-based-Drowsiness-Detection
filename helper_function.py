import glob
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, ConcatDataset, Subset
from tsai.all import *
from sklearn.model_selection import StratifiedGroupKFold

from custom_class import TimeSeriesDataset
import config as CONFIG

def get_kss_score(nf) : 
    kss = nf.split("_")
    return float(kss[3])

def apply_fixed_scaling(series, absolute_min, absolute_max):
    scaled_series = (series - absolute_min) / (absolute_max - absolute_min)
    return np.clip(scaled_series, 0.0, 1.0)

def reverse_fixed_scaling(series, absolute_min, absolute_max):
    unscaled_series = series * (absolute_max - absolute_min) + absolute_min
    return unscaled_series

def load_data_as_df_list(ft, file) : 
    folders = glob.glob(f"data_{ft}/*")
    pd_list = []

    for f in folders :
        df = pd.read_csv(f"{f}/{file}")
        df["kss_score"] = get_kss_score(f)
        df["log_time"] = pd.to_datetime(df['log_time'])
        df = df.set_index("log_time")
        df = df.resample("1s").mean().interpolate(method="linear").dropna()
        df = df.reset_index()

        df["kss_score"] = apply_fixed_scaling(df["kss_score"], 1, 9)
        df["breath_rate"] = apply_fixed_scaling(df["breath_rate"], 5, 40)
        df["heart_rate"] = apply_fixed_scaling(df["heart_rate"], 40, 200)

        pd_list.append(df)
    return pd_list

def load_data_as_df(file):
    f = Path(file).parent.name
    df = pd.read_csv(f"{file}")
    df["kss_score"] = get_kss_score(f)
    df["log_time"] = pd.to_datetime(df['log_time'])
    df = df.set_index("log_time")
    df = df.resample("1s").mean().interpolate(method="linear").dropna()
    df = df.reset_index()

    df["kss_score"] = apply_fixed_scaling(df["kss_score"], 1, 9)
    df["breath_rate"] = apply_fixed_scaling(df["breath_rate"], 5, 40)
    df["heart_rate"] = apply_fixed_scaling(df["heart_rate"], 40, 200)
    return df

def load_data_as_df_list_scaled(ft, file) : 
    folders = glob.glob(f"data_{ft}/*")
    pd_list = []

    for f in folders :
        scaler = StandardScaler()
        df = pd.read_csv(f"{f}/{file}")
        df["log_time"] = pd.to_datetime(df['log_time'])
        df = df.set_index("log_time")
        df = df.resample("1s").mean().interpolate(method="linear").dropna()
        df = df.reset_index()

        df_scaled = pd.DataFrame(scaler.fit_transform(df[["heart_rate", "breath_rate"]]), columns=["heart_rate", "breath_rate"])

        df_scaled["kss_score"] = get_kss_score(f)
        df_scaled["kss_score"] = apply_fixed_scaling(df["kss_score"], 1, 9)

        pd_list.append(df_scaled)
    return pd_list

def load_model(weight_path, train_params):
    model = TSTPlus(
            c_in = 2,
            c_out = 1,
            seq_len = train_params.get("input_chunk_length", CONFIG.INPUT_CHUNK_LEN),
            n_layers = train_params.get("n_layers", CONFIG.N_LAYERS),
            fc_dropout = train_params.get("fc_dropout", CONFIG.DROPOUT),
            d_model = train_params.get("d_model", CONFIG.D_MODEL),
            n_heads = train_params.get("n_heads", CONFIG.ATT_HEADS),
        )
    
    state_dict = torch.load(weight_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    return model

def load_config(file_path, trial=None):
    with open(file_path, 'r') as f:
        config = json.load(f)
    if isinstance(config, (np.ndarray, list)):
        print(trial)
        assert trial is not None, "must Specify param_n"

        chosen_config = next((x["params"] for x in config if x["trial_number"]==trial), None)
        if chosen_config is not None:
            return chosen_config
    return config

def create_single_loader(df, icl, bs):
    dataset = TimeSeriesDataset(df, i_chunk_len=icl)
    return DataLoader(dataset, batch_size=bs, shuffle=False)

def create_loader(df_list, i_chunk_len, batch_size):
    loader = []
    for df in df_list :
        dataset = TimeSeriesDataset(df,
            i_chunk_len=i_chunk_len,
        )
        loader.append(DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False
        )
    )
    return loader

def create_big_loader(df_list, i_chunk_len=30, batch_size=64):
    datasets = []
    for df in df_list :
        dataset = TimeSeriesDataset(df,
            i_chunk_len=i_chunk_len,
        )
        datasets.append(dataset)
    full_dataset = ConcatDataset(datasets)
    return DataLoader(full_dataset, batch_size=batch_size, shuffle=True)

def create_kfold_loaders(df_list, k_splits=5, seq_length=30, batch_size=32):
    datasets = []
    all_labels = []
    all_groups = []
    
    for file_id, df in enumerate(df_list):
        dataset = TimeSeriesDataset(
            df,
            i_chunk_len=seq_length
        )
        if len(dataset) == 0:
            continue
            
        datasets.append(dataset)
        
        labels = dataset.get_all_labels()
        all_labels.extend(labels)
        all_groups.extend([file_id] * len(dataset))
        
    full_dataset = ConcatDataset(datasets)
    
    X_dummy = np.zeros(len(full_dataset)) # SGKF only need data length, not the real feature
    groups = np.array(all_groups)

    bins = [0, 0.3751, 0.751, 1.1] 
    y_stratify = np.digitize(all_labels, bins)
    y_reg = np.array(all_labels)

    # print(X_dummy)
    # print(y_stratify)
    # print(y_reg)
    # print(groups)

    sgkf = StratifiedGroupKFold(n_splits=k_splits)
    fold_loaders = []
    for train_idx, val_idx in sgkf.split(X_dummy, y_stratify, groups):
        train_subset = Subset(full_dataset, train_idx)
        val_subset = Subset(full_dataset, val_idx)
        
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        
        fold_loaders.append((train_loader, val_loader))
        
    return fold_loaders

################ VISUALIZATION ####################

def visualize_train(history, periperal, timestamp):
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(16, 12))

    plt.subplot(2, 2, 1)
    plt.plot(epochs, history['train_loss'], 
            label='Train Loss', marker='o', linewidth=2, color='tab:blue')
    plt.plot(epochs, history['val_loss'], 
            label='Validation Loss', marker='s', linewidth=2, color='tab:orange')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss Value', fontsize=12)
    plt.title('Training Results: Train Loss vs Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.subplot(2, 2, 2)
    plt.plot(epochs, history['train_kss_mae'], label='Train MAE', marker='o', linewidth=2, color='tab:blue')
    plt.plot(epochs, history['val_kss_mae'], label='Validation MAE', marker='s', linewidth=2, color='tab:orange')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('MAE Value', fontsize=12)
    plt.title('Train vs Validation MAE', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.subplot(2, 2, 3)
    plt.plot(epochs, history['train_kss_mse'], label='Train MSE', marker='o', linewidth=2, color='tab:blue')
    plt.plot(epochs, history['val_kss_mse'], label='Validation MSE', marker='s', linewidth=2, color='tab:orange')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('MSE Value', fontsize=12)
    plt.title('Train vs Validation MSE', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.subplot(2, 2, 4)
    plt.plot(epochs, history['train_kss_acc'], label='Train Accuracy', marker='o', linewidth=2, color='tab:blue')
    plt.plot(epochs, history['val_kss_acc'], label='Validation Accuracy', marker='s', linewidth=2, color='tab:orange')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Train vs Validation Accuracy', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    basepath = f"logs/{periperal}_{timestamp}"
    os.makedirs(basepath, exist_ok=True)
    plt.savefig(f"{basepath}/metrics.jpg", bbox_inches='tight', dpi=300)

