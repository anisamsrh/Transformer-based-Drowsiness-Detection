import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pytorch_lightning import Callback
from torchmetrics import MeanAbsoluteError, MeanAbsolutePercentageError

from darts import TimeSeries

def get_kss_score(nf) : 
    kss = nf.split("_")
    return float(kss[3])

def apply_fixed_scaling(series, absolute_min, absolute_max):
    scaled_series = (series - absolute_min) / (absolute_max - absolute_min)
    return np.clip(scaled_series, 0.0, 1.0)

def reverse_fixed_scaling(series, absolute_min, absolute_max):
    unscaled_series = series * (absolute_max - absolute_min) + absolute_min
    return unscaled_series

def load_data_as_ts_list(ft, file):
    folders = glob.glob(f"data_{ft}/*")
    target_ts_list = []
    past_cov_ts_list = []

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

        target_ts = TimeSeries.from_dataframe(df, time_col="log_time", value_cols=["kss_score"])
        past_cov_ts = TimeSeries.from_dataframe(df, time_col="log_time", value_cols=["breath_rate", "heart_rate"])
        target_ts_list.append(target_ts)
        past_cov_ts_list.append(past_cov_ts)
    return target_ts_list, past_cov_ts_list

def visualize_metrics(file, timestamp="00"):
    metrics_df = pd.read_csv(f"{file}")
    metrics_df = metrics_df.groupby("epoch").agg({
        "train_loss": "mean",
        "val_loss": "first"
    }).reset_index()

    plt.figure(figsize=(10, 6))
    plt.plot(metrics_df['epoch'], metrics_df['train_loss'], 
            label='Train Loss', marker='o', linewidth=2, color='tab:blue')
    plt.plot(metrics_df['epoch'], metrics_df['val_loss'], 
            label='Validation Loss', marker='s', linewidth=2, color='tab:orange')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss Value', fontsize=12)
    plt.title('Training Results: Train Loss vs Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    import os
    os.makedirs("results_metrics/train", exist_ok=True)
    image_name, _ = os.path.splitext(os.path.basename(file))
    plt.savefig(f"results_metrics/train/{image_name}_{timestamp}.jpg", bbox_inches='tight', dpi=300)