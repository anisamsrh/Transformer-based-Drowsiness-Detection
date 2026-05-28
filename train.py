import argparse
import pandas as pd
import time
import torch
import numpy as np
from darts import TimeSeries
from darts.models import TFTModel
from darts.dataprocessing.transformers import Scaler
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import glob
import os
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt

import config as CONFIG

def get_kss_score(nf) : 
    kss = nf.split("_")
    return float(kss[3])

def apply_fixed_scaling(series, absolute_min, absolute_max):
    scaled_series = (series - absolute_min) / (absolute_max - absolute_min)
    return np.clip(scaled_series, 0.0, 1.0)

def load_data_as_ts_list(ft, file):
    folders = glob.glob(f"data_{ft}/*")
    target_ts_list = []
    past_cov_ts_list = []

    for f in folders :
        df = pd.read_csv(f"{f}/{file}")
        df["kss_score"] = get_kss_score(f)
        df["log_time"] = pd.to_datetime(df['log_time'])
        df = df.set_index("log_time")
        df = df.resample("1s").mean().interpolate(method="linear")
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

def main(file):
    parser = argparse.ArgumentParser()
    parser.add_argument('--novis', action='store_false')
    args = parser.parse_args()

    file = f"{file}"
    train_target, train_past_cov = load_data_as_ts_list("train", file)
    val_target, val_past_cov = load_data_as_ts_list("val", file)

    print("-----Data Loaded-----")

    timestamp = time.time()
    logger = CSVLogger(save_dir="logs/", name=f"tft_run_{timestamp}")

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1, # only save 1 file
        dirpath="logs/checkpoints/",
        filename=f"tft_{timestamp}_{{epoch:02d}}_{{val_loss:.4f}}",
        save_weights_only=False 
    )

    model = TFTModel(
        input_chunk_length=CONFIG.INPUT_CHUNK_LEN, # change to seq_len * context_time
        output_chunk_length=CONFIG.OUTPUT_CHUNK_LEN, # change to seq_len * prediction_time
        hidden_size=CONFIG.HIDDEN_SIZE,
        lstm_layers=CONFIG.LSTM_LAYERS,
        num_attention_heads=CONFIG.ATT_HEADS,
        dropout=CONFIG.DROPOUT,
        batch_size=CONFIG.BATCH_SIZE, # change to batch_size
        n_epochs=CONFIG.EPOCH, # change to epoch
        optimizer_kwargs={"lr": CONFIG.L_RATE},
        loss_fn=torch.nn.MSELoss(),
        add_encoders={ # automate extract future_cov from timestamp/log_time
            'cyclic': {'future': ['minute', 'second', 'hour']},
            'transformer': Scaler()
        },
        pl_trainer_kwargs={ # trainer from pytorch lightning
            "accelerator": "auto",
            # "callbacks": [checkpoint_callback],
            "logger": logger,
            # "enable_checkpointing":True,
            "log_every_n_steps": 1
        },
        random_state=42, # seed so the experiment can be reproduced
        work_dir="logs",
        save_checkpoints=True,
    )

    print("-----Start Training-----")

    model.fit(
        series=train_target,
        past_covariates=train_past_cov,
        val_series=val_target,
        val_past_covariates=val_past_cov,
        verbose=True,
        max_samples_per_ts=CONFIG.SAMPLE_PER_TS 
    )

    metrics_file = f"logs/tft_run_{timestamp}/version_0/metrics.csv"
    if args.novis :
        visualize_metrics(metrics_file, timestamp)

if __name__ == "__main__":
    main("mmwave_ss.csv")
