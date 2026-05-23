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

file = "mmwave_ss.csv"
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
    filename=f"tft_{timestamp}_epoch{{epoch:02d}}_val{{val_loss:.4f}}",
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
        "callbacks": [checkpoint_callback],
        "logger": logger,
        "enable_checkpointing":True,
        "log_every_n_steps": 1
    },
    random_state=42, # seed so the experiment can be reproduced
)

print("-----Start Training-----")

os.makedirs("log/checkpoint", exist_ok=True)

model.fit(
    series=train_target,
    past_covariates=train_past_cov,
    val_series=val_target,
    val_past_covariates=val_past_cov,
    verbose=True,
    max_samples_per_ts=CONFIG.SAMPLE_PER_TS 
)
