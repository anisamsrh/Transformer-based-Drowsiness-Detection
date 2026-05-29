import argparse
from datetime import datetime
from darts.models import TFTModel
from darts.dataprocessing.transformers import Scaler
from pytorch_lightning.loggers import CSVLogger
import torch

import config as CONFIG
from helper_function import load_data_as_ts_list, visualize_metrics

def main(file):
    parser = argparse.ArgumentParser()
    parser.add_argument('--novis', action='store_false')
    args = parser.parse_args()

    file = f"{file}"
    train_target, train_past_cov = load_data_as_ts_list("train", file)
    val_target, val_past_cov = load_data_as_ts_list("val", file)

    print("-----Data Loaded-----")

    timestamp = datetime.now().strftime("%d-%m-%Y-%H:%M:%S")
    logger = CSVLogger(save_dir="logs/", name=f"tft_run_{timestamp}")

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
