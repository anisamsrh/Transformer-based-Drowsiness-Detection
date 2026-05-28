import argparse
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
import torchmetrics

from darts import TimeSeries
from darts.metrics.metrics import mae, rmse, mape
from darts.models import TFTModel

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
    os.makedirs("results_metrics/test", exist_ok=True)
    image_name, _ = os.path.splitext(os.path.basename(file))
    plt.savefig(f"results_metrics/test/{image_name}_{timestamp}.jpg", bbox_inches='tight', dpi=300)

def main(file, model_path=""):
    parser = argparse.ArgumentParser()
    parser.add_argument('--novis', action='store_false')
    parser.add_argument('--model', type=str, default=model_path)
    args = parser.parse_args()

    model_path = args.model
    # timestamp = model_path.split("_")[1]

    torch.serialization.add_safe_globals([
        torch.optim.Adam,
        torch.nn.MSELoss,
        torchmetrics.collections.MetricCollection
        ])
    
    model = TFTModel.load_from_checkpoint(
        model_name="2026-05-29_00_02_34_torch_model_run_11427",
        work_dir="logs",
        file_name="best-epoch=3-val_loss=0.00.ckpt",
        )
    
    test_target, test_past_cov = load_data_as_ts_list("test", file)
    n_horizon = 20
    # pred = model.predict(
    #     n=n_horizon,
    #     series=test_target,
    #     past_covariates=test_past_cov,
    # )

    # for p in pred:
    #     df = p.to_dataframe()
    #     df["kss_score"] = reverse_fixed_scaling(df["kss_score"], 1, 9)
    #     # print(df)
    #     break

    pred_historical = model.historical_forecasts(
        series = test_target,
        past_covariates = test_past_cov,
        forecast_horizon = n_horizon,
        stride = 1,
        retrain = False,
        last_points_only=False,
    )

    all_mae = []
    all_rmse = []
    all_mape = []
    metrics = {
        "avg_mae" : None,
        "avg_rmse" : None,
        "avg_mape" : None,
        "quantile_loss" : None,
    }

    # TODO : output MAE, RMSE, MAPE, Quantile Loss
    # TODO : Plot actual data vs prediction (chose 3 file representing each class)
    # TODO : Plot Confussion Metrics
    # TODO : Residual Plot

    for idx, target in enumerate(test_target):
        df = target.to_dataframe()
        df["kss_score"] = reverse_fixed_scaling(df["kss_score"], 1, 9)
        reversed_target = TimeSeries.from_dataframe(df)
        for p in pred_historical[idx]:
            df_p = p.to_dataframe()
            df_p["kss_score"] = reverse_fixed_scaling(df_p["kss_score"], 1, 9)
            reversed_p = TimeSeries.from_dataframe(df_p)

            mae_result = mae(
                actual_series=reversed_target , pred_series=reversed_p,
            )
            all_mae.append(mae_result)

            rmse_result = rmse(
                actual_series=reversed_target , pred_series=reversed_p,
            )
            all_rmse.append(rmse_result)

            mape_result = mape(
                actual_series=reversed_target , pred_series=reversed_p,
            )
            all_mape.append(mape_result)
    
    metrics["avg_mae"] = np.mean(all_mae)
    metrics["avg_mape"] = np.mean(all_mape)
    metrics["avg_rmse"] = np.mean(all_rmse)

    print(metrics)

    
if __name__ == "__main__":
    main("mmwave_ss.csv")