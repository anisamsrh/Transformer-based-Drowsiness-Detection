import argparse
import numpy as np
import torch
import torchmetrics

from darts import TimeSeries
from darts.metrics.metrics import mae, rmse, mape
from darts.models import TFTModel

from helper_function import load_data_as_ts_list, reverse_fixed_scaling

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
    n_horizon = 10
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