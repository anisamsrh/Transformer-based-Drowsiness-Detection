import argparse
from datetime import datetime
import json
import numpy as np
import optuna
import os
import torch

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.metrics.metrics import rmse
from darts.models import TFTModel

from helper_function import load_data_as_ts_list, reverse_fixed_scaling

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--novis', action='store_false', help="not used")
    parser.add_argument('--data', type=str, default="mmwave_ss.csv", help="filename of source data")
    parser.add_argument('--n_trials', type=str, default=50, help="number of trials to be done")
    args = parser.parse_args()

    file = args.data
    train_target, train_past_cov = load_data_as_ts_list("train", file)
    val_target, val_past_cov = load_data_as_ts_list("val", file)

    def objective(trial):
        in_chunk = trial.suggest_int("input_chunk_length", 10, 120, step=10)
        dropout_rate = trial.suggest_float("dropout", 0.05, 0.3)
        n_lstm_layer = trial.suggest_int("lstm_layers", 1, 3)
        l_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)

        att_head = trial.suggest_categorical("num_attention_heads", [1, 2, 4, 8])
        valid_hidden_sizes = [x for x in range(16, 129, 16) if x % att_head == 0]
        hidden_dim = trial.suggest_categorical("hidden_size", valid_hidden_sizes)

        model = TFTModel(
            input_chunk_length=in_chunk,
            output_chunk_length=10,
            hidden_size=hidden_dim,
            dropout=dropout_rate,
            lstm_layers=n_lstm_layer,
            num_attention_heads=att_head,
            optimizer_kwargs={"lr": l_rate}, 

            n_epochs=10, 
            batch_size=64,
            random_state=42,
            loss_fn=torch.nn.MSELoss(),
            add_encoders={ # automate extract future_cov from timestamp/log_time
                'cyclic': {'future': ['minute', 'second', 'hour']},
                'transformer': Scaler()
            },

            pl_trainer_kwargs={"enable_progress_bar": False}
        )

        model.fit(
            series=train_target,
            past_covariates=train_past_cov,
            val_series=val_target,
            val_past_covariates=val_past_cov,
            verbose=False,
            max_samples_per_ts=100
        )

        n_horizon = 10
        pred = model.historical_forecasts(
            series = val_target,
            past_covariates = val_past_cov,
            forecast_horizon = n_horizon,
            stride = 1,
            retrain = False,
            last_points_only=False,
        )

        all_rmse=[]
        for idx, target in enumerate(val_target):
            df = target.to_dataframe()
            df["kss_score"] = reverse_fixed_scaling(df["kss_score"], 1, 9)
            reversed_target = TimeSeries.from_dataframe(df)
            for p in pred[idx]:
                df_p = p.to_dataframe()
                df_p["kss_score"] = reverse_fixed_scaling(df_p["kss_score"], 1, 9)
                reversed_p = TimeSeries.from_dataframe(df_p)

                rmse_result = rmse(
                    actual_series=reversed_target , pred_series=reversed_p,
                )
                all_rmse.append(rmse_result)

        error = np.mean(all_rmse)
        return error
    
    periperal = file.split(".")[0]
    n_trials = int(args.n_trials)
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    storage_name = f"sqlite:///ml_tft.db"
    study = optuna.create_study(
        study_name=f"h_tunning_{periperal}_{n_trials}_trials_{timestamp}", 
        storage=storage_name, 
        load_if_exists=True,
        direction="minimize",
        )
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    os.makedirs("results_h_tunning", exist_ok=True)
    with open(f'results_h_tunning/best_config_{periperal}_{n_trials}_trials_{timestamp}.json', 'w') as f:
        json.dump(best_params, f, indent=4)
    print("Best Parameter:", best_params)
    print("Best Values:", study.best_value)
    
if __name__ == "__main__":
    main()
