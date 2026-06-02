import argparse
import csv
from datetime import datetime
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import time
import torch
from tsai.all import *

import config as CONFIG
from helper_function import load_data_as_df_list, reverse_fixed_scaling, create_loader

def load_model(weight_path, train_params):
    model = TSTPlus(
            c_in = 2,
            c_out = 1,
            seq_len = train_params.get("seq_length", CONFIG.INPUT_CHUNK_LEN),
            n_layers = train_params.get("n_layers", CONFIG.N_LAYERS),
            fc_dropout = train_params.get("fc_dropout", CONFIG.DROPOUT),
            d_model = train_params.get("d_model", CONFIG.D_MODEL),
            n_heads = train_params.get("n_heads", CONFIG.ATT_HEADS),
        )
    
    state_dict = torch.load(weight_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    return model

def load_config(file_path):
    config = {}
    with open(file_path, 'r') as f:
        config = json.load(f)
    return config

def get_file_name(folder):
    folders_path = glob.glob(f"data_{folder}/*")
    folders = []
    for f in folders_path:
        folders.append(Path(f).name)
    return folders

def get_model_name(model_path):
    path_obj = Path(model_path)
    return path_obj.parent.parent.name

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--novis', action='store_false')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--data', type=str, default="mmwave_ss.csv", help="filename of source data")
    parser.add_argument('--params', type=str, default=None, help="training parameters")
    args = parser.parse_args()

    model_path = args.model
    train_params = load_config(args.params)
    model = load_model(model_path, train_params)
    
    file = args.data
    test_df_list = load_data_as_df_list("test", file)
    test_loader = create_loader(test_df_list, i_chunk_len=train_params.get("seq_length", CONFIG.INPUT_CHUNK_LEN), batch_size=CONFIG.BATCH_SIZE)
    file_names = get_file_name("test")
    
    n_pred_iterate = CONFIG.PRED_ITERATE
    
    results_list = []
    bins = [0, 3.1, 6.1, 9.1] 
    labels = ['Class A', 'Class B', 'Class C']
    all_df_combined = []

    for idx, target in enumerate(test_target):
        total_windows_count += len(pred_historical[idx])
        df = target.to_dataframe()
        df["kss_score"] = reverse_fixed_scaling(df["kss_score"], 1, 9)
        reversed_target = TimeSeries.from_dataframe(df)

        clas_point = []
        pred_point = []
        all_mae = []
        all_rmse = []
        all_mape = []

        for jdx, p in enumerate(pred_historical[idx]):
            df_p = p.to_dataframe()
            df_p["kss_score"] = reverse_fixed_scaling(df_p["kss_score"], 1, 9)
            reversed_p = TimeSeries.from_dataframe(df_p)

            clas_val = df_p["kss_score"].iloc[0]
            clas_idx = df_p.index[0]
            
            pred_val = df_p["kss_score"].iloc[-1]
            pred_idx = df_p.index[-1]
            
            clas_point.append((clas_idx, clas_val))
            pred_point.append((pred_idx, pred_val))
            
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
        
        base_test_path = f"results_metrics/test/{get_model_name(args.model)}"
        os.makedirs(base_test_path, exist_ok=True)

        def plot_real_vs_pred(points_list, label):
            plt.figure(figsize=(10, 6))
            df['kss_score'].plot(label='Real')

            indices = [x[0] for x in points_list]
            values = [x[1] for x in points_list]
            series = pd.Series(values, index=indices)
            
            series.plot(label=f"{label}", color="red", linestyle="--")
            plt.title(f"Real vs {label.title()}")
            plt.xlabel("Time")
            plt.ylabel("KSS Score")
            plt.ylim(0, 10)
            plt.yticks(np.arange(1, 10, 1))
            plt.legend()
            plt.grid(True, linestyle=':', alpha=0.7)
            plt.savefig(f"{base_test_path}/{file_names[idx]}_real_vs_{label}.jpg")
            plt.close()

        plot_real_vs_pred(clas_point, "prediction")
        plot_real_vs_pred(pred_point, "forecasting (in 10s)")

        def get_df_combined(points_list):
            indices = [x[0] for x in points_list]
            values = [x[1] for x in points_list]
            series = pd.Series(values, index=indices, name="kss_pred")
            df_combined = pd.DataFrame(df[["kss_score"]].rename(columns={"kss_score": "kss_real"}).join(series, how='inner'))
            return df_combined

        def plot_confussion_matrix(points_list):
            df_combined = get_df_combined(points_list)
            
            df_combined['class_real'] = pd.cut(df_combined['kss_real'], bins=bins, labels=labels, include_lowest=True)
            df_combined['class_pred'] = pd.cut(df_combined['kss_pred'], bins=bins, labels=labels, include_lowest=True)
            all_df_combined.append(df_combined)

            cm = confusion_matrix(df_combined['class_real'], df_combined['class_pred'], labels=labels)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
            disp.plot(cmap='Blues')
            plt.title('Confusion Matrix: KSS Class Prediction')
            plt.savefig(f"{base_test_path}/{file_names[idx]}_confussion_matrix.jpg")
            plt.close()

        plot_confussion_matrix(clas_point)

        def plot_residual(points_list):
            df_combined = get_df_combined(points_list)
            df_combined['residual'] = df_combined['kss_real'] - df_combined['kss_pred']

            plt.figure(figsize=(10, 5))
            plt.plot(df_combined.index, df_combined['residual'], label='Residual', color='purple', alpha=0.6)
            plt.axhline(y=0, color='black', linestyle='--', linewidth=1.5)

            plt.title('Residual Plot (Real - Prediction)')
            plt.xlabel('Time')
            plt.ylabel('Error')
            plt.legend()
            plt.grid(True, linestyle=':', alpha=0.7)

            plt.savefig(f"{base_test_path}/{file_names[idx]}_residual_plot.jpg")
            plt.close()

        plot_residual(clas_point)

        results_list.append({
            "File": file_names[idx],
            "Window_Count": len(pred_historical[idx]),
            "MAE": np.mean(all_mae),
            "RMSE": np.mean(all_mape),
            "MAPE": np.mean(all_rmse),
        })

    df_results = pd.DataFrame(results_list)
    df_results.to_csv(f"{base_test_path}/model_summary_metrics.csv", index=False)
    summary = df_results.groupby("File")[["MAE", "RMSE", "MAPE"]].mean()
    mean_row = summary.mean().rename("MEAN")
    std_row = summary.std().rename("STD")
    final_table = pd.concat([summary, mean_row.to_frame().T, std_row.to_frame().T])
    final_table = final_table.round(4)
    print(final_table)

    avg_inference_time = (end_time-start_time)/(n_pred_iterate*total_windows_count)*1000
    file_exists = os.path.isfile('results_metrics/test/inference_time_log.csv')
    with open('results_metrics/test/inference_time_log.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['model_name', 'inference_time_ms', 'timestamp'])
        writer.writerow([get_model_name(args.model), avg_inference_time, datetime.now().strftime("%d-%m-%Y_%H-%M-%S")])
        
    print("Avg. Inference Time : ", avg_inference_time, " ms")

    def save_table_with_stats(df, filename):
        fig, ax = plt.subplots(figsize=(10, len(df) * 0.5 + 2))
        ax.axis('off')
        
        table = ax.table(cellText=df.values, 
                        rowLabels=df.index, 
                        colLabels=df.columns, 
                        loc='center', 
                        cellLoc='center')
        
        num_rows = len(df)
        table.scale(1, 1.5)
        table[(num_rows - 1, -1)].set_facecolor("#e6f3ff")
        table[(num_rows, -1)].set_facecolor("#f2f2f2")
        for i in range(len(df.columns)):
            table[(num_rows - 1, i)].set_facecolor("#e6f3ff")
            table[(num_rows, i)].set_facecolor("#f2f2f2")
            table[(num_rows - 1, i)].get_text().set_fontweight('bold')
            table[(num_rows, i)].get_text().set_fontweight('bold')

        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()

    save_table_with_stats(final_table, f"{base_test_path}/model_summary_with_stats.png")

    all_df_class = pd.concat(all_df_combined, ignore_index=True)

    def plot_gloal_cm():
        cm = confusion_matrix(all_df_class['class_real'], all_df_class['class_pred'], labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap='Blues')
        plt.title('Confusion Matrix: KSS Class Prediction')
        plt.savefig(f"{base_test_path}/model_confussion_matrix.jpg")
        plt.close()

    plot_gloal_cm()
    
if __name__ == "__main__":
    main()