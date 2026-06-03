import argparse
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn

from helper_function import *

def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default="mmwave_ss.csv", help="filename of source data")
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--rvsp', action='store_true', help="create real vs prediction visualization")
    args = parser.parse_args()
    return args


def main():
    args = load_args()
    file = args.data
    test_df= load_data_as_df(file)
    test_loader = create_single_loader(test_df, icl=120, bs=64)

    kss_real = []
    kss_pred = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = args.model
    train_params = {
        "input_chunk_length": 120,
        "fc_dropout": 0.09922139173047125,
        "n_layers": 2,
        "learning_rate": 0.00014279267867086287,
        "n_heads": 8,
        "d_model": 32
    }
    model = load_model(model_path, train_params)
    loss_func = nn.MSELoss()

    model.eval()
    with torch.no_grad():
        for (data, label) in test_loader:
            data = data.to(device).permute(0, 2, 1)
            label = label.float().to(device)
            print(data.shape)
            print(label.shape)

            pred_kss = model(data)
            
            u_label = reverse_fixed_scaling(label, 1, 9)
            u_pred = reverse_fixed_scaling(pred_kss, 1, 9)

            kss_real.extend(u_label)
            kss_pred.extend(u_pred)

    if args.rvsp:
        def vis_real_vs_pred():
            plt.figure(figsize=(10, 6))
            plt.plot(kss_real, label="real")
            plt.plot(kss_pred, label="pred")
            plt.title("Real vs Prediction")
            plt.xlabel("Time")
            plt.ylabel("KSS Value")
            plt.show()

        vis_real_vs_pred()

if __name__ == "__main__":
    main()