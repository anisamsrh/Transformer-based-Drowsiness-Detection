import argparse
from datetime import datetime
import glob
import numpy as np
import pandas as pd
from tsai.all import *
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from custom_class import TimeSeriesDataset
from helper_function import load_data_as_df_list, reverse_fixed_scaling

def load_config(file_path):
    with open(file_path, 'r') as f:
        config = json.load(f)
    return config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--novis', action='store_false', help="chose if would not automaically generate jpg plof for training process. Default : False")
    parser.add_argument('--data', type=str, default="mmwave_ss.csv", help="filename of source data")
    parser.add_argument('--params', type=str, default=None, help="training parameters")
    parser.add_argument('--log', action="store_true", help="display logging on terminal")
    args = parser.parse_args()

    file = args.data
    periperal = file.split(".")[0]
    train_df_list = load_data_as_df_list("train", file)
    val_df_list = load_data_as_df_list("val", file)

    def create_loader(df_list):
        loader = []
        for df in df_list :
            dataset = TimeSeriesDataset(config, df)
            loader.append(DataLoader(
                dataset,
                batch_size=config.BATCH_SIZE,
                shuffle=True
            )
        )
        return loader

    train_loader = create_loader(train_df_list)
    val_loader = create_loader(val_df_list)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training using {device}")
    model = TSTPlus(2, 1, 120).to(device)

    loss_func = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.L_RATE)

    history = {'train_loss' : [], 
           'train_kss_mae': [],
           'train_kss_mse': [],
           'train_kss_acc': [],
           'val_loss' : [],
           'val_kss_mae': [],
           'val_kss_mse': [],
           'val_kss_acc': [],
        }

    best_val_loss = float('inf')
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    basepath = f"logs/{periperal}_{timestamp}"

    for epoch in range(config.EPOCH):
        # TRAINING
        model.train()
        total_train_loss = 0.0
        total_train_mae = 0.0
        total_train_acc = 0.0
        total_train_mse = 0.0

        total_samples = 0

        for loader in train_loader:
            for batch_idx, (data, label) in enumerate(loader):
                data = data.to(device).permute(0, 2, 1)
                label = label.float().to(device)

                batch_size = data.size(0)
                total_samples += batch_size

                optimizer.zero_grad()
                pred_kss = model(data)
                loss_kss = loss_func(pred_kss, label)
                loss_kss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                with torch.no_grad():
                    total_train_loss += loss_kss.item() * batch_size
                    u_label = reverse_fixed_scaling(label, 1, 9)
                    u_pred = reverse_fixed_scaling(pred_kss, 1, 9)

                    total_train_mae += torch.abs(u_pred - u_label).sum().item()
                    total_train_mse += F.mse_loss(u_pred, u_label, reduction='sum').item()
                    rounded_kss = torch.clamp(torch.round(u_pred), min=1.0, max=9.0)
                    total_train_acc += (rounded_kss == u_label).sum().item()
        
        epoch_loss = total_train_loss / total_samples
        epoch_mae = total_train_mae / total_samples
        epoch_mse = total_train_mse / total_samples
        epoch_acc = total_train_acc / total_samples

        history['train_loss'].append(epoch_loss)
        history['train_kss_mae'].append(epoch_mae)
        history['train_kss_acc'].append(epoch_acc)
        history['train_kss_mse'].append(epoch_mse)
        
        if args.log or epoch == config.EPOCH - 1 :
            print(f"[Epoch {epoch+1}/{config.EPOCH}] Loss : {epoch_loss:.4f} | MAE : {epoch_mae:.4f} | MSE : {epoch_mse:.4f} | Accuracy : {epoch_acc:.4f}")

        # VALIDATION
        model.eval()
        total_val_loss = 0.0
        total_val_mae = 0.0
        total_val_acc = 0.0
        total_val_mse = 0.0

        total_samples = 0

        for loader in val_loader:
            for batch_idx, (data, label) in enumerate(loader):
                data = data.to(device).permute(0, 2, 1)
                label = label.float().to(device)

                batch_size = data.size(0)
                total_samples += batch_size

                with torch.no_grad():
                    pred_kss = model(data)
                    loss_kss = loss_func(pred_kss, label)

                    total_val_loss += loss_kss.item() * batch_size
                    u_label = reverse_fixed_scaling(label, 1, 9)
                    u_pred = reverse_fixed_scaling(pred_kss, 1, 9)

                    total_val_mae += torch.abs(u_pred - u_label).sum().item()
                    total_val_mse += F.mse_loss(u_pred, u_label, reduction='sum').item()
                    rounded_kss = torch.clamp(torch.round(u_pred), min=1.0, max=9.0)
                    total_val_acc += (rounded_kss == u_label).sum().item()

        epoch_val_loss = total_val_loss / total_samples
        epoch_val_mae = total_val_mae / total_samples
        epoch_val_mse = total_val_mse / total_samples
        epoch_val_acc = total_val_acc / total_samples

        history['val_loss'].append(epoch_val_loss)
        history['val_kss_mae'].append(epoch_val_mae)
        history['val_kss_acc'].append(epoch_val_acc)
        history['val_kss_mse'].append(epoch_val_mse)

        if args.log or epoch == config.EPOCH - 1 :
            print(f"[Epoch {epoch+1}/{config.EPOCH}] Val Loss : {epoch_val_loss:.4f} | Val MAE : {epoch_val_mae:.4f} | Val MSE : {epoch_val_mse:.4f} | Val Accuracy : {epoch_val_acc:.4f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            f_model_path = f"{basepath}/models/best_weight.pth"
            os.makedirs(f"{basepath}/models", exist_ok=True)
            torch.save(model.state_dict(), f_model_path)

            os.makedirs(f"{basepath}/checkpoints", exist_ok=True)
            f_checkpoint_path = f"{basepath}/checkpoints/best-epoch.pth.tar"
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }
            torch.save(checkpoint, f_checkpoint_path)
        
        os.makedirs(f"{basepath}/checkpoints", exist_ok=True)
        f_checkpoint_path = f"{basepath}/checkpoints/last-epoch.pth.tar"
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
        }
        torch.save(checkpoint, f_checkpoint_path)

    metrics = pd.DataFrame(history)
    metrics.to_csv(f"{basepath}/train_eval_{timestamp}.csv", index=False)

    if args.novis:
        def visualize():
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
            os.makedirs("results/train", exist_ok=True)
            plt.savefig(f"results/train/{periperal}_{timestamp}.jpg", bbox_inches='tight', dpi=300)

        visualize()

if __name__ == "__main__":
    main()