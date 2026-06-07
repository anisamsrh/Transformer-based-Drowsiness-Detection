import argparse
from datetime import datetime
import pandas as pd
import random
from tsai.all import *
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb

import config as CONFIG
from helper_function import *

def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--novis', action='store_false', help="chose if would not automaically generate jpg plof for training process. Default : False")
    parser.add_argument('--data', type=str, default="mmwave_ss.csv", help="filename of source data")
    parser.add_argument('--params', type=str, default=None, help="training parameters")
    parser.add_argument('--wandb', action="store_true", help="activate logging to wandb")
    parser.add_argument('--epoch', type=int, default=None, help="number of epoch")
    parser.add_argument('--log', action="store_true", help="display logging on terminal")
    args = parser.parse_args()
    return args

def init_wandb(periperal, timestamp,
        lr=0.0001,
        nl=1,
        dr=0.1,
        dm=32,
        nh=1,
        loss_func="MSELoss",
    ):
    wandb.init(
        project="ML-PROTEL", 
        name=f"model_{periperal}_{timestamp}",
        config={
            "learning_rate": lr,
            "architecture": "TSTPlus",
            "loss-function" : loss_func,
            "n_layers" : nl,
            "dropout": dr,
            "d_model": dm,
            "n_heads": nh,
        },
    )

def main():
    args = load_args()
    train_params = {}
    if args.params is not None:
        train_params = load_config(args.params, trial=args.param_n)

    file = args.data
    periperal = file.split(".")[0]
    train_df_list = load_data_as_df_list_scaled("train", file)
    val_df_list = load_data_as_df_list_scaled("val", file)

    ########## LOCAL VARS ############
    BATCH_SIZE = train_params.get("batch_size", CONFIG.BATCH_SIZE)
    ICL = train_params.get("seq_length", CONFIG.INPUT_CHUNK_LEN)
    N_LAYERS = train_params.get("n_layers", CONFIG.N_LAYERS)
    DROPOUT = train_params.get("fc_dropout", CONFIG.DROPOUT)
    D_MODEL = train_params.get("d_model", CONFIG.D_MODEL)
    N_HEADS = train_params.get("n_heads", CONFIG.N_HEADS)
    LR = train_params.get("learning_rate", CONFIG.L_RATE)
    EPOCH = args.epoch or CONFIG.EPOCH
    RANDOM_SEED = CONFIG.RANDOM_SEED
    ##################################

    random.seed(RANDOM_SEED)
    train_loader = create_loader(train_df_list, 
        batch_size=BATCH_SIZE, 
        i_chunk_len=ICL)
    val_loader = create_loader(val_df_list, 
        batch_size=BATCH_SIZE,
        i_chunk_len=ICL)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training using {device}")
    model = TSTPlus(
            c_in = 2,
            c_out = 1,
            seq_len = ICL,
            n_layers = N_LAYERS,
            fc_dropout = DROPOUT,
            d_model = D_MODEL,
            n_heads = N_HEADS,
        ).to(device)
    loss_func = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

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

    if args.wandb : init_wandb(periperal, timestamp, LR, N_LAYERS, DROPOUT, D_MODEL, N_HEADS, loss_func.__class__.__name__)

    for e in range(EPOCH):
        # TRAINING
        model.train()
        total_train_loss = 0.0
        total_train_mae = 0.0
        total_train_acc = 0.0
        total_train_mse = 0.0

        total_samples = 0

        random.shuffle(train_loader)
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
        
        if args.log or e == EPOCH - 1 :
            print(f"[Epoch {e+1}/{EPOCH}] Loss : {epoch_loss:.4f} | MAE : {epoch_mae:.4f} | MSE : {epoch_mse:.4f} | Accuracy : {epoch_acc:.4f}")

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

        if args.log or e == EPOCH - 1 :
            print(f"[Epoch {e+1}/{EPOCH}] Val Loss : {epoch_val_loss:.4f} | Val MAE : {epoch_val_mae:.4f} | Val MSE : {epoch_val_mse:.4f} | Val Accuracy : {epoch_val_acc:.4f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            f_model_path = f"{basepath}/models/best_weight.pth"
            os.makedirs(f"{basepath}/models", exist_ok=True)
            torch.save(model.state_dict(), f_model_path)

            os.makedirs(f"{basepath}/checkpoints", exist_ok=True)
            f_checkpoint_path = f"{basepath}/checkpoints/best-epoch.pth.tar"
            checkpoint = {
                'epoch': e,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }
            torch.save(checkpoint, f_checkpoint_path)
        
        os.makedirs(f"{basepath}/checkpoints", exist_ok=True)
        f_checkpoint_path = f"{basepath}/checkpoints/last-epoch.pth.tar"
        checkpoint = {
            'epoch': e,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
        }
        torch.save(checkpoint, f_checkpoint_path)

        if args.wandb : wandb.log({
            "train_loss": epoch_loss,
            "train_kss_mae": epoch_mae,
            "train_kss_mse": epoch_mse,
            "train_kss_acc": epoch_acc,
            "val_loss": epoch_val_loss,
            "val_kss_mae": epoch_val_mae,
            "val_kss_mse": epoch_val_mse,
            "val_kss_acc": epoch_val_acc,
        })

    metrics = pd.DataFrame(history)
    metrics.to_csv(f"{basepath}/train_eval_{timestamp}.csv", index=False)

    if args.novis:
        visualize_train(history, periperal, timestamp)

    if args.wandb : wandb.finish()

if __name__ == "__main__":
    main()