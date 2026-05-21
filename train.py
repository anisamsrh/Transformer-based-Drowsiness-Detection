import glob
import random
import time
import os
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

import config
from helper_class import TimeSeriesDataset, PositionalEncoding, TimeSeriesTransformer

def get_kss_score(nf) : 
    # use post-record score
    kss = nf.split("_")
    return float(kss[2])

def load_data_as_cache(folder, file, scaler=None, usecols=["log_time", "heart_rate", "breath_rate"], rename=["timestamp", "hr", "br"]) : 
    dc = {}
    dt = glob.glob(f"data_{folder}/*")

    all_input_data = []
    for i in dt:
        # use either of this if the data spread across files
        # df = pd.read_csv(f"{i}/{file}", usecols=usecols)
        # df = pd.concat((pd.read_csv(f) for f in glob.glob(f"data_{folder}/{i}/mmwave_ss*.csv")), ignore_index=True)

        df = pd.read_csv(f"data_{folder}/{i}/mmwave_ss.csv")
        df.info()
        df.rename(columns=dict(zip(usecols, rename)), inplace=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        df["label"] = get_kss_score(i)

        df = df.set_index("timestamp")
        df = df.resample("1s").mean().interpolate(method="linear")
        df = df.reset_index()

        all_input_data.append(df[["data"]])

        dc[i] = df

    if scaler is None :
        scaler = StandardScaler()
        df_all_input_data = pd.concat(all_input_data, ignore_index=True)
        scaler.fit(df_all_input_data)

    for i in dc.keys():
        dc[i]["data"] = scaler.transform(dc[i][["data"]]).flatten()
    return dc, scaler

train_df, scaler = load_data_as_cache("train", "mmwave_ss.csv")
train_folder = glob.glob(f"data_train/*")
val_df, _ = load_data_as_cache("val", "mmwave_ss.csv", scaler=scaler)
val_folder = glob.glob(f"data_val/*")

model = TimeSeriesTransformer(
    config=config,
    input_dim=config.INPUT_DIM,
    d_model=config.D_MODEL,
    nhead=config.NUM_HEADS,
    num_layers=config.NUM_LAYERS,
)

# Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training using {device}")

model = model.to(device)

# Define Loss Function
classification_criterion = nn.MSELoss()
forecasting_criterion = nn.MSELoss()
kss_f_criterion = nn.MSELoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=config.L_RATE)

history = {'train_loss' : [], 
           'train_kss_mae': [],
           'train_kss_acc': [],
           'train_forecast_mae': [],
           'train_forecast_rmse': [],
           'train_kss_forecast_mae': [],
           'train_kss_forecast_acc': [],
           'val_loss' : [],
           'val_kss_mae': [],
           'val_kss_acc': [],
           'val_forecast_mae': [],
           'val_forecast_rmse': [],
           'train_kss_forecast_mae': [],
           'train_kss_forecast_acc': [],
        }

best_val_loss = float('inf')
timestamp = time.time()

for epoch in range(config.EPOCH):
    # TRAINING
    model.train()
    total_train_loss = 0.0
    train_kss_mae_sum = 0.0
    train_kss_acc_sum = 0.0
    train_forecast_mae_sum = 0.0
    train_forecast_mse_sum = 0.0
    train_kss_f_mae_sum = 0.0
    train_kss_f_acc_sum = 0.0

    total_samples = 0
    total_batch = 0
    random.shuffle(train_folder)

    for folder in train_folder:
        kss = get_kss_score(folder)

        df = train_df[folder]

        train_dataset = TimeSeriesDataset(config, df)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True
        )
        total_batch += len(train_loader)

        for batch_idx, (data, label, future_data, future_label) in enumerate(train_loader):
            # Move data to GPU if available
            data = data.to(device)
            label = label.float().unsqueeze(1).to(device)
            future_data = future_data.to(device)
            future_label = future_label.unsqueeze(1).to(device)

            batch_size = data.size(0)
            total_samples += batch_size

            # Set gradient from last epoch to zero
            optimizer.zero_grad()

            # get prediction
            pred_kss, pred_forecast, pred_kss_f = model(data)

            # calculate loss
            loss_kss = classification_criterion(pred_kss, label)
            loss_forecast = forecasting_criterion(pred_forecast, future_data)
            loss_kss_f = kss_f_criterion(pred_kss_f, future_label)
            batch_loss = (config.ALPHA_KSS * loss_kss) + (config.ALPHA_FORCASTING * loss_forecast) + (config.ALPHA_KSS * loss_kss_f)
            batch_loss.backward()

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            
            total_train_loss += batch_loss.item()

            # Calculate Metrics
            # 1. KSS MAE
            kss_mae = torch.abs(pred_kss - label).mean().item()
            train_kss_mae_sum += kss_mae * batch_size

            # 2. KSS Accuracy
            rounded_kss = torch.clamp(torch.round(pred_kss), min=1.0, max=9.0)
            correct_kss = (rounded_kss == label).sum().item()
            train_kss_acc_sum += correct_kss

            # 3. Forecast MAE
            forecast_mae = torch.abs(pred_forecast - future_data).mean().item()
            train_forecast_mae_sum += forecast_mae * batch_size

            # 4. Forcast MSE
            forecast_mse = torch.nn.functional.mse_loss(pred_forecast, future_data).item()
            train_forecast_mse_sum += forecast_mse * batch_size

            # 5. Future KSS MAE
            kss_f_mae = torch.abs(pred_kss_f - future_label).mean().item()
            train_kss_f_mae_sum += kss_mae * batch_size

            # 6. Future KSS Accuracy
            rounded_kss_f = torch.clamp(torch.round(pred_kss_f), min=1.0, max=9.0)
            correct_kss_f = (rounded_kss_f == future_label).sum().item()
            train_kss_f_acc_sum += correct_kss_f

    avg_train_loss = total_train_loss / total_batch
    avg_kss_mae = train_kss_mae_sum / total_samples
    avg_kss_acc = train_kss_acc_sum / total_samples # in decimal
    avg_forecast_mae = train_forecast_mae_sum / total_samples
    avg_forecast_rmse = math.sqrt(train_forecast_mse_sum / total_samples)
    avg_kss_f_mae = train_kss_f_mae_sum / total_samples
    avg_kss_f_acc = train_kss_f_acc_sum / total_samples # in decimal

    # Save history
    history['train_loss'].append(avg_train_loss)
    history['train_kss_mae'].append(avg_kss_mae)
    history['train_kss_acc'].append(avg_kss_acc)
    history['train_forecast_mae'].append(avg_forecast_mae)
    history['train_forecast_rmse'].append(avg_forecast_rmse)
    history['train_kss_forecast_mae'].append(avg_kss_f_mae)
    history['train_kss_forecast_acc'].append(avg_kss_f_acc)

    # VALIDATION
    model.eval()
    total_val_loss = 0.0
    val_kss_mae_sum = 0.0
    val_kss_acc_sum = 0.0
    val_forecast_mae_sum = 0.0
    val_forecast_mse_sum = 0.0
    val_kss_f_mae_sum = 0.0
    val_kss_f_acc_sum = 0.0

    total_samples = 0
    total_batch = 0
    random.shuffle(val_folder)

    for folder in val_folder:
        kss = get_kss_score(folder)

        df = val_df[folder]

        val_dataset = TimeSeriesDataset(config, df)
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True
        )
        total_batch += len(val_loader)

        with torch.no_grad() :
            for data, label, future_data, future_label in val_loader:
                data = data.to(device)
                label = label.float().unsqueeze(1).to(device)
                future_data = future_data.to(device)
                future_label = future_label.unsqueeze(1).to(device)

                batch_size = data.size(0)
                total_samples += batch_size
                
                pred_kss, pred_forecast, pred_kss_f = model(data)

                rounded_kss = torch.round(pred_kss)
                rounded_kss = torch.clamp(rounded_kss, min=1.0, max=9.0)

                loss_kss = classification_criterion(pred_kss, label)
                loss_forecast = forecasting_criterion(pred_forecast, future_data)
                loss_kss_f = kss_f_criterion(pred_kss_f, future_label)
                batch_loss = (config.ALPHA_KSS * loss_kss) + (config.ALPHA_FORCASTING * loss_forecast) + (config.ALPHA_KSS * loss_kss_f)
                total_val_loss += batch_loss.item()

                # Calculate Metrics
                # 1. KSS MAE
                kss_mae = torch.abs(pred_kss - label).mean().item()
                val_kss_mae_sum += kss_mae * batch_size

                # 2. KSS Accuracy
                correct_kss = (rounded_kss == label).sum().item()
                val_kss_acc_sum += correct_kss

                # 3. Forecast MAE
                forecast_mae = torch.abs(pred_forecast - future_data).mean().item()
                val_forecast_mae_sum += forecast_mae * batch_size

                # 4. Forcast MSE
                forecast_mse = torch.nn.functional.mse_loss(pred_forecast, future_data).item()
                val_forecast_mse_sum += forecast_mse * batch_size

                # 5. Future KSS MAE
                kss_mae_f = torch.abs(pred_kss_f - future_label).mean().item()
                val_kss_f_mae_sum += kss_mae_f * batch_size

                # 2. KSS Accuracy
                rounded_kss_f = torch.clamp(torch.round(pred_kss_f), min=1.0, max=9.0)
                correct_kss_f = (rounded_kss_f == future_label).sum().item()
                val_kss_f_acc_sum += correct_kss

    avg_val_loss = total_val_loss / total_batch
    avg_kss_mae = val_kss_mae_sum / total_samples
    avg_kss_acc = val_kss_acc_sum / total_samples # in decimal
    avg_forecast_mae = val_forecast_mae_sum / total_samples
    avg_forecast_rmse = math.sqrt(val_forecast_mse_sum / total_samples)
    avg_kss_f_mae = val_kss_f_mae_sum / total_samples
    avg_kss_f_acc = val_kss_f_acc_sum / total_samples # in decimal

    # Save History
    history['val_loss'].append(avg_val_loss)
    history['val_kss_mae'].append(avg_kss_mae)
    history['val_kss_acc'].append(avg_kss_acc)
    history['val_forecast_mae'].append(avg_forecast_mae)
    history['val_forecast_rmse'].append(avg_forecast_rmse)
    history['val_kss_forecast_mae'].append(avg_kss_f_mae)
    history['val_kss_forecast_acc'].append(avg_kss_f_acc)
    print(f"Epoch [{epoch+1}/{config.EPOCH}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        # Save Model Weight
        f_model_path = f"model/weight_{timestamp}.pth"
        if not os.path.exists("model"):
            os.mkdir("model")
        torch.save(model.state_dict(), f_model_path)

    # Saving Checkpoint
    if not os.path.exists("log/checkpoint"):
        os.makedirs("log/checkpoint")
    f_checkpoint_path = f"log/checkpoint/checkpoint_{timestamp}.pth.tar"
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
    }

    torch.save(checkpoint, f_checkpoint_path)
        
if not os.path.exists("log") :
    os.mkdir("log")
metrics = pd.DataFrame(history)
metrics.to_csv(f"log/trai_eval_{timestamp}.csv", index=False)


