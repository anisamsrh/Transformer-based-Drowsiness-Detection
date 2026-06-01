import argparse
from datetime import datetime
import optuna
from tsai.all import *
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from custom_class import TimeSeriesDataset
import config

from helper_function import load_data_as_df_list

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default="mmwave_ss.csv", help="filename of source data")
    parser.add_argument('--n_trials', type=str, default=50, help="number of trials to be done")
    args = parser.parse_args()

    file = args.data
    periperal = file.split(".")[0]
    train_df_list = load_data_as_df_list("train", file)
    val_df_list = load_data_as_df_list("val", file)

    def create_loader(df_list,
        stride=1,
        gap=0,
        i_chunk_len=30,
        o_chunk_len=10,
        ):
        loader = []
        for df in df_list :
            dataset = TimeSeriesDataset(df, stride=stride, gap=gap, i_chunk_len=i_chunk_len, o_chunk_len=o_chunk_len)
            loader.append(DataLoader(
                dataset,
                batch_size=config.BATCH_SIZE,
                shuffle=True
            )
        )
        return loader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_func = nn.MSELoss()

    def objective(trial):
        seq_len = trial.suggest_int("input_chunk_length", 10, 120, step=10)
        fc_dropout = trial.suggest_float("fc_dropout", 0.05, 0.3)
        n_layers = trial.suggest_int("n_layers", 1, 3)
        l_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)

        n_heads = trial.suggest_categorical("n_heads", [1, 2, 4, 8])
        valid_d_model = [x for x in range(16, 129, 16) if x % n_heads == 0]
        d_model = trial.suggest_categorical("d_model", valid_d_model)

        train_loader = create_loader(train_df_list, i_chunk_len=seq_len)
        val_loader = create_loader(val_df_list, i_chunk_len=seq_len)

        train_loader = create_loader(train_df_list, i_chunk_len=seq_len)
        val_loader = create_loader(val_df_list, i_chunk_len=seq_len)

        model = TSTPlus(
            c_in = 2,
            c_out = 1,
            seq_len = seq_len,
            n_layers = n_layers,
            n_heads = n_heads,
            fc_dropout = fc_dropout,
            d_model = d_model,
        )
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=l_rate)

        for epoch in range(4):
            model.train()
            for loader in train_loader:
                for batch_idx, (data, label) in enumerate(loader):
                    data = data.to(device).permute(0, 2, 1)
                    label = label.float().to(device)

                    optimizer.zero_grad()
                    pred_kss = model(data)
                    loss_kss = loss_func(pred_kss, label)
                    loss_kss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
            
            model.eval()
            total_val_loss = 0.0
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
        error = total_val_loss / total_samples
        return error
    
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
