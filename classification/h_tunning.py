import argparse
from datetime import datetime
import optuna
from tsai.all import *
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from custom_class import TimeSeriesDataset
import config as CONFIG

from helper_function import create_kfold_loaders_tsdc, load_data_as_df_list_tsdc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default="mmwave_ss.csv", help="filename of source data")
    parser.add_argument('--n_trials', type=str, default=50, help="number of trials to be done")
    args = parser.parse_args()

    file = args.data
    periperal = file.split(".")[0]
    train_df_list = load_data_as_df_list_tsdc("train", file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_func = nn.CrossEntropyLoss() #classification

    def objective(trial):
        seq_len = trial.suggest_int("input_chunk_length", 30, 120, step=5)
        fc_dropout = trial.suggest_float("fc_dropout", 0.05, 0.5)
        n_layers = trial.suggest_int("n_layers", 1, 4)
        l_rate = trial.suggest_float("learning_rate", 1e-6, 1e-2, log=True)

        n_heads = trial.suggest_categorical("n_heads", [1, 2, 4, 8])
        valid_d_model = [x for x in range(16, 129, 16) if x % n_heads == 0]
        d_model = trial.suggest_categorical("d_model", valid_d_model)

        kfold_dataloaders = create_kfold_loaders_tsdc(
            train_df_list, 
            k_splits=2, 
            seq_length=seq_len, 
            batch_size=CONFIG.BATCH_SIZE
        )

        fold_validation_scores = []
        fold_validation_acc = []
        for fold, (train_loader, val_loader) in enumerate(kfold_dataloaders):
            model = TSTPlus(
                c_in = 2,
                c_out = 3, #classification 3 class
                seq_len = seq_len,
                n_layers = n_layers,
                n_heads = n_heads,
                fc_dropout = fc_dropout,
                d_model = d_model,
            )
            model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=l_rate)

            for epoch in range(15):
                model.train()
                for (data, label) in train_loader:
                    data = data.to(device).permute(0, 2, 1)
                    label = label.squeeze(1).long().to(device)

                    optimizer.zero_grad()
                    pred_kss = model(data)
                    loss_kss = loss_func(pred_kss.squeeze(-1), label)
                    loss_kss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                
            model.eval()
            total_val_loss = 0.0
            total_samples = 0
            total_val_acc = 0
            for (data, label) in val_loader:
                data = data.to(device).permute(0, 2, 1)
                label = label.squeeze(1).long().to(device)

                batch_size = data.size(0)
                total_samples += batch_size
                with torch.no_grad():
                    pred_kss = model(data)
                    loss_kss = loss_func(pred_kss, label)
                    total_val_loss += loss_kss.item() * batch_size
                    
                    pred_class = torch.argmax(pred_kss, dim=1)
                    total_val_acc += (pred_class == label).sum().item()

            error = total_val_loss / total_samples
            fold_validation_scores.append(error)
            fold_validation_acc.append(total_val_acc / total_samples)
        error = np.array(fold_validation_scores).mean()
        avg_acc = np.array(fold_validation_acc).mean()
        return (error, avg_acc)
    
    n_trials = int(args.n_trials)
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    storage_name = f"sqlite:///ml_tft.db"
    study = optuna.create_study(
        study_name=f"h_tunning_mc_{periperal}_{n_trials}_trials_{timestamp}", 
        storage=storage_name, 
        load_if_exists=True,
        directions=["minimize","maximize"]
        )
    study.optimize(objective, n_trials=n_trials)

    # 1 OBJECTIVE
    # best_params = study.best_params
    # os.makedirs("results_h_tunning", exist_ok=True)
    # with open(f'results_h_tunning/best_config_{periperal}_{n_trials}_trials_{timestamp}.json', 'w') as f:
    #     json.dump(best_params, f, indent=4)
    # print("Best Parameter:", best_params)
    # print("Best Values:", study.best_value)

    # MULTIPLE OBJECTIVE
    best_trials = study.best_trials
    results_to_save = []
    for trial in best_trials:
        results_to_save.append({
            "trial_number": trial.number,
            "values": trial.values,
            "params": trial.params
        })
    os.makedirs("results_h_tunning", exist_ok=True)
    filename = f'results_h_tunning/best_configs_{periperal}_{n_trials}_trials_{timestamp}.json'
    with open(filename, 'w') as f:
        json.dump(results_to_save, f, indent=4)
    for trial in best_trials:
        print(f"Trial {trial.number}: Values={trial.values}, Params={trial.params}")
    
if __name__ == "__main__":
    main()
