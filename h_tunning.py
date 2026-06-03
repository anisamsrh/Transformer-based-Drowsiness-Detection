import argparse
from datetime import datetime
import optuna
from tsai.all import *
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from custom_class import TimeSeriesDataset
import config as CONFIG

from helper_function import create_kfold_loaders, load_data_as_df_list, reverse_fixed_scaling

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default="mmwave_ss.csv", help="filename of source data")
    parser.add_argument('--n_trials', type=str, default=50, help="number of trials to be done")
    args = parser.parse_args()

    file = args.data
    periperal = file.split(".")[0]
    train_df_list = load_data_as_df_list("train", file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_func = nn.MSELoss()

    def objective(trial):
        seq_len = trial.suggest_int("input_chunk_length", 30, 120, step=10)
        fc_dropout = trial.suggest_float("fc_dropout", 0.05, 0.3)
        n_layers = trial.suggest_int("n_layers", 1, 3)
        l_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)

        n_heads = trial.suggest_categorical("n_heads", [1, 2, 4, 8])
        valid_d_model = [x for x in range(16, 129, 16) if x % n_heads == 0]
        d_model = trial.suggest_categorical("d_model", valid_d_model)

        kfold_dataloaders = create_kfold_loaders(
            train_df_list, 
            k_splits=5, 
            seq_length=seq_len, 
            batch_size=CONFIG.BATCH_SIZE
        )

        fold_validation_scores = []
        fold_validation_acc = []
        for fold, (train_loader, val_loader) in enumerate(kfold_dataloaders):
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

            for epoch in range(2):
                model.train()
                for (data, label) in train_loader:
                    data = data.to(device).permute(0, 2, 1)
                    label = label.float().to(device)

                    optimizer.zero_grad()
                    pred_kss = model(data)
                    loss_kss = loss_func(pred_kss, label)
                    loss_kss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                print("Fold {}, Epoch {}, Loss: {:.4f}".format(fold+1, epoch+1, loss_kss.item()))
                
            model.eval()
            total_val_loss = 0.0
            total_samples = 0
            total_val_acc = 0
            for (data, label) in val_loader:
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
                    rounded_kss = torch.clamp(torch.round(u_pred), min=1.0, max=9.0)
                    total_val_acc += (rounded_kss == u_label).sum().item()

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
        study_name=f"h_tunning_{periperal}_{n_trials}_trials_{timestamp}", 
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
