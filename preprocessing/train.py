import argparse
from datetime import datetime
import pandas as pd
import random
from sklearn.metrics import (
    f1_score,
    classification_report, 
    confusion_matrix, 
    balanced_accuracy_score, 
    cohen_kappa_score, 
    roc_auc_score
)
from tsai.all import *
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.utils.class_weight import compute_class_weight
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb

import config as CONFIG
from helper_function import *
from helper_data import *

def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--novis', action='store_false', help="chose if would not automaically generate jpg plof for training process. Default : False")
    parser.add_argument('--data', type=str, default="mmwave_ss.csv", help="filename of source data")
    parser.add_argument('--params', type=str, default=None, help="training parameters")
    parser.add_argument('--wandb', action="store_true", help="activate logging to wandb")
    parser.add_argument('--epoch', type=int, default=None, help="number of epoch")
    parser.add_argument('--log', action="store_true", help="display logging on terminal")
    parser.add_argument('--notes', type=str, default="training", help="notes for the experiment")
    args = parser.parse_args()
    return args

def init_wandb(subject_id, periperal, timestamp, 
            project_name="Evaluation",
            lr=0.0001,
            nl=1,
            dr=0.1,
            dm=32,
            nh=1,
            loss_func="CrossEntropyLoss",
            weight_decay=1e-4,
            job_type="train"
    ):
    wandb.init(
        project=project_name, 
        name=f"LOSO_{subject_id}",
        group="LOSO_Evaluation_v3",
        job_type=job_type,
        config={
            "learning_rate": lr,
            "architecture": "TSTPlus",
            "loss-function" : loss_func,
            "n_layers" : nl,
            "dropout": dr,
            "d_model": dm,
            "n_heads": nh,
            "weight_decay": weight_decay,
        },
    )

def main():
    args = load_args()
    train_params = {}
    if args.params is not None:
        train_params = load_config(args.params, trial=args.param_n)

    file = args.data
    periperal = file.split(".")[0]

    ########## LOCAL VARS ############
    BATCH_SIZE = train_params.get("batch_size", CONFIG.BATCH_SIZE)
    ICL = train_params.get("seq_length", CONFIG.INPUT_CHUNK_LEN)
    N_LAYERS = train_params.get("n_layers", CONFIG.N_LAYERS)
    DROPOUT = train_params.get("fc_dropout", CONFIG.DROPOUT)
    D_MODEL = train_params.get("d_model", CONFIG.D_MODEL)
    N_HEADS = train_params.get("n_heads", CONFIG.N_HEADS)
    LR = train_params.get("learning_rate", CONFIG.L_RATE)
    WD = train_params.get("weight_decay", CONFIG.WEIGHT_DECAY)
    EPOCH = args.epoch or CONFIG.EPOCH
    RANDOM_SEED = CONFIG.RANDOM_SEED
    NUM_CLASSES = 3
    ##################################

    random.seed(RANDOM_SEED)
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    basepath = f"logs/{periperal}_{timestamp}"

    data = np.load("data_ready/train.npz")
    X_sec, X_tab, y, groups = data["X_seq"], data["X_tab"], data["y"], data["id"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training using {device}")

    logo = LeaveOneGroupOut()

    all_y_true = []
    all_y_pred = []
    all_y_prob = []

    for fold, (train_idx, val_idx) in enumerate(logo.split(X_sec, y, groups)):
        subject_id = groups[val_idx][0]
        print(f"Fold {fold + 1} | Validation on SUBJECT ID: {subject_id}")

        X_train, X_tab_train, y_train = X_sec[train_idx], X_tab[train_idx], y[train_idx]
        X_val, X_tab_val, y_val = X_sec[val_idx], X_tab[val_idx], y[val_idx]

        classes_in_train = np.unique(y_train)
        weights = compute_class_weight(class_weight='balanced', 
                                       classes=classes_in_train, 
                                       y=y_train)
        safe_weights = np.zeros(NUM_CLASSES)
        for cls, w in zip(classes_in_train, weights):
            safe_weights[cls] = w
            
        class_weights_tensor = torch.tensor(safe_weights, dtype=torch.float32).to(device)
        print(f"Class weights : {safe_weights}")

        train_dataset = LOGODataset(X_train, X_tab_train, y_train)
        val_dataset = LOGODataset(X_val, X_tab_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        model = TSTPlus(
                c_in = 6,
                c_out = NUM_CLASSES, # 3 = multiclass classification
                seq_len = ICL,
                n_layers = N_LAYERS,
                fc_dropout = DROPOUT,
                dropout = DROPOUT,
                d_model = D_MODEL,
                n_heads = N_HEADS,
            ).to(device)
        loss_func = nn.CrossEntropyLoss(weight=class_weights_tensor) # CrossEntropyLoss for multiclass classification
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

        # history = {'train_loss' : [], 
            # 'train_kss_acc': [],
            # 'val_loss' : [],
            # 'val_kss_acc': [],
            # }

        # best_val_loss = float('inf')

        if args.wandb : 
            project_name="LOSO_Evaluation"
            init_wandb(subject_id, periperal, timestamp, project_name,
                       LR, N_LAYERS, DROPOUT, D_MODEL, N_HEADS, loss_func.__class__.__name__, weight_decay=WD,
                       job_type="train_fold")
            wandb.run.notes = args.notes
            class_weight = {
                "awake": safe_weights[0],
                "drowsy": safe_weights[1],
                "sleep": safe_weights[2]
            }
            wandb.config.update({"class_weights":class_weight})

        for e in range(EPOCH):
            # TRAINING
            model.train()
            total_train_loss = 0.0
            total_train_acc = 0.0

            total_samples = 0

            for data_seq, data_tab, label in train_loader:
                data = data_seq.to(device)
                data_tab = data_tab.to(device) 
                label = label.to(device)

                batch_size = data_seq.size(0)
                total_samples += batch_size

                pred_label = model(data)
                loss = loss_func(pred_label.squeeze(-1), label)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                with torch.no_grad():
                    total_train_loss += loss.item() * batch_size

                    pred_class = torch.argmax(pred_label, dim=1)
                    total_train_acc += (pred_class == label).sum().item()
            
            epoch_loss = total_train_loss / total_samples
            epoch_acc = total_train_acc / total_samples

            # history['train_loss'].append(epoch_loss)
            # history['train_kss_acc'].append(epoch_acc)
            
            if args.log or e == EPOCH - 1 :
                print(f"[Epoch {e+1}/{EPOCH}] Loss : {epoch_loss:.4f} | Accuracy : {epoch_acc:.4f}")

            # VALIDATION
            model.eval()
            fold_y_true = []
            fold_y_pred = []
            fold_y_prob = []

            total_val_loss = 0.0

            total_samples = 0

            for data_seq, data_tab, label in val_loader:
                data = data_seq.to(device)
                data_tab = data_tab.to(device) 
                label = label.to(device)

                batch_size = data_seq.size(0)
                total_samples += batch_size

                with torch.no_grad():
                    pred_label = model(data)
                    loss = loss_func(pred_label.squeeze(-1), label)
                    total_val_loss += loss.item() * batch_size

                    pred_class = torch.argmax(pred_label, dim=1)
                    # _, preds = torch.max(outputs, 1)
                    probs = F.softmax(pred_label, dim=1)

                    fold_y_pred.extend(pred_class.cpu().numpy())
                    fold_y_true.extend(label.cpu().numpy())
                    fold_y_prob.extend(probs.cpu().numpy())

            if e == EPOCH - 1:
                all_y_true.extend(fold_y_true)
                all_y_pred.extend(fold_y_pred)
                all_y_prob.extend(fold_y_prob)

            fold_acc = np.mean(np.array(fold_y_true) == np.array(fold_y_pred))
            epoch_val_loss = total_val_loss / total_samples

            if args.log or e == EPOCH - 1 :
                print(f"[Epoch {e+1}/{EPOCH}] Val Accuracy : {fold_acc:.4f}")

            # if epoch_val_loss < best_val_loss:
            #     best_val_loss = epoch_val_loss
            #     f_model_path = f"{basepath}/models/best_weight.pth"
            #     os.makedirs(f"{basepath}/models", exist_ok=True)
            #     torch.save(model.state_dict(), f_model_path)

            #     os.makedirs(f"{basepath}/checkpoints", exist_ok=True)
            #     f_checkpoint_path = f"{basepath}/checkpoints/best-epoch.pth.tar"
            #     checkpoint = {
            #         'epoch': e,
            #         'model_state_dict': model.state_dict(),
            #         'optimizer_state_dict': optimizer.state_dict(),
            #         'best_val_loss': best_val_loss,
            #     }
            #     torch.save(checkpoint, f_checkpoint_path)
            
            # os.makedirs(f"{basepath}/checkpoints", exist_ok=True)
            # f_checkpoint_path = f"{basepath}/checkpoints/last-epoch.pth.tar"
            # checkpoint = {
            #     'epoch': e,
            #     'model_state_dict': model.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            #     'best_val_loss': best_val_loss,
            # }
            # torch.save(checkpoint, f_checkpoint_path)

            val_bal_acc = balanced_accuracy_score(fold_y_true, fold_y_pred)
            report_dict = classification_report(fold_y_true, fold_y_pred, zero_division=0, target_names=["Awake", "Drowsy", "Sleep"], output_dict=True)
            epoch_kappa = cohen_kappa_score(fold_y_true, fold_y_pred)
            
            if args.wandb : 
                wandb.log({
                    "Acc/train": epoch_acc,
                    "Acc/val": fold_acc,
                    "Acc/val_balanced" : val_bal_acc,
                    "Loss/train": epoch_loss,
                    "Loss/val": epoch_val_loss,
                    "Metrics_Macro/F1": report_dict["macro avg"]["f1-score"],
                    "Metrics_Macro/Precision": report_dict["macro avg"]["precision"],
                    "Metrics_Macro/Recall": report_dict["macro avg"]["recall"],
                    "Metrics_Macro/Kappa": epoch_kappa,
                    "Metrics_Drowsy/F1": report_dict["Drowsy"]["f1-score"],
                    "Metrics_Drowsy/Precision": report_dict["Drowsy"]["precision"],
                    "Metrics_Drowsy/Recall": report_dict["Drowsy"]["recall"],
                })

                if e == EPOCH -1:
                    df_report = pd.DataFrame(report_dict).transpose()
                    # df_report.reset_index(inplace=True)
                    # df_report.columns = ['class', 'precision', 'recall', 'f1-score', 'support']
                    wandb.log({
                        "confusion_matrix": wandb.plot.confusion_matrix(
                            preds=fold_y_pred, 
                            y_true=fold_y_true, 
                            class_names=["Awake", "Drowsy", "Sleep"]
                        ),
                        "roc_curve": wandb.plot.roc_curve(
                            fold_y_true, 
                            fold_y_prob, 
                            labels=["Awake", "Drowsy", "Sleep"]
                        ),
                        "classification_report": wandb.Table(dataframe=df_report)
                    })

        if args.wandb : wandb.finish()

    if args.wandb : 
        subject_id = "Global"
        project_name="LOSO_Evaluation"
        init_wandb(subject_id, periperal, timestamp, project_name,
                    LR, N_LAYERS, DROPOUT, D_MODEL, N_HEADS, loss_func.__class__.__name__, weight_decay=WD,
                    job_type="global_eval")
        wandb.run.notes = args.notes

    cm = confusion_matrix(all_y_true, all_y_pred)
    print("Confusion Matrix")
    print(cm)
    cr = classification_report(all_y_true, all_y_pred, zero_division=0, target_names=["Awake", "Drowsy", "Sleep"])
    report_dict = classification_report(all_y_true, all_y_pred, target_names=["Awake", "Drowsy", "Sleep"], output_dict=True, zero_division=0)
    df_global_report = pd.DataFrame(report_dict).transpose()
    print("Classification Report")
    print(cr)

    global_macro_f1 = f1_score(all_y_true, all_y_pred, average='macro', zero_division=0)
    print(f"MACRO F1 : {global_macro_f1:.4f}")

    global_bal_acc = balanced_accuracy_score(all_y_true, all_y_pred)
    print(f"BALANCED ACCURACY : {global_bal_acc:.4f}")

    kappa = cohen_kappa_score(all_y_true, all_y_pred)
    print(f"COHEN'S KAPPA    : {kappa:.4f}")

    # use multi_class='ovr' (One-vs-Rest) cuz 3 class
    roc_auc = roc_auc_score(all_y_true, np.array(all_y_prob), multi_class='ovr', average='macro', labels=[0, 1, 2])
    print(f"MACRO ROC-AUC    : {roc_auc:.4f}")

    os.makedirs(f"{basepath}", exist_ok=True)
    report_path = f"{basepath}/report.txt"
    with open(report_path, "w") as f:
        f.write(f"Epoch : {EPOCH} \n\n")
        f.write("Confusion Matrix \n")
        f.write(np.array2string(cm, separator=", "))
        f.write("\n\n")
        f.write("Classification Report \n")
        f.write(cr)
        f.write("\n\n")
        f.write(f"BALANCED ACCURACY : {global_bal_acc:.4f}")
        f.write(f"COHEN'S KAPPA    : {kappa:.4f}")
        f.write(f"MACRO ROC-AUC    : {roc_auc:.4f}")


    wandb.log({
        "Global/balanced_accuracy": global_bal_acc,
        "Global/macro_f1": global_macro_f1,
        "Global/macro_precision": report_dict["macro avg"]["precision"],
        "Global/macro_recall": report_dict["macro avg"]["f1-score"],
        "Global/kappa": kappa,
        "global_confusion_matrix": wandb.plot.confusion_matrix(
            preds=all_y_pred, 
            y_true=all_y_true, 
            class_names=["Awake", "Drowsy", "Sleep"]
        ),
        "global_roc_curve": wandb.plot.roc_curve(
            all_y_true, 
            all_y_prob, 
            labels=["Awake", "Drowsy", "Sleep"]
        ),
        "global_classification_report": wandb.Table(dataframe=df_global_report)
    })

    wandb.finish()

if __name__ == "__main__":
    main()