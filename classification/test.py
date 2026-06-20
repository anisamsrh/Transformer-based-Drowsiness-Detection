import argparse
import pandas as pd
from tsai.all import *
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

import config as CONFIG
from helper_function import *

def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default="mmwave_ss.csv", help="filename of source data")
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--n_class', type=int, default=3, help="number of class")
    args = parser.parse_args()
    return args

def main():
    args = load_args()
    params = {}
    if args.params is not None:
        params = load_config(args.params, trial=args.param_n)

    file = args.data
    folder, filename = file.split("/")
    basename = f"{folder.split('_')[0]}_{filename.split('.')[0]}"
    # test_df_list = load_data_as_df_list_scaled("test", file)
    test_df = load_data_as_df_scaled(file)

    ########## LOCAL VARS ############
    BATCH_SIZE = params.get("batch_size", CONFIG.BATCH_SIZE)
    ICL = params.get("seq_length", CONFIG.INPUT_CHUNK_LEN)
    N_LAYERS = params.get("n_layers", CONFIG.N_LAYERS)
    DROPOUT = params.get("fc_dropout", CONFIG.DROPOUT)
    D_MODEL = params.get("d_model", CONFIG.D_MODEL)
    N_HEADS = params.get("n_heads", CONFIG.N_HEADS)
    LR = params.get("learning_rate", CONFIG.L_RATE)
    EPOCH = args.epoch or CONFIG.EPOCH
    RANDOM_SEED = CONFIG.RANDOM_SEED
    NUM_CLASSES = args.n_class or 3
    ##################################

    test_loader = create_single_loader_tsdc(test_df, ICL, BATCH_SIZE)
    model_path = args.model
    model_params = {
        "c_out" : NUM_CLASSES, # 3 = multiclass classification
        "seq_len" : ICL,
        "n_layers" : N_LAYERS,
        "fc_dropout" : DROPOUT,
        "d_model" : D_MODEL,
        "n_heads" : N_HEADS,
    }
    model = load_model(model_path, model_params)
    
    real = []
    pred = []

    model.eval()
    with torch.no_grad():
        for (data, label) in test_loader:
            data = data.permute(0, 2, 1)
            label = label.long()

            pred_logits = model(data)
            pred_class = torch.argmax(pred_logits, dim=1)

            real.extend(label)
            pred.extend(pred_class)
    
    def vis_real_vs_pred():
        plt.figure(figsize=(10, 6))
        plt.plot(real, label="real")
        plt.plot(pred, label="pred")
        plt.title("Real vs Prediction")
        plt.xlabel("Time")
        plt.ylabel("Class")

        pred_series = pd.Series(pred)
        pred_series = pred_series.rolling(window=100, min_periods=1).mean()
        plt.plot(pred_series, label="pred (smoothed)", linestyle='--', color="blue")

        plt.legend()
        basepath = f"test"
        os.makedirs(basepath, exist_ok=True)
        plt.savefig(f"{basepath}/{basename}_rvp.jpg", bbox_inches='tight', dpi=300)

    def vis_cm(real, pred):
        cm = confusion_matrix(real, pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        basepath = f"test"
        os.makedirs(basepath, exist_ok=True)
        plt.savefig(f"{basepath}/{basename}_cm.jpg", bbox_inches='tight', dpi=300)
    
    vis_real_vs_pred(real, pred)
    vis_cm(real, pred)
    print(classification_report(real, pred))

if __name__ == "__main__":
    main()