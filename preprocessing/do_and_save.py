import glob
import numpy as np
import os
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from scipy.signal import medfilt

def get_id(nf):
    id = nf.split("_")
    return str(Path(id[0]).name)

def get_kss_score(nf) : 
    kss = nf.split("_")
    return int(kss[3])

def get_pre_post_score(nf) : 
    kss = nf.split("_")
    return int(kss[2]), int(kss[3])

def do_cleaning_scaling(df):
    scaler = StandardScaler()
    df["log_time"] = pd.to_datetime(df['log_time'])
    df = df.set_index("log_time")
    df = df.resample("1s").mean().interpolate(method="linear").dropna()
    df = df.reset_index()

    df['delta_hr'] = df['heart_rate'].diff().abs()
    batas_fisiologis = 10 
    kondisi_spike = df['delta_hr'] > batas_fisiologis
    kondisi_drop = (df['heart_rate'] < 40) | (df['heart_rate'] > 190)
    df['hr_clean'] = df['heart_rate'].copy()
    df.loc[kondisi_spike | kondisi_drop, 'hr_clean'] = np.nan
    df['hr_clean'] = df['hr_clean'].interpolate(method='linear')

    df['delta_br'] = df['breath_rate'].diff().abs()
    batas_fisiologis_br = 4 
    kondisi_spike = df['delta_br'] > batas_fisiologis_br
    kondisi_drop = (df['breath_rate'] < 6) | (df['breath_rate'] > 60)
    df['br_clean'] = df['breath_rate'].copy()
    df.loc[kondisi_spike | kondisi_drop, 'br_clean'] = np.nan
    df['br_clean'] = df['br_clean'].interpolate(method='linear')

    df["heart_rate"] = medfilt(df["hr_clean"], kernel_size=5)
    df["breath_rate"] = medfilt(df["br_clean"], kernel_size=5)

    df_scaled = pd.DataFrame()
    df_scaled["heart_rate"] = df["heart_rate"]
    df_scaled["breath_rate"] = df["breath_rate"]
    df_scaled["timestamp"] = df["log_time"]
    df_scaled["delta_hr"] = df["delta_hr"]
    df_scaled["delta_br"] = df["delta_br"]
    return df_scaled

def do_windowing(df_list, window_size=60, stride=15):
    x_sequences = []
    x_tabular = []
    y_labels = []
    ids = []

    for df in df_list:
        df = df.iloc[1:].reset_index()
        features = df[["breath_rate", "heart_rate", "delta_br", "delta_hr", "savgol_br", "savgol_hr"]].values
        labels = df["class"].values
        id = df["id"][0]

        total_windows = (len(features) - window_size) // stride + 1

        for i in range(total_windows):
            window_seq = features[i*stride:i*stride+window_size]

            mean_features = np.mean(window_seq, axis=0)
            std_features = np.std(window_seq, axis=0)
            combined_features = np.concatenate([mean_features, std_features])

            x_sequences.append(window_seq)
            x_tabular.append(combined_features)
            y_labels.append(labels[i*stride + window_size - 1])
            ids.append(id)

    x_sequence = np.array(x_sequences, dtype=np.float32)
    x_tabular = np.array(x_tabular, dtype=np.float32)
    y_labels = np.array(y_labels, dtype=np.int64)
    return x_sequences, x_tabular, y_labels, ids

def extract_savgol(df):
    from scipy.signal import savgol_filter
    df["savgol_hr"] = savgol_filter(df["heart_rate"], window_length=11, polyorder=2)
    df["savgol_br"] = savgol_filter(df["breath_rate"], window_length=11, polyorder=2)
    return df

def scaling(dfs):
    df_all = pd.concat(dfs, ignore_index=True)
    
    cols = ["heart_rate", "breath_rate", "delta_hr", "delta_br"]
    for col in cols:
        df_all[col] = df_all.groupby('id')[col].transform(
            lambda x: StandardScaler().fit_transform(x.to_frame()).flatten()
        )
        
    start_idx = 0
    for df in dfs:
        n_rows = len(df)
        end_idx = start_idx + n_rows
        df[cols] = df_all.loc[start_idx : end_idx - 1, cols].values
        start_idx = end_idx
        
    return dfs

def load_data_as_df_list_tsdc(ft, file, classes=3, filter=[]) : 
    current_dir = Path(__file__).resolve().parent
    parent = current_dir.parent
    if len(filter) == 0:
        folders = glob.glob(f"{parent}/{ft}/*")
    else:
        folders = [f for f in glob.glob(f"{parent}/{ft}/*")
                    if any(str(keyword) in Path(f).name for keyword in filter)
                    ]
    pd_list = []

    for f in folders:
        df = pd.read_csv(f"{f}/{file}")
        df_scaled = do_cleaning_scaling(df)
        df_scaled["id"] = get_id(f)
        df_scaled["kss_score"] = get_kss_score(f)
        if classes == 2 :
            df_scaled["class"] = [0 if v < 7 else 1 for v in df_scaled["kss_score"].values]
        elif classes == 3:
            df_scaled["class"] = [0 if v <= 4 else (1 if v <=7 else 2) for v in df_scaled["kss_score"].values]
        pd_list.append(df_scaled)

    pd_list = scaling(pd_list)
    for df in pd_list:
        df = extract_savgol(df)
    return pd_list

def load_data(ft, file, classes=3, filter=[]) : 
    current_dir = Path(__file__).resolve().parent
    parent = current_dir.parent
    if len(filter) == 0:
        folders = glob.glob(f"{parent}/{ft}/*")
    else:
        folders = [f for f in glob.glob(f"{parent}/{ft}/*")
                    if any(str(keyword) in Path(f).name for keyword in filter)
                    ]
    pd_list = []

    for f in folders:
        df = pd.read_csv(f"{f}/{file}")
        df_scaled = do_cleaning_scaling(df)
        df_scaled["id"] = get_id(f)

        df_start = df_scaled.iloc[:301].reset_index(drop=True)
        df_end = df_scaled.iloc[-301:].reset_index(drop=True)
        pre, post = get_pre_post_score(f)

        df_start["kss_score"] = pre
        df_end["kss_score"] = post
        if classes == 2 :
            df_start["class"] = [0 if v < 7 else 1 for v in df_start["kss_score"].values]
            df_end["class"] = [0 if v < 7 else 1 for v in df_end["kss_score"].values]
        elif classes == 3:
            df_start["class"] = [0 if v <= 4 else (1 if v <=7 else 2) for v in df_start["kss_score"].values]
            df_end["class"] = [0 if v <= 4 else (1 if v <=7 else 2) for v in df_end["kss_score"].values]
        pd_list.append(df_start)
        pd_list.append(df_end)

    pd_list = scaling(pd_list)
    for df in pd_list:
        df = extract_savgol(df)
        
    return pd_list

file = "mmwave_ss.csv"
save_path = "data_ready"
os.makedirs(save_path, exist_ok=True)

train_df_list = load_data_as_df_list_tsdc("data", file)
# train_df_list = load_data("data", file)
X_sequence, X_tabular, y_labels, ids = do_windowing(train_df_list)
np.savez(f"{save_path}/train.npz", X_seq=X_sequence, X_tab=X_tabular, y=y_labels, id=ids)

# val_df_list = load_data_as_df_list_tsdc("data", file)
# X_sequence, X_tabular, y_labels, ids = do_windowing(val_df_list)
# np.savez(f"{save_path}/val.npz", X_seq=X_sequence, X_tab=X_tabular, y=y_labels, id=ids)