import glob
import numpy as np
import os
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from scipy.signal import medfilt
from scipy.stats import linregress, entropy
from scipy.signal import welch

def get_id(nf):
    id = nf.split("_")
    return str(Path(id[0]).name)

def get_kss_score(nf) : 
    kss = nf.split("_")
    return int(kss[3])

def get_pre_post_score(nf) : 
    kss = nf.split("_")
    return int(kss[2]), int(kss[3])

def do_cleaning_ecg(df):
    df["log_time"] = pd.to_datetime(df['log_time'])
    df = df.set_index("log_time")
    df = df.resample("1s").mean().interpolate(method="linear").dropna()
    df = df.reset_index()
    df = df.rename(columns={"Heart_Rate_BPM": "heart_rate_ecg"})
    return df[["log_time", "heart_rate_ecg"]]

def do_cleaning(df):
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
    return df_scaled

def extract_window_features(window):
    hr = window[:,1]
    br = window[:,0]

    delta_hr = np.diff(hr)
    delta_br = np.diff(br)

    feat = {}

    ##################################
    # Heart Rate
    ##################################

    feat["hr_mean"] = np.mean(hr)
    feat["hr_std"] = np.std(hr)
    feat["hr_min"] = np.min(hr)
    feat["hr_max"] = np.max(hr)
    feat["hr_range"] = np.ptp(hr)

    ##################################
    # Breath Rate
    ##################################

    feat["br_mean"] = np.mean(br)
    feat["br_std"] = np.std(br)
    feat["br_min"] = np.min(br)
    feat["br_max"] = np.max(br)
    feat["br_range"] = np.ptp(br)

    ##################################
    # Trend
    ##################################

    x = np.arange(len(hr))

    feat["hr_slope"] = linregress(x, hr).slope
    feat["br_slope"] = linregress(x, br).slope

    ##################################
    # Variability
    ##################################

    feat["hr_sdnn"] = np.std(hr)
    feat["br_sdnn"] = np.std(br)

    feat["hr_rmssd"] = np.sqrt(np.mean(delta_hr**2))
    feat["br_rmssd"] = np.sqrt(np.mean(delta_br**2))

    feat["hr_cv"] = feat["hr_std"] / (feat["hr_mean"] + 1e-6)
    feat["br_cv"] = feat["br_std"] / (feat["br_mean"] + 1e-6)

    ##################################
    # Entropy
    ##################################

    hist,_ = np.histogram(hr,bins=10)

    feat["hr_entropy"] = entropy(hist+1)

    hist,_ = np.histogram(br,bins=10)

    feat["br_entropy"] = entropy(hist+1)

    ##################################
    # Frequency
    ##################################

    # f,p = welch(hr)

    # feat["hr_dom_freq"] = f[np.argmax(p)]
    # feat["hr_energy"] = np.sum(p)

    # f,p = welch(br)

    # feat["br_dom_freq"] = f[np.argmax(p)]
    # feat["br_energy"] = np.sum(p)

    return feat

def do_windowing(df_list, window_size=60, stride=15, min_accuracy=0.85):
    x_sequences = []
    x_tabular = []
    y_labels = []
    ids = []
    exploration=[]
    metadata=[]

    accepted_windows = 0
    rejected_windows = 0

    for session_idx, df in enumerate(df_list):
        df = df.iloc[1:].reset_index()
        # window[:, 0] adalah breath_rate, window[:, 1] adalah heart_rate
        features = df[["breath_rate", "heart_rate", "savgol_br", "savgol_hr"]].values
        
        hr_mmwave = df["hr_mmwave_raw"].values 
        hr_ecg = df["heart_rate_ecg"].values
        
        labels = df["class"].values
        id = df["id"][0]

        total_windows = (len(features) - window_size) // stride + 1

        for i in range(total_windows):
            start = i * stride
            end = start + window_size
            
            window_hr_mm = hr_mmwave[start:end]
            window_hr_ecg = hr_ecg[start:end]
            
            mape = np.abs(window_hr_mm - window_hr_ecg) / (window_hr_ecg + 1e-6)
            similarity = np.clip(1.0 - mape, 0, 1)
            window_accuracy = np.mean(similarity)
            
            if window_accuracy >= min_accuracy:
                accepted_windows += 1
                window_seq = features[start:end]

                # --- BAGIAN YANG DIUBAH ---
                # 1. Ekstrak fitur handcrafted dari window ini
                exp_feat = extract_window_features(window_seq)
                
                # 2. Ambil values-nya saja untuk dijadikan array fitur X_tabular
                combined_features = list(exp_feat.values())
                # --------------------------

                x_sequences.append(window_seq)
                x_tabular.append(combined_features) # Sekarang berisi puluhan fitur keren!
                y_labels.append(labels[end - 1]) 
                ids.append(id)

                exploration.append(exp_feat)

                metadata.append({
                    "subject": id,
                    "session": session_idx,
                    "window": i,
                    "time_sec": start,
                    "label": labels[end-1],
                    "ecg_accuracy": round(window_accuracy, 3)
                })
            else:
                rejected_windows += 1

    print(f"Windowing Selesai | Diterima: {accepted_windows} | Dibuang karena noise: {rejected_windows}")

    exploration = pd.DataFrame(exploration)
    metadata = pd.DataFrame(metadata)
    x_sequence = np.array(x_sequences, dtype=np.float32)
    x_tabular = np.array(x_tabular, dtype=np.float32)
    y_labels = np.array(y_labels, dtype=np.int64)
    return x_sequence, x_tabular, y_labels, ids, exploration, metadata

def extract_savgol(df):
    from scipy.signal import savgol_filter
    df["savgol_hr"] = savgol_filter(df["heart_rate"], window_length=11, polyorder=2)
    df["savgol_br"] = savgol_filter(df["breath_rate"], window_length=11, polyorder=2)
    return df

def scaling(dfs):
    df_all = pd.concat(dfs, ignore_index=True)
    
    cols = ["heart_rate", "breath_rate"] # "delta_hr", "delta_br"
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
        df_scaled = do_cleaning(df)
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

def load_data_with_ecg(ft, file_mmwave="mmwave_ss.csv", file_ecg="mam_sense.csv", classes=3, filter=[]): 
    current_dir = Path(__file__).resolve().parent
    parent = current_dir.parent
    if len(filter) == 0:
        folders = glob.glob(f"{parent}/{ft}/*")
    else:
        folders = [f for f in glob.glob(f"{parent}/{ft}/*")
                    if any(str(keyword) in Path(f).name for keyword in filter)]
    pd_list = []

    for f in folders:
        df_mm = pd.read_csv(f"{f}/{file_mmwave}")
        df_mm_scaled = do_cleaning(df_mm) # Outputnya punya kolom 'timestamp'
        df_ecg = pd.read_csv(f"{f}/{file_ecg}")
        df_ecg_cleaned = do_cleaning_ecg(df_ecg) # Outputnya punya kolom 'log_time'
        
        # Sinkronisasi Data (Merge berdasarkan waktu)
        df_merged = pd.merge(df_mm_scaled, df_ecg_cleaned, left_on="timestamp", right_on="log_time", how="inner")
        df_merged["hr_mmwave_raw"] = df_merged["heart_rate"].copy()
        
        df_merged["id"] = get_id(f)
        df_merged["kss_score"] = get_kss_score(f)
        if classes == 2 :
            df_merged["class"] = [0 if v < 7 else 1 for v in df_merged["kss_score"].values]
        elif classes == 3:
            df_merged["class"] = [0 if v <= 4 else (1 if v <=7 else 2) for v in df_merged["kss_score"].values]
        
        pd_list.append(df_merged)

    pd_list = scaling(pd_list)
    for df in pd_list:
        df = extract_savgol(df)
        
    return pd_list

file_mm = "mmwave_ss.csv"
file_ecg = "mam_sense.csv"
save_path = "data_ready"
os.makedirs(save_path, exist_ok=True)

train_df_list = load_data_with_ecg("data", file_mmwave=file_mm, file_ecg=file_ecg, classes=2)
# train_df_list = load_data_as_df_list_tsdc("data", file, filter=["22009", "22020", "22026", "22038", "22054", "22064", "23015", "23051", "23066", "24059", "24088"]) 221056
# train_df_list = load_data("data", file)
# X_sequence, X_tabular, y_labels, ids = do_windowing(train_df_list)
X_sequence, X_tabular, y_labels, ids, exploration, metadata = do_windowing(
    train_df_list, 
    window_size=60, 
    stride=10, # Boleh diatur ulang, misal 10 agar datanya lebih padat
    min_accuracy=0.85
)


import pickle
tabular_cols = exploration.columns.tolist()
df_tabular = pd.DataFrame(X_tabular, columns=tabular_cols)

# Bungkus data
data_to_save = {
    "X_seq": X_sequence,
    "X_tab": df_tabular, # Sekarang data tabularmu punya nama kolom seperti 'hr_rmssd', 'br_entropy', dll.
    "y": y_labels,
    "id": ids,
    "exploration": exploration,
    "metadata": metadata
}

save_file = f"{save_path}/train_filter.pkl"
with open(save_file, "wb") as f:
    pickle.dump(data_to_save, f)

np.savez(f"{save_path}/train_filter.npz", X_seq=X_sequence, X_tab=X_tabular, y=y_labels, id=ids, exploration=exploration, metadata=metadata)