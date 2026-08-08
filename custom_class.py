import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from tsai.all import PatchTST
import torch.nn as nn

class TimeSeriesDataset(Dataset):
    def __init__(self, data, 
        stride=1,
        gap=0,
        i_chunk_len=30,
        o_chunk_len=10,
        x_vals=["breath_rate", "heart_rate"], 
        y_vals=["kss_score"]):
        self.features = data[x_vals].values
        self.labels = data[y_vals].values

        # self.labels = data["kss_score"].values # use this for classification, not regression
        # bins = [0, 3.1, 6.1, 9.1] 
        # labels = ['Class A', 'Class B', 'Class C']
        # self.labels = pd.cut(self.labels, bins=bins, labels=labels, include_lowest=True)

        self.stride = stride # offset between window, 
        self.gap = gap # offset between context and prediction
        self.context_length = i_chunk_len
        self.prediction_length = o_chunk_len
        self.total_window_needed = self.context_length + self.gap + self.prediction_length

    def __len__(self):
        # total sliding windows that can be made from the data
        if len(self.features) < self.total_window_needed:
            return 0
        return (len(self.features) - self.total_window_needed) // self.stride + 1

    def __getitem__(self, idx):
        actual_idx = idx * self.stride
        x = self.features[actual_idx : actual_idx + self.context_length]
        y_classification = self.labels[actual_idx + self.context_length - 1]

        return (
            torch.tensor(x, dtype=torch.float32), # [batch, seq, channel]
            torch.tensor(y_classification, dtype=torch.float32), #[batch, 1]
        )

    def get_all_labels(self):
        extracted_labels = []
        for idx in range(len(self)):
            actual_idx = idx * self.stride
            y = self.labels[actual_idx + self.context_length - 1]
            extracted_labels.append(y[0] if isinstance(y, (np.ndarray, list)) else y)
            
        return np.array(extracted_labels)

class TSDforClassification(Dataset):
    def __init__(self, data, 
        stride=1,
        gap=0,
        i_chunk_len=30,
        o_chunk_len=10,
        x_vals=["breath_rate", "heart_rate"], 
        y_vals=["kss_score"],
        classes=2
        ):

        self.features = data[x_vals].values
        self.labels = data[y_vals].values

        self.stride = stride # offset between the beginning of each window, 
        self.gap = gap # offset between context and prediction
        self.context_length = i_chunk_len
        self.prediction_length = o_chunk_len
        self.total_window_needed = self.context_length + self.gap + self.prediction_length

    def __len__(self):
        # total sliding windows that can be made from the data
        if len(self.features) < self.total_window_needed:
            return 0
        return (len(self.features) - self.total_window_needed) // self.stride + 1

    def __getitem__(self, idx):
        actual_idx = idx * self.stride
        x = self.features[actual_idx : actual_idx + self.context_length]
        y = self.labels[actual_idx + self.context_length - 1]

        return (
            torch.tensor(x, dtype=torch.float32), # [batch, seq, channel]
            torch.tensor(y, dtype=torch.long), #[batch, 1]
        )
    
    def get_all_labels(self): #for k-folds
        extracted_labels = []
        for idx in range(len(self)):
            actual_idx = idx * self.stride
            y = self.labels[actual_idx + self.context_length - 1]
            extracted_labels.append(y[0] if isinstance(y, (np.ndarray, list)) else y)
            
        return np.array(extracted_labels)

class LOGODataset(Dataset):
    def __init__(self, X_sec, X_tab, y):
        self.X_sec = torch.tensor(X_sec, dtype=torch.float32)
        self.X_tab = torch.tensor(X_tab, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        seq = self.X_sec[idx]
        # seq = seq.permute(1, 0)
        return seq, self.X_tab[idx], self.y[idx]

class PatchTSTClassification(nn.Module):
    def __init__(self, 
                c_in = 2,
                c_out = 3, # 3 = multiclass classification
                seq_len = 120,
                n_layers = 2,
                dropout = 0.1,
                d_model = 512,
                n_heads = 8,
                patch_len = 16,):
        super().__init__()

        self.patchtst = PatchTST(
            c_in = c_in,
            c_out = c_out,
            seq_len = seq_len,
            n_layers = n_layers,
            dropout = dropout,
            d_model = d_model,
            n_heads = n_heads,
            patch_len = 16,
            )
        
        self.classifier = nn.Linear(c_in * seq_len, c_out)
        
    def forward(self, x):
        out = self.patchtst(x) 
        out = out.view(out.size(0), -1) 
        out = self.classifier(out)
        return out

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        # Inisialisasi weight untuk attention
        self.W = nn.Linear(hidden_dim, 1, bias=True)

    def forward(self, x):
        # x memiliki dimensi: (batch_size, seq_length, hidden_dim)
        e = torch.tanh(self.W(x)) 
        a = F.softmax(e, dim=1)
        output = x * a
        return torch.sum(output, dim=1) # Hasil akhir: (batch_size, hidden_dim)

class Hybrid_CNN_BiLSTM_Attention(nn.Module):
    def __init__(self, c_in, tab_in, c_out=3, d_model=64, dropout=0.3):
        super(Hybrid_CNN_BiLSTM_Attention, self).__init__()
        
        # === Blok CNN ===
        # PyTorch Conv1D menggunakan format (batch, channel, seq_len)
        # Jaringan ini menggunakan 64 filter sesuai referensi
        self.conv1 = nn.Conv1d(in_channels=c_in, out_channels=d_model, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.dropout_cnn = nn.Dropout(dropout)

        # === Blok BiLSTM ===
        # Terdiri dari 3 lapis BiLSTM dengan 64 unit dan dropout 0.3
        self.lstm1 = nn.LSTM(input_size=d_model, hidden_size=d_model, batch_first=True, bidirectional=True)
        self.dropout_lstm1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(input_size=d_model*2, hidden_size=d_model, batch_first=True, bidirectional=True)
        self.dropout_lstm2 = nn.Dropout(dropout)
        self.lstm3 = nn.LSTM(input_size=d_model*2, hidden_size=d_model, batch_first=True, bidirectional=True)
        self.dropout_lstm3 = nn.Dropout(dropout)

        # === Mekanisme Attention ===
        self.attention = AttentionLayer(hidden_dim=d_model*2)

        # === Lapisan Ekstraksi Fitur Temporal ===
        # Diekstrak menggunakan Dense/Linear layer berukuran 64
        self.fc_temporal_features = nn.Linear(d_model*2, d_model)
        
        # === Lapisan Klasifikasi Akhir (Fusion) ===
        # Menggabungkan 64 fitur temporal dengan jumlah fitur tabular (tab_in)
        self.classifier = nn.Linear(d_model + tab_in, c_out)

    def forward(self, x_seq, x_tab):
        # 1. Pemrosesan Data Sekuensial (X_sec)
        # Transpose dari (batch, seq_len, channels) ke (batch, channels, seq_len) untuk Conv1D
        x = x_seq.transpose(1, 2)
        
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.dropout_cnn(x)

        # Transpose kembali untuk input LSTM: (batch, seq_len_baru, channels)
        x = x.transpose(1, 2)

        x, _ = self.lstm1(x)
        x = self.dropout_lstm1(x)
        x, _ = self.lstm2(x)
        x = self.dropout_lstm2(x)
        x, _ = self.lstm3(x)
        x = self.dropout_lstm3(x)

        x = self.attention(x)
        temporal_features = F.relu(self.fc_temporal_features(x)) #

        # 2. Fusi dengan Data Tabular (X_tab)
        # Konkatenasi fitur sekuensial yang sudah diekstrak dengan fitur tabular mentah
        fused_features = torch.cat((temporal_features, x_tab), dim=1)

        # 3. Klasifikasi
        logits = self.classifier(fused_features)
        
        # Kembalikan logits mentah karena kamu menggunakan nn.CrossEntropyLoss
        return logits

class Light_Hybrid_Fusion(nn.Module):
    # d_model diturunkan default-nya ke 32, dropout dinaikkan ke 0.5
    def __init__(self, c_in, tab_in, c_out=3, d_model=32, dropout=0.5):
        super(Light_Hybrid_Fusion, self).__init__()
        
        # === Blok CNN (Lebih Ramping + BatchNorm) ===
        self.conv1 = nn.Conv1d(in_channels=c_in, out_channels=d_model, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(d_model)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        # Hanya gunakan 2 lapis CNN, atau bahkan bisa dikurangi jadi 1 jika masih overfit
        self.conv2 = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(d_model)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.dropout_cnn = nn.Dropout(dropout)

        # === Blok BiLSTM (Dipangkas jadi 1 Lapis saja) ===
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout_lstm = nn.Dropout(dropout)

        # === Mekanisme Attention ===
        self.attention = AttentionLayer(hidden_dim=d_model*2)

        # === Lapisan Ekstraksi Fitur Temporal ===
        self.fc_temporal_features = nn.Linear(d_model*2, d_model)
        
        # === Lapisan Klasifikasi Akhir ===
        self.classifier = nn.Linear(d_model + tab_in, c_out)

    def forward(self, x_seq, x_tab):
        # 1. Pemrosesan Sekuensial
        x = x_seq.transpose(1, 2)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.dropout_cnn(x)

        x = x.transpose(1, 2)

        x, _ = self.lstm(x)
        x = self.dropout_lstm(x)

        x = self.attention(x)
        temporal_features = F.relu(self.fc_temporal_features(x))

        # 2. Fusi dengan Tabular
        fused_features = torch.cat((temporal_features, x_tab), dim=1)

        # 3. Klasifikasi
        logits = self.classifier(fused_features)
        return logits