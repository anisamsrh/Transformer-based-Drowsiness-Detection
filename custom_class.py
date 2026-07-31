import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
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

class PreprocessedDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.X_seq = data['X_seq']
        self.X_tab = data['X_tab']
        self.y = data['y']
        self.id = data['id'] if 'id' in data else None

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        seq = torch.tensor(self.X_seq[idx], dtype=torch.float32)
        tab = torch.tensor(self.X_tab[idx], dtype=torch.float32)
        label = torch.tensor(self.y[idx], dtype=torch.long)
        seq = seq.permute(1, 0) 
        if self.id is not None:
            return seq, tab, label, self.id[idx]
        else:
            return seq, tab, label

class LOGODataset(Dataset):
    def __init__(self, X_sec, X_tab, y):
        self.X_sec = torch.tensor(X_sec, dtype=torch.float32)
        self.X_tab = torch.tensor(X_tab, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        seq = self.X_sec[idx]
        seq = seq.permute(1, 0)
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