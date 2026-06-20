import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

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
        if classes == 2 :
            self.labels = [0 if v < 7 else 1 for v in data[y_vals].values]
        elif classes == 3:
            self.labels = [0 if v <= 4 else (1 if v <=7 else 2) for v in data[y_vals].values]
        else :
            raise Exception("classes should either 2 or 3")

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
        y = self.labels[actual_idx + self.context_length - 1]

        return (
            torch.tensor(x, dtype=torch.float32), # [batch, seq, channel]
            torch.tensor(y, dtype=torch.long), #[batch, 1]
        )
