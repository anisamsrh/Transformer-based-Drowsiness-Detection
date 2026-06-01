import numpy as np
import torch
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, config, data, x_vals=["breath_rate", "heart_rate"], y_vals=["kss_score"]):
        self.features = data[x_vals].values
        self.labels = data[y_vals].values

        self.stride = config.WINDOW_STRIDE # offset between window, 
        self.gap = config.GAP # offset between context and prediction
        self.context_length = config.INPUT_CHUNK_LEN
        self.prediction_length = config.OUTPUT_CHUNK_LEN
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
