import torch
import torch.nn as nn
from torch.utils.data import Dataset
import math
import numpy as np

class TimeSeriesDataset(Dataset):
    def __init__(self, config, data):
        self.features = data[["data"]].values
        self.labels = data["label"].values

        self.seq_length = config.SEQ_LENGTH # len of 1 minute data point
        self.stride = config.WINDOW_STRIDE # offset between window, 
        self.gap = config.GAP # offset between context and prediction
        self.context_length = config.CONTEXT_TIME * self.seq_length # 5 minutes
        self.prediction_length = config.PREDICTION_TIME * self.seq_length # 1 minutes
        self.total_window_needed = self.context_length + self.gap + self.prediction_length # 360 data point

    def __len__(self):
        # total sliding windows that can be made from the data
        if len(self.features) < self.total_window_needed:
            return 0
        return (len(self.features) - self.total_window_needed) // self.stride + 1

    def __getitem__(self, idx):
        # PREP : Sliding Window / context length + prediction length
        # Context Length : Prediction Length = 5 : 1
        actual_idx = idx * self.stride
        x = self.features[actual_idx : actual_idx + self.context_length]
        y_classification = self.labels[actual_idx + self.context_length - 1]
        start_pred_idx = actual_idx + self.context_length + self.gap
        end_pred_idx = start_pred_idx + self.prediction_length
        y_forecasting = self.features[start_pred_idx : end_pred_idx]

        # TODO : add label for future classification (use Modus/Max)
        future_labels = self.labels[start_pred_idx : end_pred_idx]
        y_kss_future = np.max(future_labels)

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y_classification, dtype=torch.float32),
            torch.tensor(y_forecasting, dtype=torch.float32),
            torch.tensor(y_kss_future, dtype=torch.float32)
        )

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1): # max_len == seq_len
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # shape = [max_length, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) # pos * e**(-2i.ln(10000)/d_model)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # shape = [1, max_len, d_model] so it suitable for batching [batch_size, seq_len, channels]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :] # if input [Batch, Seq, Channels], use size(1) (size from seq_len)
        return self.dropout(x)

class TimeSeriesTransformer(nn.Module):
    def __init__(self, config, input_dim, d_model, nhead, num_layers, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()

        # Input Layer
        self.input_projection = nn.Linear(input_dim, d_model) #input_dim = channels
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead,
            dropout=dropout,
            batch_first=True # cuz format [batch_size, seq_len, channels]
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )

        # Output Layer
        self.kss_head = nn.Linear(d_model, 1) # approaching kss score using regression
        self.forecasting_head = nn.Linear(d_model, config.SEQ_LENGTH * config.PREDICTION_TIME)
        self.kss_f_head = nn.Linear(d_model, 1)

    def mask_future(self, seq_length):
        mask = torch.triu(torch.ones(seq_length, seq_length) * float('-inf'), diagonal=1)
        return mask

    def forward(self, x):
        x = self.input_projection(x) * math.sqrt(self.input_projection.out_features) # [batch_size, seq_length, d_model] 
        x = self.positional_encoding(x)

        mask = self.mask_future(x.size(1)).to(x.device) # change it to use is_causal=True
        encoded = self.transformer_encoder(x) # [seq_length, batch_size, d_model] # without mask to get bidirectional context
        
        # cls_token = encoded[:, -1, :] # use the last token for classification (if mask)
        cls_token = encoded.mean(dim=1)
        kss_output = self.kss_head(cls_token)
        forecasting_output = self.forecasting_head(cls_token)
        kss_f_output = self.kss_f_head(cls_token)

        return kss_output, forecasting_output.unsqueeze(2), kss_f_output

