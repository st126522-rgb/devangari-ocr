import torch
import torch.nn as nn

class LSTMDecoder(nn.Module):
    """Simple LSTM decoder stub for sequence modeling after CRNN features."""
    def __init__(self, input_size, hidden_size, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

    def forward(self, x):
        # x: [batch, seq_len, features]
        out, _ = self.lstm(x)
        return out
