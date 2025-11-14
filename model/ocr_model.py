import torch
import torch.nn as nn
from .crnn import CRNN
from .lstm import LSTMDecoder

class OCRModel(nn.Module):
    def __init__(self, vocab_size, hidden_size=256):
        super().__init__()
        self.crnn = CRNN(in_channels=1, out_channels=hidden_size)
        self.decoder = LSTMDecoder(input_size=hidden_size, hidden_size=hidden_size//2)
        self.classifier = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        features = self.crnn(x)
        # collapse spatial dims into sequence: placeholder
        b, c, h, w = features.size()
        seq = features.permute(0,3,1,2).contiguous().view(b, w, -1)
        dec = self.decoder(seq)
        logits = self.classifier(dec)
        return logits
