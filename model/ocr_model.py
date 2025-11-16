import torch
import torch.nn as nn

from model.crnn import CRNNFeatureExtractor
from model.lstm import BidirectionalLSTM

class OCRModel(nn.Module):
    """
    CRNN + BiLSTM + CTC Head.
    """

    def __init__(self, num_classes, img_channels=1, hidden_size=256):
        super().__init__()

        self.cnn = CRNNFeatureExtractor(img_channels)

        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, num_classes)
        )

        self.ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)

    def forward(self, x):
        # x shape = (B, C, H, W)
        features = self.cnn(x)          # (T, B, 512)
        output = self.rnn(features)     # (T, B, num_classes)
        return output

    def compute_ctc_loss(self, preds, targets, pred_lengths, target_lengths):
        """
        preds: (T, B, num_classes)
        """
        preds_log = preds.log_softmax(2)
        return self.ctc_loss(preds_log, targets, pred_lengths, target_lengths)
