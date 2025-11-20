import torch
import torch.nn as nn
from model.backbones import get_backbone


class BidirectionalLSTM(nn.Module):
    """BiLSTM layer for CRNN."""
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.0):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=1,
            bidirectional=True,
            dropout=dropout_rate if hidden_size > 0 else 0
        )
        self.embedding = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        recurrent, _ = self.rnn(x)
        output = self.embedding(recurrent)
        return output


class OCRModelV2(nn.Module):
    """CRNN + BiLSTM + CTC Head with multiple backbone options."""
    def __init__(
        self,
        num_classes,
        backbone_name="resnet18",
        img_channels=1,
        hidden_size=256,
        dropout_rate=0.0
    ):
        super().__init__()
        
        self.backbone = get_backbone(backbone_name, img_channels=img_channels, dropout_rate=dropout_rate)
        
        self.rnn = nn.Sequential(
            BidirectionalLSTM(self.backbone.output_channels, hidden_size, hidden_size, dropout_rate),
            BidirectionalLSTM(hidden_size, hidden_size, num_classes, dropout_rate)
        )
        
        self.ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)

    def forward(self, x):
        features = self.backbone(x)
        output = self.rnn(features)
        return output

    def compute_ctc_loss(self, preds, targets, pred_lengths, target_lengths):
        preds_log = preds.log_softmax(2)
        return self.ctc_loss(preds_log, targets, pred_lengths, target_lengths)
