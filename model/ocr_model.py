import torch
import torch.nn as nn

class CRNNFeatureExtractor(nn.Module):
    """CNN backbone for CRNN."""
    def __init__(self, img_channels=1):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(img_channels, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),

            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),

            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),

            nn.Conv2d(512, 512, 2, stride=1, padding=0),
            nn.ReLU()
        )

    def forward(self, x):
        conv_output = self.cnn(x)
        b, c, h, w = conv_output.size()
        assert h == 1 or h == 2
        conv_output = conv_output.mean(2)
        conv_output = conv_output.permute(2, 0, 1)
        return conv_output


class BidirectionalLSTM(nn.Module):
    """BiLSTM layer for CRNN."""
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=1,
            bidirectional=True
        )
        self.embedding = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        recurrent, _ = self.rnn(x)
        output = self.embedding(recurrent)
        return output


class OCRModel(nn.Module):
    """CRNN + BiLSTM + CTC Head."""
    def __init__(self, num_classes, img_channels=1, hidden_size=256):
        super().__init__()
        self.cnn = CRNNFeatureExtractor(img_channels)
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, num_classes)
        )
        self.ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)

    def forward(self, x):
        features = self.cnn(x)
        output = self.rnn(features)
        return output

    def compute_ctc_loss(self, preds, targets, pred_lengths, target_lengths):
        preds_log = preds.log_softmax(2)
        return self.ctc_loss(preds_log, targets, pred_lengths, target_lengths)
