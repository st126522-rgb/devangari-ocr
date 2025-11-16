import torch.nn as nn

class BidirectionalLSTM(nn.Module):
    """
    Standard BiLSTM layer used for CRNN.
    """

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
