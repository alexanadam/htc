import torch.nn as nn


class LSTMModule(nn.Module):
    def __init__(self, input, hidden, output, dropout=0.3):
        super(LSTMModule, self).__init__()
        self.lstm = nn.LSTM(input, hidden, batch_first=True, dropout=dropout)

        self.fc = nn.Linear(hidden, output)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)

        last_hidden_state = h_n[-1]

        output = self.fc(self.dropout(last_hidden_state))

        return output
