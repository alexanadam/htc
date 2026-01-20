import torch.nn as nn
import torch.nn.functional as F


class MLPModule(nn.Module):
    def __init__(self, input, hidden, output, dropout=0.3):
        super(MLPModule, self).__init__()
        self.fc1 = nn.Linear(input, hidden)
        self.fc2 = nn.Linear(hidden, output)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
