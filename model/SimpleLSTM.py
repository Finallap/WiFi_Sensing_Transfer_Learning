import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleLSTMNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bidirectional = True
        self.num_layers = 2
        self.lstm = nn.LSTM(input_size=config['input_feature'], hidden_size=config['hidden_size'],
                            num_layers=self.num_layers, bidirectional=self.bidirectional,
                            dropout=0.5, batch_first=True)
        if self.bidirectional:
            self.fc = nn.Linear(config['hidden_size'] * 2, config['n_class'])
        else:
            self.fc = nn.Linear(config['hidden_size'], config['n_class'])

    def forward(self, x):
        states, hidden = self.lstm(x)
        x = self.fc(states[:, -1, :])
        return F.log_softmax(x, dim=1)
