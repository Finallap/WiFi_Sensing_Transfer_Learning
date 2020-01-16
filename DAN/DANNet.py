import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import DAN.mmd as mmd
import torch


class DANNet(nn.Module):
    def __init__(self, config):
        super(DANNet, self).__init__()
        self.sharedNet = SharedNet(config)
        self.cls_fc = nn.Linear(128, config['n_class'])

    def forward(self, source, target):
        loss = 0
        source = self.sharedNet(source)
        if self.training == True:
            target = self.sharedNet(target)
            # loss += mmd.mmd_rbf_accelerate(source, target)
            loss += mmd.mmd_rbf_noaccelerate(source, target)

        source = self.cls_fc(source)
        # target = self.cls_fc(target)

        return source, loss


class SharedNet(nn.Module):
    def __init__(self, config):
        super(SharedNet, self).__init__()
        self.config = config

        if self.config['model_type'] == 'lstm':
            # lstm
            self.bidirectional = True
            self.num_layers = 2
            self.base_network = nn.LSTM(input_size=config['input_feature'], hidden_size=config['hidden_size'],
                                        num_layers=self.num_layers, bidirectional=self.bidirectional,
                                        dropout=0.5, batch_first=True)
            if self.bidirectional:
                self.bottleneck = nn.Linear(config['hidden_size'] * 2, 128)
            else:
                self.bottleneck = nn.Linear(config['hidden_size'], 128)
        else:
            # conv
            self.base_network = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.ReLU(),
            )
            self.bottleneck = nn.Linear(config['batch_size'] * 3680, 128)

    def forward(self, input):
        # lstm
        if self.config['model_type'] == 'lstm':
            x, _ = self.base_network(input)
            output = self.bottleneck(x[:, -1, :])
        else:
            # conv
            x = self.base_network(input)
            x = x.view(self.config['batch_size'], -1)
            output = self.bottleneck(x)
        return output
