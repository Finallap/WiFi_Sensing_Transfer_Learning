import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import torch
from torch.autograd import Function
import torch.nn.functional as F


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class DAANNet(nn.Module):

    def __init__(self, config):
        super(DAANNet, self).__init__()
        self.config = config
        # conv
        self.sharedNet = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(8, 16, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU(),
        )
        self.bottleneck = nn.Linear(config['batch_size'] * 810, 128)

        # lstm
        # self.bidirectional = True
        # self.num_layers = 2
        # self.sharedNet = nn.LSTM(input_size=config['input_feature'], hidden_size=config['hidden_size'],
        #                           num_layers=self.num_layers, bidirectional=self.bidirectional,
        #                           dropout=0.5, batch_first=True)
        # self.bottleneck = nn.Linear(config['hidden_size'], 128)
        # if self.bidirectional:
        #     self.bottleneck = nn.Linear(config['hidden_size'] * 2, 128)
        # else:
        #     self.bottleneck = nn.Linear(config['hidden_size'], 128)

        self.source_fc = nn.Linear(128, config['n_class'])
        self.softmax = nn.Softmax(dim=1)
        self.classes = config['n_class']

        # global domain discriminator
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('fc1', nn.Linear(128, 256))
        self.domain_classifier.add_module('relu1', nn.ReLU(True))
        self.domain_classifier.add_module('dpt1', nn.Dropout())
        self.domain_classifier.add_module('fc2', nn.Linear(256, 256))
        self.domain_classifier.add_module('relu2', nn.ReLU(True))
        self.domain_classifier.add_module('dpt2', nn.Dropout())
        self.domain_classifier.add_module('fc3', nn.Linear(256, 2))

        # local domain discriminator
        self.dcis = nn.Sequential()
        self.dci = {}
        for i in range(config['n_class']):
            self.dci[i] = nn.Sequential()
            self.dci[i].add_module('fc1', nn.Linear(128, 256))
            self.dci[i].add_module('relu1', nn.ReLU(True))
            self.dci[i].add_module('dpt1', nn.Dropout())
            self.dci[i].add_module('fc2', nn.Linear(256, 256))
            self.dci[i].add_module('relu2', nn.ReLU(True))
            self.dci[i].add_module('dpt2', nn.Dropout())
            self.dci[i].add_module('fc3', nn.Linear(256, 2))
            self.dcis.add_module('dci_' + str(i), self.dci[i])

    def forward(self, source, target, s_label, DEV, alpha=0.0):
        # conv
        source_share = self.sharedNet(source)
        source_share = source_share.view(self.config['batch_size'], -1)
        source_share = F.relu(self.bottleneck(source_share))
        target = self.sharedNet(target)
        target = target.view(self.config['batch_size'], -1)
        target = F.relu(self.bottleneck(target))

        # lstm
        # source_share, _  = self.sharedNet(source)
        # source_share = self.bottleneck(source_share[:, -1, :])
        # target, _ = self.sharedNet(target)
        # target = self.bottleneck(target[:, -1, :])

        source = self.source_fc(source_share)
        p_source = self.softmax(source)
        t_label = self.source_fc(target)
        p_target = self.softmax(t_label)
        t_label = t_label.data.max(1)[1]
        s_out = []
        t_out = []
        if self.training == True:
            # RevGrad
            s_reverse_feature = ReverseLayerF.apply(source_share, alpha)
            t_reverse_feature = ReverseLayerF.apply(target, alpha)
            s_domain_output = self.domain_classifier(s_reverse_feature)
            t_domain_output = self.domain_classifier(t_reverse_feature)

            # p*feature-> classifier_i ->loss_i
            for i in range(self.classes):
                ps = p_source[:, i].reshape((target.shape[0], 1))
                fs = ps * s_reverse_feature
                pt = p_target[:, i].reshape((target.shape[0], 1))
                ft = pt * t_reverse_feature
                outsi = self.dcis[i](fs)
                s_out.append(outsi)
                outti = self.dcis[i](ft)
                t_out.append(outti)
        else:
            s_domain_output = 0
            t_domain_output = 0
            s_out = [0] * self.classes
            t_out = [0] * self.classes
        return source, s_domain_output, t_domain_output, s_out, t_out
