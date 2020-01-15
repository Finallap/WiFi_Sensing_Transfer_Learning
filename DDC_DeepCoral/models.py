import torch.nn as nn
import torchvision
from DDC_DeepCoral.Coral import CORAL
import DDC_DeepCoral.mmd as mmd


class Transfer_Net(nn.Module):
    def __init__(self, config, transfer_loss='mmd', use_bottleneck=True, bottleneck_width=256, width=1024):
        super(Transfer_Net, self).__init__()
        self.config = config

        if self.config['model_type'] == 'lstm':
            # lstm
            self.bidirectional = True
            self.num_layers = 2
            self.base_network = nn.LSTM(input_size=config['input_feature'], hidden_size=config['hidden_size'],
                                        num_layers=self.num_layers, bidirectional=self.bidirectional,
                                        dropout=0.5, batch_first=True)
            if self.bidirectional:
                self.base_network_output_size = config['hidden_size'] * 2
            else:
                self.base_network_output_size = config['hidden_size']
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
            self.base_network_output_size = config['batch_size'] * 3680

        self.use_bottleneck = use_bottleneck
        self.transfer_loss = transfer_loss
        bottleneck_list = [nn.Linear(self.base_network_output_size, 256), nn.BatchNorm1d(256), nn.ReLU(),
                           nn.Dropout(0.5)]
        self.bottleneck_layer = nn.Sequential(*bottleneck_list)
        classifier_layer_list = [nn.Linear(self.base_network_output_size, 256), nn.ReLU(), nn.Dropout(0.5),
                                 nn.Linear(256, config['n_class'])]
        self.classifier_layer = nn.Sequential(*classifier_layer_list)

        self.bottleneck_layer[0].weight.data.normal_(0, 0.005)
        self.bottleneck_layer[0].bias.data.fill_(0.1)
        for i in range(2):
            self.classifier_layer[i * 3].weight.data.normal_(0, 0.01)
            self.classifier_layer[i * 3].bias.data.fill_(0.0)

    def forward(self, source, target):
        # lstm
        if self.config['model_type'] == 'lstm':
            source, _ = self.base_network(source)
            target, _ = self.base_network(target)
            source_clf = self.classifier_layer(source[:, -1, :])
            if self.use_bottleneck:
                source = self.bottleneck_layer(source[:, -1, :])
                target = self.bottleneck_layer(target[:, -1, :])
        else:
            # conv
            source = self.base_network(source)
            source = source.view(self.config['batch_size'], -1)
            target = self.base_network(target)
            target = target.view(self.config['batch_size'], -1)
            source_clf = self.classifier_layer(source)
            if self.use_bottleneck:
                source = self.bottleneck_layer(source)
                target = self.bottleneck_layer(target)

        transfer_loss = self.adapt_loss(source, target, self.transfer_loss)
        return source_clf, transfer_loss

    def predict(self, x):
        if self.config['model_type']=='lstm':
            features, _ = self.base_network(x)
            clf = self.classifier_layer(features[:, -1, :])
        else:
            features = self.base_network(x)
            features = features.view(self.config['batch_size'], -1)
            clf = self.classifier_layer(features)
        return clf

    def adapt_loss(self, X, Y, adapt_loss):
        """Compute adaptation loss, currently we support mmd and coral

        Arguments:
            X {tensor} -- source matrix
            Y {tensor} -- target matrix
            adapt_loss {string} -- loss type, 'mmd' or 'coral'. You can add your own loss

        Returns:
            [tensor] -- adaptation loss tensor
        """
        if adapt_loss == 'mmd':
            mmd_loss = mmd.MMD_loss()
            loss = mmd_loss(X, Y)
        elif adapt_loss == 'coral':
            loss = CORAL(X, Y)
        else:
            loss = 0
        return loss
