import torch
import torch.nn as nn
import DAN.mmd as mmd

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
            loss += mmd.mmd_rbf_accelerate(source, target)

        source = self.cls_fc(source)
        # target = self.cls_fc(target)

        return source, loss

class SharedNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        pretrained_model = torch.load(config['model_save_path'])
        self.base_network = pretrained_model.base_network
        self.bottleneck = pretrained_model.bottleneck

    def forward(self, input):
        # lstm
        if self.config['model_type'] == 'lstm':
            x, _ = self.base_network(input)
            output = self.bottleneck(x[:, -1, :])
        elif self.config['model_type'] == 'conv1d':
            # conv1d
            x = self.base_network(input)
            output = x.view(self.config['batch_size'], -1)
        else:
            # conv
            x = self.base_network(input)
            x = x.view(self.config['batch_size'], -1)
            output = self.bottleneck(x)
        return output

    # def resnet50(self,config):
    #     pretrained_model = torch.load(config['model_save_path'])
    #     self.base_network = pretrained_model.base_network
    #     self.bottleneck = pretrained_model.bottleneck
    #     return model