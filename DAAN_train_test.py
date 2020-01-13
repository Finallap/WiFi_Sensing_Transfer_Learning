import torch
import time
import math
import numpy as np
import torch.optim as optim
import torch.nn as nn
from config import CONFIG
from torch.utils.tensorboard import SummaryWriter
import tqdm
import torch.nn.functional as F
import model.DAAN as DAAN
import data_loader_preprocessing.load_csi_data as load_csi_data

arg = CONFIG
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def print_learning_rate(optimizer):
    for p in optimizer.param_groups:
        outputs = ''
        for k, v in p.items():
            if k is 'params':
                outputs += (k + ': ' + str(v[0].shape).ljust(30) + ' ')
            else:
                outputs += (k + ': ' + str(v).ljust(10) + ' ')
        print(outputs)


def train(epoch, model, source_loader, target_loader):
    # total_progress_bar = tqdm.tqdm(desc='Train iter', total=args.epochs)
    LEARNING_RATE = CONFIG['lr'] / math.pow((1 + 10 * (epoch - 1) / CONFIG['epochs']), 0.75)
    if CONFIG['diff_lr']:
        # optimizer = torch.optim.SGD([
        #     {'params': model.sharedNet.parameters()},
        #     {'params': model.bottleneck.parameters()},
        #     {'params': model.domain_classifier.parameters()},
        #     {'params': model.dcis.parameters()},
        #     {'params': model.source_fc.parameters(), 'lr': LEARNING_RATE},
        # ], lr=LEARNING_RATE / 10, momentum=CONFIG['momentum'], weight_decay=CONFIG['l2_decay'])
        optimizer = torch.optim.RMSprop([
            {'params': model.sharedNet.parameters()},
            {'params': model.bottleneck.parameters()},
            {'params': model.domain_classifier.parameters()},
            {'params': model.dcis.parameters()},
            {'params': model.source_fc.parameters(), 'lr': LEARNING_RATE},
        ], lr=LEARNING_RATE / 10,  alpha=0.9, weight_decay=CONFIG['l2_decay'])
    else:
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=CONFIG['momentum'], weight_decay=CONFIG['l2_decay'])

    # print_learning_rate(optimizer)

    global D_M, D_C, MU
    model.train()
    len_dataloader = len(source_loader)
    DEV = DEVICE

    d_m = 0
    d_c = 0
    ''' update mu per epoch '''
    if D_M == 0 and D_C == 0 and MU == 0:
        MU = 0.5
    else:
        D_M = D_M / len_dataloader
        D_C = D_C / len_dataloader
        MU = 1 - D_M / (D_M + D_C)

    for batch_idx, (source_data, source_label) in tqdm.tqdm(enumerate(source_loader),
                                                            total=len_dataloader,
                                                            desc='Train epoch = {}'.format(epoch), ncols=80,
                                                            leave=False):
        p = float(batch_idx + 1 + epoch * len_dataloader) / CONFIG['epochs'] / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        optimizer.zero_grad()
        source_data, source_label = source_data.to(DEVICE), source_label.to(DEVICE)
        for target_data, target_label in target_loader:
            target_data, target_label = target_data.to(DEVICE), target_label.to(DEVICE)
            break
        out = model(source_data, target_data, source_label, DEV, alpha)
        s_output, s_domain_output, t_domain_output = out[0], out[1], out[2]
        s_out = out[3]
        t_out = out[4]

        # global loss
        sdomain_label = torch.zeros(CONFIG['batch_size']).long().to(DEV)
        err_s_domain = F.nll_loss(F.log_softmax(s_domain_output, dim=1), sdomain_label)
        tdomain_label = torch.ones(CONFIG['batch_size']).long().to(DEV)
        err_t_domain = F.nll_loss(F.log_softmax(t_domain_output, dim=1), tdomain_label)

        # local loss
        loss_s = 0.0
        loss_t = 0.0
        tmpd_c = 0
        for i in range(CONFIG['n_class']):
            loss_si = F.nll_loss(F.log_softmax(s_out[i], dim=1), sdomain_label)
            loss_ti = F.nll_loss(F.log_softmax(t_out[i], dim=1), tdomain_label)
            loss_s += loss_si
            loss_t += loss_ti
            tmpd_c += 2 * (1 - 2 * (loss_si + loss_ti))
        tmpd_c /= CONFIG['n_class']

        d_c = d_c + tmpd_c.cpu().item()

        global_loss = 0.05 * (err_s_domain + err_t_domain)
        local_loss = 0.01 * (loss_s + loss_t)

        d_m = d_m + 2 * (1 - 2 * global_loss.cpu().item())

        join_loss = (1 - MU) * global_loss + MU * local_loss
        soft_loss = F.nll_loss(F.log_softmax(s_output, dim=1), source_label)
        if CONFIG['gamma'] == 1:
            gamma = 2 / (1 + math.exp(-10 * (epoch) / CONFIG['epochs'])) - 1
        if CONFIG['gamma'] == 2:
            gamma = epoch / CONFIG['epochs']
        loss = soft_loss - join_loss  # -
        loss.backward()
        optimizer.step()

        if batch_idx % CONFIG['log_interval'] == 0:
            print(
                '\nLoss: {:.6f},  label_Loss: {:.6f},  join_Loss: {:.6f}, global_Loss:{:.4f}, local_Loss:{:.4f}'.format(
                    loss.item(), soft_loss.item(), join_loss.item(), global_loss.item(), local_loss.item()))
        # total_progress_bar.update(1)
    D_M = np.copy(d_m).item()
    D_C = np.copy(d_c).item()


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            out = model(data, data, target, DEVICE)
            s_output = out[0]
            test_loss += F.nll_loss(F.log_softmax(s_output, dim=1), target,
                                    size_average=False).item()  # sum up batch loss
            pred = s_output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    return correct

def load_data():
    source_train_loader, source_test_loader, source_csi_train_label = load_csi_data.load_data(CONFIG['source_path'],CONFIG)
    target_train_loader, target_test_loader, target_csi_train_label = load_csi_data.load_data(CONFIG['target_path'],
                                                                                              CONFIG)
    return source_train_loader, target_train_loader, target_test_loader

if __name__ == '__main__':
    model = DAAN.DAANNet(CONFIG).to(DEVICE)
    train_loader, unsuptrain_loader, test_loader = load_data()
    correct = 0
    D_M = 0
    D_C = 0
    MU = 0
    for epoch in range(1, CONFIG['epochs'] + 1):
        train_loader, unsuptrain_loader, test_loader = load_data()
        train(epoch, model, train_loader, unsuptrain_loader)
        t_correct = test(model, test_loader)
        if t_correct > correct:
            correct = t_correct
        print("%s max correct:" % CONFIG['target_path'], correct.item())
        print(CONFIG['source_path'], "to", CONFIG['target_path'])
