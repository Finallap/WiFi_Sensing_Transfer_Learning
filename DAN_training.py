from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import os
import math
import time
import data_loader_preprocessing.load_csi_data as load_csi_data
from DAN import DANNet as models
from torch.utils import model_zoo
from config import CONFIG
from torch.utils.tensorboard import SummaryWriter

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

cuda = torch.cuda.is_available()
# torch.manual_seed(1)
# if cuda:
#     torch.cuda.manual_seed(1)

src_loader, _, _ = load_csi_data.load_data(
    os.path.join(CONFIG['dir_path'], CONFIG['source_name']), CONFIG)
tgt_train_loader, tgt_test_loader, _ = load_csi_data.load_data(
    os.path.join(CONFIG['dir_path'], CONFIG['target_name']), CONFIG)

src_dataset_len = len(src_loader.dataset)
tgt_dataset_len = len(tgt_test_loader.dataset)
src_loader_len = len(src_loader)
tgt_loader_len = len(tgt_train_loader)


def train(model, optimizer, writer):
    src_iter = iter(src_loader)
    tgt_iter = iter(tgt_train_loader)
    correct = 0
    for i in range(1, CONFIG['epochs'] + 1):
        model.train()
        LEARNING_RATE = CONFIG['lr'] / math.pow((1 + 10 * (i - 1) / (CONFIG['epochs'])), 0.75)
        if (i - 1) % 100 == 0:
            print('learning rate{: .4f}'.format(LEARNING_RATE))
        # optimizer = torch.optim.SGD([
        #     {'params': model.sharedNet.parameters()},
        #     {'params': model.cls_fc.parameters(), 'lr': LEARNING_RATE},
        # ], lr=LEARNING_RATE / 10, momentum=CONFIG['momentum'], weight_decay=CONFIG['l2_decay'])
        try:
            src_data, src_label = src_iter.next()
        except Exception as err:
            src_iter = iter(src_loader)
            src_data, src_label = src_iter.next()

        try:
            tgt_data, _ = tgt_iter.next()
        except Exception as err:
            tgt_iter = iter(tgt_train_loader)
            tgt_data, _ = tgt_iter.next()

        if cuda:
            src_data, src_label = src_data.cuda(), src_label.cuda()
            tgt_data = tgt_data.cuda()

        optimizer.zero_grad()
        src_pred, mmd_loss = model(src_data, tgt_data)
        cls_loss = F.nll_loss(F.log_softmax(src_pred, dim=1), src_label)
        lambd = 2 / (1 + math.exp(-10 * (i) / CONFIG['epochs'])) - 1
        loss = cls_loss + lambd * mmd_loss
        loss.backward()
        optimizer.step()

        training_pred = src_pred.data.max(1)[1]  # get the index of the max log-probability
        training_correct = training_pred.eq(src_label.data.view_as(training_pred)).cpu().sum()
        writer.add_scalar('training acc', 100. * training_correct.__float__() / len(src_label), i, time.time())

        # if i % CONFIG['log_interval'] == 0:
        print('Train iter: {} [({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tmmd_Loss: {:.6f}'.format(
            i, 100. * i / CONFIG['epochs'], loss.item(), cls_loss.item(), mmd_loss.item()))
        writer.add_scalar('training loss', loss.item(), i, time.time())
        writer.add_scalar('classification loss', cls_loss.item(), i, time.time())
        writer.add_scalar('mmd loss', mmd_loss.item(), i, time.time())

        # if i % (CONFIG['log_interval'] * 20) == 0:
        t_correct, t_loss = test(model)
        if t_correct > correct:
            correct = t_correct
        print('src: {} to tgt: {} max correct: {} max accuracy{: .2f}%\n'.format(
            CONFIG['source_name'], CONFIG['target_name'], correct, 100. * correct.__float__() / tgt_dataset_len))
        writer.add_scalar('validation acc', 100. * t_correct.__float__() / tgt_dataset_len, i, time.time())
        writer.add_scalar('validation loss', t_loss, i, time.time())


def test(model):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for tgt_test_data, tgt_test_label in tgt_test_loader:
            if cuda:
                tgt_test_data, tgt_test_label = tgt_test_data.cuda(), tgt_test_label.cuda()
            tgt_test_data, tgt_test_label = Variable(tgt_test_data), Variable(tgt_test_label)
            tgt_pred, mmd_loss = model(tgt_test_data, tgt_test_data)
            test_loss += F.nll_loss(F.log_softmax(tgt_pred, dim=1), tgt_test_label,
                                    reduction='sum').item()  # sum up batch loss
            pred = tgt_pred.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(tgt_test_label.data.view_as(pred)).cpu().sum()

    test_loss /= tgt_dataset_len
    print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        CONFIG['target_name'], test_loss, correct, tgt_dataset_len,
        100. * correct.__float__() / tgt_dataset_len))
    return correct, test_loss


if __name__ == '__main__':
    model = models.DANNet(CONFIG)
    # optimizer = optim.RMSprop(model.parameters(), lr=CONFIG['lr'], alpha=0.9, weight_decay=CONFIG['l2_decay'])
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['l2_decay'])
    print(model)
    if cuda:
        model.cuda()
    writer = SummaryWriter(CONFIG['tensorboard_log_path'])
    train(model, optimizer, writer)
    writer.close()
