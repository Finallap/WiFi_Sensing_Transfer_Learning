import torch
import os
import math
import time
import DDC_DeepCoral.models as models
from config import CONFIG
import DDC_DeepCoral.utils as utils
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import data_loader_preprocessing.load_csi_data as load_csi_data

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log = []


def test(model, target_test_loader, epoch):
    model.eval()
    test_loss = utils.AverageMeter()
    correct = 0
    criterion = torch.nn.CrossEntropyLoss()
    len_target_dataset = len(target_test_loader.dataset)
    with torch.no_grad():
        for data, target in target_test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            s_output = model.predict(data)
            loss = criterion(s_output, target)
            test_loss.update(loss.item())
            pred = torch.max(s_output, 1)[1]
            correct += torch.sum(pred == target)

    print('{} --> {}: max correct: {}, accuracy{: .2f}%\n'.format(
        source_name, target_name, correct, 100. * correct.__float__() / len_target_dataset))
    writer.add_scalar('validation acc', 100. * correct.__float__() / len_target_dataset, epoch + 1, time.time())


def train(source_loader, target_train_loader, target_test_loader, model, optimizer, CONFIG):
    len_source_loader = len(source_loader)
    len_target_loader = len(target_train_loader)
    for e in range(CONFIG['epochs']):
        train_loss_clf = utils.AverageMeter()
        train_loss_transfer = utils.AverageMeter()
        train_loss_total = utils.AverageMeter()
        model.train()
        iter_source, iter_target = iter(
            source_loader), iter(target_train_loader)
        n_batch = min(len_source_loader, len_target_loader)
        criterion = torch.nn.CrossEntropyLoss()
        for i in range(n_batch):
            data_source, label_source = iter_source.next()
            data_target, _ = iter_target.next()
            data_source, label_source = data_source.to(
                DEVICE), label_source.to(DEVICE)
            data_target = data_target.to(DEVICE)

            optimizer.zero_grad()
            label_source_pred, transfer_loss = model(data_source, data_target)
            clf_loss = criterion(label_source_pred, label_source)
            loss = clf_loss + CONFIG['lambda'] * transfer_loss
            loss.backward()
            optimizer.step()
            train_loss_clf.update(clf_loss.item())
            train_loss_transfer.update(transfer_loss.item())
            train_loss_total.update(loss.item())
            if i % CONFIG['log_interval'] == 0:
                print(
                    'Train Epoch: [{}/{} ({:02d}%)], cls_Loss: {:.6f}, transfer_loss: {:.6f}, total_Loss: {:.6f}'.format(
                        e + 1,
                        CONFIG['epochs'],
                        int(100. * i / n_batch), train_loss_clf.avg, train_loss_transfer.avg, train_loss_total.avg))
        log.append([train_loss_clf.avg, train_loss_transfer.avg, train_loss_total.avg])
        np_log = np.array(log, dtype=float)
        np.savetxt('train_log.csv', np_log, delimiter=',', fmt='%.6f')
        # Tensorborad
        writer.add_scalar('classification loss', train_loss_clf.avg, e + 1, time.time())
        writer.add_scalar('transfer loss', train_loss_transfer.avg, e + 1, time.time())
        writer.add_scalar('training loss', train_loss_total.avg, e + 1, time.time())
        # Test
        test(model, target_test_loader, e)


def load_data():
    source_train_loader, source_test_loader, source_csi_train_label = load_csi_data.load_data(
        os.path.join(CONFIG['dir_path'], CONFIG['source_name']), CONFIG)
    target_train_loader, target_test_loader, target_csi_train_label = load_csi_data.load_data(
        os.path.join(CONFIG['dir_path'], CONFIG['target_name']), CONFIG)
    return source_train_loader, target_train_loader, target_test_loader


if __name__ == '__main__':
    torch.manual_seed(0)
    writer = SummaryWriter(CONFIG['tensorboard_log_path'])

    source_name = CONFIG['source_name']
    target_name = CONFIG['target_name']

    print('Src: %s, Tar: %s' % (source_name, target_name))

    source_loader, target_train_loader, target_test_loader = load_data()

    model = models.Transfer_Net(
        CONFIG, transfer_loss='mmd').to(DEVICE)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=CONFIG['lr'], momentum=CONFIG['momentum'],
                                    weight_decay=CONFIG['l2_decay'])

    train(source_loader, target_train_loader,
          target_test_loader, model, optimizer, CONFIG)

    # writer.close()
