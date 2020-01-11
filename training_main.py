import torch
import time
import numpy as np
import torch.optim as optim
import torch.nn as nn
from config import CONFIG
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data_loader_preprocessing.load_csi_data import load_data
from model.SimpleConv import ConvNet
from model.SimpleLSTM import SimpleLSTMNet
from helper.pr_curve import create_pr_curve

RESULT_TRAIN = []
RESULT_TEST = []


def train(config, model, device, train_loader, optimizer, epoch, writer):
    total_loss_train = 0.0
    criterion = nn.CrossEntropyLoss().to(device)
    correct = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        model.train()
        output = model(data)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss_train += loss.data
        preds = output.data.max(dim=1, keepdim=True)[1]
        correct += preds.eq(target.data.view_as(preds)).to(device).sum()
        res_i = 'Epoch: [{}/{}], Batch: [{}/{}], loss: {:.6f}'.format(
            epoch, config['epochs'], batch_idx + 1, len(train_loader), loss.data
        )
    total_loss_train /= len(train_loader)
    accuracy = correct.__float__() * 100. / len(train_loader.dataset)
    res_e = 'Epoch: [{}/{}], training loss: {:.6f}, correct: [{}/{}], training accuracy: {:.4f}%'.format(
        epoch, config['epochs'], total_loss_train, correct, len(train_loader.dataset), accuracy
    )
    # TensorBoard中进行记录
    writer.add_scalar('training loss', total_loss_train, epoch, time.time())
    writer.add_scalar('training acc', accuracy, epoch, time.time())

    tqdm.write(res_e)
    RESULT_TRAIN.append([epoch, total_loss_train, accuracy])

    return model


def test(args, model, device, test_loader, epoch, writer):
    total_loss_test = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_id, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            model.eval()
            ypred = model(data)
            loss = criterion(ypred, target)
            pred = ypred.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).to(device).sum()
            total_loss_test += loss.data

        accuracy = correct.__float__() * 100. / len(test_loader.dataset)
        res = 'Test: total loss: {:.6f}, correct: [{}/{}], testing accuracy: {:.4f}%'.format(
            total_loss_test, correct, len(test_loader.dataset), accuracy
        )
    # TensorBoard中进行记录
    writer.add_scalar('validation loss', loss, epoch, time.time())
    writer.add_scalar('validation acc', accuracy, epoch, time.time())

    tqdm.write(res)
    RESULT_TEST.append([epoch, total_loss_test, accuracy])


if __name__ == '__main__':
    # torch.backends.cudnn.benchmark = True
    torch.manual_seed(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 载入数据集
    train_loader, test_loader, csi_train_label = load_data(CONFIG)

    # 使用TensorBoard进行记录
    writer = SummaryWriter(CONFIG['tensorboard_log_path'])

    # model = ConvNet()
    model = SimpleLSTMNet(CONFIG)
    model.to(device)
    # writer.add_graph(model,data)

    # optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['l2_decay'])
    optimizer = optim.RMSprop(model.parameters(), lr=CONFIG['lr'], alpha=0.9, weight_decay=CONFIG['l2_decay'])
    # optimizer = optim.SGD(model.parameters(), lr=CONFIG['lr'], momentum=CONFIG['momentum'],
    #                       weight_decay=CONFIG['l2_decay'])

    for epoch in tqdm(range(1, CONFIG['epochs'] + 1)):
        train(CONFIG, model, device, train_loader, optimizer, epoch, writer)
        test(CONFIG, model, device, test_loader, epoch, writer)

    # 绘制pr曲线
    # create_pr_curve(writer, model, test_loader, csi_train_label)

    torch.save(model, CONFIG['model_save_path'])  # 保存模型
    writer.close()  # 关闭TensorBoard写入

    # 保存结果成为cvs
    np.savetxt('res_train.csv', np.asarray(RESULT_TRAIN), fmt='%.6f', delimiter=',')
    np.savetxt('res_test.csv', np.asarray(RESULT_TEST), fmt='%.6f', delimiter=',')
