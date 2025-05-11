from datetime import datetime
import numpy as np
import torch
import torch.utils.data
from sklearn import preprocessing
import util
import torch.nn as nn
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import matplotlib

class MLP(torch.nn.Module):
    def __init__(self, i, h, o):
        super(MLP, self).__init__()
        self.h = torch.nn.Linear(i, h)
        #self.a = torch.nn.Tanh()
        #self.a = torch.nn.Sigmoid()
        self.a = torch.nn.ReLU()
        self.o = torch.nn.Linear(h, o)
        self.sigmoid = torch.nn.Sigmoid()  # 添加Sigmoid激活函数

    def forward(self, x):
        x = self.sigmoid(self.o(self.a(self.h(x))))  # 在输出层应用Sigmoid
        return x

    def predict(self, x):
        self.eval()  # 切换到评估模式
        with torch.no_grad():
            hidden = self.a(self.h(x))  # 隐藏层输出
            pred_x = self.sigmoid(self.o(hidden))  # 最终预测（应用Sigmoid）
            loss = torch.mean(torch.pow(pred_x - x, 2))  # 平均平方误差
            loss_single = torch.mean(torch.pow(pred_x - x, 2), dim=1)  # 单样本误差
            # 计算T2统计量
            mean_hidden = torch.mean(hidden, dim=0)
            cov_hidden = torch.cov(hidden.T)
            inv_cov_hidden = torch.linalg.pinv(cov_hidden)
            delta = hidden - mean_hidden
            T2 = torch.diag(delta @ inv_cov_hidden @ delta.T)
        return loss, pred_x, loss_single


class Metirc(object):
    def __init__(self):
        self._sum = 0
        self._count = 0

    def update(self, pred, target):
        self._sum += np.sum(np.mean(np.power(target - pred, 2), axis=1))
        self._count += pred.shape[0]

    def get(self):
        return self._sum / self._count


def train(model, optimizer, train_loader):
    model.train()
    l2norm = Metirc()
    for data in train_loader:
        x = torch.autograd.Variable(data.float())  # 确保数据类型正确
        y = model(x)
        loss = torch.mean(torch.pow(x - y, 2))
        l2norm.update(x.data.cpu().numpy(), y.data.cpu().numpy())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return l2norm.get()


def validate(model, val_loader):
    model.eval()
    l2norm = Metirc()
    with torch.no_grad():
        for data in val_loader:
            x = torch.autograd.Variable(data.float())
            y = model(x)
            l2norm.update(y.data.cpu().numpy(), x.data.cpu().numpy())
    return l2norm.get()


def get_train_data():
    train_data, _ = util.read_data(error=0, is_train=True)
    train_data = preprocessing.StandardScaler().fit_transform(train_data)
    return train_data.astype(np.float32)  # 转换为float32


def get_test_data():
    test_data = []
    i = 7
    data, _ = util.read_data(error=i, is_train=False)
    test_data.append(data)
    test_data = np.concatenate(test_data)
    train_data, _ = util.read_data(error=0, is_train=True)
    scaler = preprocessing.StandardScaler().fit(train_data)
    test_data = scaler.transform(test_data)
    return test_data.astype(np.float32)  # 转换为float32


def main():
    # 数据准备
    train_data = get_train_data()
    test_data = get_test_data()

    train_dataset = torch.from_numpy(train_data)
    test_dataset = torch.from_numpy(test_data)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 模型初始化
    model = MLP(52, 27, 52)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)

    # 训练循环
    for epoch in range(500):
        train_loss = train(model, optimizer, train_loader)

        # 计算训练集统计量
        with torch.no_grad():
            _, T2_train, spe_train = model.predict(train_dataset.float())
            _, T2_test, spe_test = model.predict(test_dataset.float())

        if epoch % 5 == 0:
            val_loss = validate(model, test_loader)
            print(f'epoch[{epoch}]  train_loss={train_loss:.3f}  val_loss={val_loss:.3f}')


    # 保存结果
    util.write_data('7_spe_train_ae', spe_train.numpy())
    util.write_data('7_spe_test_ae', spe_test.numpy())
    util.write_data('7_T2_train_ae', T2_train.numpy())
    util.write_data('7_T2_test_ae', T2_test.numpy())


if __name__ == '__main__':
    main()