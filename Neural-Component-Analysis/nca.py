from datetime import datetime
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn import preprocessing
import util
import scipy.io


class MLP(torch.nn.Module):

    def __init__(self, i, h, o):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(i, h)
        self.bn = torch.nn.BatchNorm1d(h)
        self.act = torch.nn.LeakyReLU(True)
        self.fc2 = torch.nn.Linear(h, o)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class NCA(object):

    def __init__(self, i, h, o, B):
        self.model = MLP(i, h, o).cuda()
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)
        self.B = B

    def train(self, x):
        self.model.train()

        o = self.model(x)
        loss = torch.mean(torch.pow(torch.mm(o, self.B.t()) - x, 2))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        U, _, V = torch.svd(torch.mm(x.t().data, o.data))
        self.B = torch.autograd.Variable(torch.mm(U, V.t()))

        return loss.data.cpu().numpy()

    def predict(self, x):
        self.model.eval()
        out = self.model(x)
        return out


def get_train_data():
    train_data, _ = util.read_data(error=0, is_train=True)
    train_data = preprocessing.StandardScaler().fit_transform(train_data)
    return train_data


def get_test_data():
    test_data = []
    for i in range(22):
        data, _ = util.read_data(error=i, is_train=False)
        test_data.append(data)
    test_data = np.concatenate(test_data)
    train_data, _ = util.read_data(error=0, is_train=True)
    scaler = preprocessing.StandardScaler().fit(train_data)
    test_data = scaler.transform(test_data)
    return test_data


# 保存模型的权重和其他相关参数
def save_model(model, B, filename="nca_model.pth"):
    torch.save({
        'model_state_dict': model.state_dict(),
        'B': B
    }, filename)


# 加载模型并提取权重
def load_model(filename="nca_model.pth"):
    checkpoint = torch.load(filename)
    model = MLP(52, 27, 27)
    model.load_state_dict(checkpoint['model_state_dict'])
    B = checkpoint['B']
    return model, B


def main():
    train_data = get_train_data()
    test_data = get_test_data()
    pca = PCA(27).fit(train_data)
    B = torch.autograd.Variable(torch.from_numpy(pca.components_.T).cuda())
    x = torch.autograd.Variable(torch.from_numpy(train_data).cuda())

    nca = NCA(52, 27, 27, B)

    for i in range(500):
        loss = nca.train(x)
        if i % 5 == 0:
            print('{} epoch[{}] loss = {:0.3f}'.format(datetime.now(), i, loss))

    pred = nca.predict(torch.autograd.Variable(torch.from_numpy(test_data).cuda()))
    # 假设模型已经训练好了
    save_model(nca.model, nca.B)
    # 训练完后，保存模型和B矩阵
    save_model(nca.model, nca.B, "nca_model.pth")
    # 保存B矩阵到MAT文件
    scipy.io.savemat("B_matrix.mat", {'B': nca.B.cpu().numpy()})


if __name__ == '__main__':
    main()
