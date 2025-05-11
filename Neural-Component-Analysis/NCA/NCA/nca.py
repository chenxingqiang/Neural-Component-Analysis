from datetime import datetime
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn import preprocessing
import util
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import matplotlib

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
        self.model = MLP(i, h, o)
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
        return loss

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            out = self.model(x)
            loss = torch.mean(torch.pow(torch.mm(out, self.B.t()) - x, 2))
            loss_single = torch.mean(torch.pow(torch.mm(out, self.B.t()) - x, 2), dim=1)
        return loss, out, loss_single



def get_train_data():
    train_data, _ = util.read_data(error=0, is_train=True)
    train_data = preprocessing.StandardScaler().fit_transform(train_data)
    return train_data


def get_test_data():
    test_data = []

    i = 8
    data, _ = util.read_data(error=i, is_train=False)
    test_data.append(data)
    test_data = np.concatenate(test_data)
    train_data, _ = util.read_data(error=0, is_train=True)
    scaler = preprocessing.StandardScaler().fit(train_data)
    test_data = scaler.transform(test_data)
    return test_data





def main():
    train_data = get_train_data()
    test_data = get_test_data()
    pca = PCA(27).fit(train_data)
    B = torch.autograd.Variable(torch.from_numpy(pca.components_.T))

    x = torch.autograd.Variable(torch.from_numpy(train_data))
    nca = NCA(52, 27, 27, B)
    for i in range(500):
        train_loss = nca.train(x)
        train_loss_2, T2_train, spe_train = nca.predict(torch.autograd.Variable(x))
        test_loss, T2_test, spe_test = nca.predict(torch.autograd.Variable(torch.from_numpy(test_data)))
        if i % 5 == 0:
            print('epoch[{}]  train_loss = {:0.3f} train_loss_2 = {:0.3f}  test_loss = {:0.3f}  '.format(i, train_loss, train_loss_2, test_loss))
    util.write_data('1_spe_train', spe_train.data.cpu().numpy())
    util.write_data('1_spe_test', spe_test.data.cpu().numpy())
    util.write_data('1_T2_train', T2_train.data.cpu().numpy())
    util.write_data('1_T2_test', T2_test.data.cpu().numpy())

if __name__ == '__main__':
    main()