import numpy as np
import torch
import torch.utils.data
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Qt5Agg')  # 或者 'TkAgg' 视环境而定

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



class NCA(torch.nn.Module):

    def __init__(self, i, h, o, B):
        super(NCA, self).__init__()
        self.model = MLP(i, h, o)  # Assuming MLP is a valid torch model
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


def load_secom_data():
    path_to_secom = "F:/0002025_bishe/code/Neural-Component-Analysis-master/Neural-Component-Analysis-master/NCA/NCA/secom_data/SECOM.TXT"
    path_to_secom_labels = "F:/0002025_bishe/code/Neural-Component-Analysis-master/Neural-Component-Analysis-master/NCA/NCA/secom_data/SECOM_labels.TXT"

    # Load features and labels
    features = pd.read_csv(path_to_secom, delim_whitespace=True, header=None)
    labels = pd.read_csv(path_to_secom_labels, delim_whitespace=True, header=None, names=["Label", "Timestamp"])

    # Fill missing values
    features.fillna(features.mean(), inplace=True)

    # Extract valid labels (non-zero)
    labels = labels["Label"].astype(int)
    valid_indices = labels != 0
    features = features[valid_indices]
    labels = labels[valid_indices]

    # Normalize features
    scaler = preprocessing.StandardScaler()
    features = scaler.fit_transform(features)

    return features

class Metric:
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
    l2norm = Metric()
    for data in train_loader:
        x = data
        #x = torch.autograd.Variable(data)
        y = model(x)

        loss = torch.mean(torch.pow(x - y, 2))
        l2norm.update(x.data.cpu().numpy(), y.data.cpu().numpy())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return l2norm.get()


def kde(spe_train, spe_test, T2_train, T2_test):
    # 计算Gtrain的协方差矩阵

    cov_T2_train = np.cov(T2_train, rowvar=False)  # rowvar=False表示变量在列中
    cov_T2_test = np.cov(T2_test, rowvar=False)
    T2 = np.zeros(T2_train.shape[0])
    XT2 = np.zeros(T2_test.shape[0])
    for i in range(T2_train.shape[0]):
        T2[i] = np.dot(T2_train[i, :], np.dot(cov_T2_train, T2_train[i, :].T))
    for i in range(T2_test.shape[0]):
        XT2[i] = np.dot(T2_test[i, :], np.dot(cov_T2_test, T2_test[i, :].T))

    SPE = spe_train.T
    XSPE = spe_test.T


    kde_T2 = gaussian_kde(T2)#T2的核密度估计
    kde_SPE = gaussian_kde(SPE)#SPE的核密度估计

def compute_hidden_stats(model, data_loader):
    hidden_outputs = []
    for data in data_loader:
        x = data
        z = model.a(model.h(x))
        hidden_outputs.append(z.data.cpu().numpy())

    hidden_outputs = np.concatenate(hidden_outputs, axis=0)
    mean_z = np.mean(hidden_outputs, axis=0)
    cov_z = np.cov(hidden_outputs, rowvar=False)
    cov_inv_z = np.linalg.inv(cov_z)
    return mean_z, cov_inv_z



def compute_limit(data, r=0.99):
    kde = gaussian_kde(data)
    xmesh = np.linspace(min(data) - 1, max(data) + 1, 1000)  # 扩展范围
    density = kde(xmesh)
    cdf = np.cumsum(density) * (xmesh[1] - xmesh[0])

    if len(np.where(cdf >= r)[0]) == 0:
        print("Warning: No value in CDF meets the threshold.")
        return xmesh[-1]  # 返回最大值作为控制限

    i = np.where(cdf >= r)[0][0]
    return xmesh[i]

def validate_with_spe_t2(model, val_loader, mean_z, cov_inv_z):
    model.eval()
    spe_values = []
    t2_values = []
    for data in val_loader:
        x = data
        y = model(x)
        z = model.a(model.h(x))

        spe = torch.mean(torch.pow(x - y, 2), dim=1).data.cpu().numpy()
        spe_values.extend(spe)

        z_np = z.data.cpu().numpy()
        t2 = np.sum((z_np - mean_z) @ cov_inv_z * (z_np - mean_z), axis=1)
        t2_values.extend(t2)

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            out = self.model(x)
            loss = torch.mean(torch.pow(torch.mm(out, self.B.t()) - x, 2))
            loss_single = torch.mean(torch.pow(torch.mm(out, self.B.t()) - x, 2), dim=1)
        return loss, out, loss_single


    return np.array(spe_values), np.array(t2_values)



def plot_spe_t2_results(test_spe, test_t2, spe_cl, t2_cl):
    # 使用 `constrained_layout=True` 自动调整布局
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)

    # SPE 折线图
    axes[0].plot(test_spe, label='SPE Values', color='blue', alpha=0.7)
    axes[0].axhline(y=spe_cl, color='red', linestyle='--', label='SPE Control Limit')
    axes[0].set_title('SPE Results')
    axes[0].set_xlabel('Sample Index')
    axes[0].set_ylabel('SPE Value')
    axes[0].legend()
    axes[0].grid(True)

    # T² 折线图
    axes[1].plot(test_t2, label='T² Values', color='green', alpha=0.7)
    axes[1].axhline(y=t2_cl, color='red', linestyle='--', label='T² Control Limit')
    axes[1].set_title('T² Results')
    axes[1].set_xlabel('Sample Index')
    axes[1].set_ylabel('T² Value')
    axes[1].legend()
    axes[1].grid(True)

    # 显示图表
    plt.show()


def main():
    data = load_secom_data()
    train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

    train_dataset = torch.tensor(train_data, dtype=torch.float32)
    test_dataset = torch.tensor(test_data, dtype=torch.float32)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    pca = PCA(27).fit(train_data)
    B = torch.autograd.Variable(torch.from_numpy(pca.components_.T))

    x = torch.autograd.Variable(torch.from_numpy(train_data))
    model= NCA(train_data.shape[1], 27, 27, B)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)

    for _ in range(500):
        train(model, optimizer, train_loader)

    mean_z, cov_inv_z = compute_hidden_stats(model, train_loader)
    train_spe, train_t2 = validate_with_spe_t2(model, train_loader, mean_z, cov_inv_z)

    T2_limit = compute_limit(train_t2)
    SPE_limit = compute_limit(train_spe)

    print(f"T² Limit: {T2_limit}")
    print(f"SPE Limit: {SPE_limit}")

    test_spe, test_t2 = validate_with_spe_t2(model, test_loader, mean_z, cov_inv_z)
    plot_spe_t2_results(test_spe, test_t2, SPE_limit, T2_limit)

if __name__ == '__main__':
    main()