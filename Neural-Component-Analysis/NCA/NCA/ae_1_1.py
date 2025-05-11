import torch
import torch.utils.data
import numpy as np
from sklearn import preprocessing
import util
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt


class MLP(torch.nn.Module):
    def __init__(self, i, h, o):
        super(MLP, self).__init__()
        # 编码器结构
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(i, h),
            torch.nn.Sigmoid()
        )
        # 解码器结构
        self.decoder = torch.nn.Linear(h, o)

        # 训练集统计量存储
        self.register_buffer('mean_h', None)
        self.register_buffer('inv_cov_h', None)

    def forward(self, x):
        hidden = self.encoder(x)
        return self.decoder(hidden)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            # 前向传播获取潜在空间
            hidden = self.encoder(x)
            pred_x = self.decoder(hidden)

            # 计算SPE（按样本）
            spe = torch.sum((x - pred_x) ** 2, dim=1)

            # 计算T²统计量
            if self.mean_h is not None and self.inv_cov_h is not None:
                delta = hidden - self.mean_h.unsqueeze(0)
                T2 = torch.diag(delta @ self.inv_cov_h @ delta.T)
            else:
                T2 = torch.zeros(x.size(0))

        return spe, T2


class Metric(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self._sum = 0
        self._count = 0

    def update(self, pred, target):
        self._sum += np.sum(np.mean(np.power(target - pred, 2), axis=1))
        self._count += pred.shape[0]

    def get(self):
        return self._sum / self._count if self._count != 0 else 0


def train(model, optimizer, train_loader):
    model.train()
    metric = Metric()
    for data in train_loader:
        x = data.float()
        optimizer.zero_grad()

        # 前向传播
        pred = model(x)
        loss = torch.mean((pred - x) ** 2)

        # 反向传播
        loss.backward()
        optimizer.step()

        # 更新指标
        metric.update(pred.detach().cpu().numpy(), x.detach().cpu().numpy())

    return metric.get()


def compute_control_limits(model, dataloader):
    spe_list, T2_list = [], []
    with torch.no_grad():
        for data in dataloader:
            x = data.float()
            spe, T2 = model.predict(x)
            spe_list.append(spe)
            T2_list.append(T2)

    # 合并所有批次的结果
    spe_all = torch.cat(spe_list).numpy().flatten()  # 确保形状为 (N,)
    T2_all = torch.cat(T2_list).numpy().flatten()  # 确保形状为 (N,)

    # 生成网格点用于评估KDE
    x_grid_spe = np.linspace(spe_all.min(), spe_all.max(), 1000)
    x_grid_T2 = np.linspace(T2_all.min(), T2_all.max(), 1000)

    # 计算SPE控制限（95%分位数）
    kde_spe = gaussian_kde(spe_all)
    cdf_spe = np.array([kde_spe.integrate_box_1d(-np.inf, x) for x in x_grid_spe])
    spe_limit = x_grid_spe[np.where(cdf_spe >= 0.95)[0][0]]

    # 计算T²控制限（95%分位数）
    kde_T2 = gaussian_kde(T2_all)
    cdf_T2 = np.array([kde_T2.integrate_box_1d(-np.inf, x) for x in x_grid_T2])
    T2_limit = x_grid_T2[np.where(cdf_T2 >= 0.95)[0][0]]

    return spe_limit, T2_limit

def get_train_data():
    train_data, _ = util.read_data(error=0, is_train=True)
    scaler = preprocessing.StandardScaler().fit(train_data)
    train_data = scaler.transform(train_data)
    return torch.from_numpy(train_data.astype(np.float32))


def get_test_data(fault_id):
    test_data, _ = util.read_data(error=fault_id, is_train=False)
    train_data, _ = util.read_data(error=0, is_train=True)
    scaler = preprocessing.StandardScaler().fit(train_data)
    test_data = scaler.transform(test_data)
    return torch.from_numpy(test_data.astype(np.float32))


def main():
    # 数据准备
    train_data = get_train_data()
    test_data = get_test_data(7)  # 使用故障7作为示例

    # 创建DataLoader
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=64, shuffle=True)

    # 模型初始化
    model = MLP(52, 27, 52)  # 输入输出维度52，隐藏层27
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 训练循环
    for epoch in range(300):
        train_loss = train(model, optimizer, train_loader)

        if epoch % 10 == 0:
            print(f'Epoch {epoch:03d} | Train Loss: {train_loss:.4f}')

    # 计算训练集统计量
    with torch.no_grad():
        # 获取全部训练数据的潜在变量
        hidden_list = []
        for data in train_loader:
            hidden = model.encoder(data.float())
            hidden_list.append(hidden)
        hidden_all = torch.cat(hidden_list)

        # 计算并存储全局统计量
        model.mean_h = torch.mean(hidden_all, dim=0)
        cov_h = torch.cov(hidden_all.T)
        model.inv_cov_h = torch.linalg.pinv(cov_h)

    # 计算控制限
    spe_limit, T2_limit = compute_control_limits(model, train_loader)

    # 测试集预测
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64)
    spe_test, T2_test = model.predict(test_data.float())

    # 保存结果
    #np.save('spe_train.npy', spe_train.numpy())
    #np.save('T2_train.npy', T2_train.numpy())
    np.save('spe_test.npy', spe_test.numpy())
    np.save('T2_test.npy', T2_test.numpy())

    # 打印控制限
    print(f'\nControl Limits (95%):')
    print(f'SPE Limit: {spe_limit:.4f}')
    print(f'T² Limit: {T2_limit:.4f}')

    # 可视化监控图
    plt.figure(figsize=(12, 6))
    plt.subplot(211)
    plt.plot(spe_test.numpy(), 'b', label='SPE')
    plt.axhline(spe_limit, color='r', linestyle='--', label='95% Limit')
    plt.ylabel('SPE')

    plt.subplot(212)
    plt.plot(T2_test.numpy(), 'g', label='T²')
    plt.axhline(T2_limit, color='r', linestyle='--', label='95% Limit')
    plt.ylabel('T²')
    plt.show()


if __name__ == '__main__':
    main()
