import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.covariance import EmpiricalCovariance

# ===== 1. 位置编码 =====
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # [1, max_len, d_model]

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

# ===== 2. 改进版 Transformer AutoEncoder =====
class TransformerWithFusion(nn.Module):
    def __init__(self, input_dim, extra_dim=8, embed_dim=64, num_heads=4, num_layers=2):
        super().__init__()
        self.input_fc = nn.Linear(input_dim + extra_dim, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_fc = nn.Linear(embed_dim, input_dim)

    def forward(self, x_fusion):
        x_embed = self.input_fc(x_fusion).unsqueeze(1)
        x_pos = self.pos_encoder(x_embed)
        encoded = self.transformer_encoder(x_pos).mean(dim=1)
        recon = self.output_fc(encoded)
        return encoded, recon

# ===== 3. Center Loss =====
class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, device='cpu'):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim).to(device))

    def forward(self, features, labels):
        centers_batch = self.centers.index_select(0, labels)
        return ((features - centers_batch) ** 2).sum() / features.size(0)

# ===== 4. 数据加载 + PCA工程变量 =====
def load_data_with_pca(n_pca=8):

    train = np.fromfile(
        'F:/0002025_bishe/code/Neural-Component-Analysis-master/Neural-Component-Analysis-master/NCA/NCA/data/d00.dat',
        dtype=np.float32, sep='   ').reshape(-1, 52)
    test0 = np.fromfile(
        'F:/0002025_bishe/code/Neural-Component-Analysis-master/Neural-Component-Analysis-master/NCA/NCA/data/d00_te.dat',
        dtype=np.float32, sep='  ').reshape(-1, 52)
    test1 = np.fromfile(
        'F:/0002025_bishe/code/Neural-Component-Analysis-master/Neural-Component-Analysis-master/NCA/NCA/data/d04_te.dat',
        dtype=np.float32, sep='  ').reshape(-1, 52)


    X_train = preprocessing.StandardScaler().fit_transform(train)
    X_test = preprocessing.StandardScaler().fit_transform(np.vstack([test0, test1]))
    y_test = np.array([0] * len(test0) + [1] * len(test1))


    pca = PCA(n_components=n_pca)
    train_pca = pca.fit_transform(X_train)
    test_pca = pca.transform(X_test)

    return X_train, train_pca, X_test, test_pca, y_test

# ===== 5. 统计量计算 =====
def calc_statistics(encoded, recon, raw_input, mean, inv_cov):
    SPE = np.sum((raw_input - recon) ** 2, axis=1)
    T2 = np.einsum('ij,jk,ik->i', encoded - mean, inv_cov, encoded - mean)
    return T2, SPE

# ===== 6. 可视化 =====
def plot_T2_SPE(T2, SPE, t2_limit, spe_limit, split_index=160):
    idx = np.arange(len(T2))
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axs[0].plot(idx[:split_index], T2[:split_index], 'b-', label='Normal')
    axs[0].plot(idx[split_index:], T2[split_index:], 'r-', label='Fault')
    axs[0].axhline(t2_limit, linestyle='--', color='black')
    axs[0].set_ylabel('T²'); axs[0].legend(); axs[0].grid(True)

    axs[1].plot(idx[:split_index], SPE[:split_index], 'b-')
    axs[1].plot(idx[split_index:], SPE[split_index:], 'r-')
    axs[1].axhline(spe_limit, linestyle='--', color='black')
    axs[1].set_ylabel('SPE'); axs[1].set_xlabel('Index')
    axs[1].grid(True)
    plt.tight_layout()
    plt.show()

# ===== 7. 主流程 =====
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train, X_train_pca, X_test, X_test_pca, y_test = load_data_with_pca(n_pca=8)

    X_train_fused = np.concatenate([X_train, X_train_pca], axis=1)
    X_test_fused = np.concatenate([X_test, X_test_pca], axis=1)

    x_train_tensor = torch.tensor(X_train_fused, dtype=torch.float32).to(device)
    x_test_tensor = torch.tensor(X_test_fused, dtype=torch.float32).to(device)
    y_train_tensor = torch.zeros(len(X_train), dtype=torch.long).to(device)

    model = TransformerWithFusion(input_dim=52, extra_dim=8).to(device)
    center_loss_fn = CenterLoss(num_classes=1, feat_dim=64, device=device)
    recon_loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(list(model.parameters()) + list(center_loss_fn.parameters()), lr=1e-3)

    # ===== 训练 =====
    for epoch in range(20):
        model.train()
        feat, recon = model(x_train_tensor)
        loss_recon = recon_loss_fn(recon, torch.tensor(X_train, dtype=torch.float32).to(device))
        loss_center = center_loss_fn(feat, y_train_tensor)
        loss = loss_recon + 0.1 * loss_center
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        print(f"Epoch {epoch+1}, Recon: {loss_recon.item():.4f}, Center: {loss_center.item():.4f}")

    # ===== 推理 & 统计量 =====
    model.eval()
    feat_train, recon_train = model(x_train_tensor)
    feat_test,  recon_test  = model(x_test_tensor)

    feat_train, recon_train = feat_train.cpu().detach().numpy(), recon_train.cpu().detach().numpy()
    feat_test,  recon_test  = feat_test.cpu().detach().numpy(),  recon_test.cpu().detach().numpy()

    cov = EmpiricalCovariance().fit(feat_train)
    T2_train, SPE_train = calc_statistics(feat_train, recon_train, X_train, cov.location_, cov.precision_)
    T2_test,  SPE_test  = calc_statistics(feat_test,  recon_test,  X_test,  cov.location_, cov.precision_)

    t2_limit = np.quantile(T2_train, 0.99)
    spe_limit = np.quantile(SPE_train, 0.99)

    print(f"T² Limit: {t2_limit:.2f}, SPE Limit: {spe_limit:.2f}")
    plot_T2_SPE(T2_test, SPE_test, t2_limit, spe_limit, split_index=160)

if __name__ == '__main__':
    main()
