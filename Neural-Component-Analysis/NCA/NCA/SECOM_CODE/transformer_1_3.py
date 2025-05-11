from sklearn.covariance import EmpiricalCovariance
from datetime import datetime
import torch
import torch.utils.data
from sklearn import preprocessing
import util
import torch.nn as nn
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os

class PatchEmbedding(nn.Module):
    def __init__(self, input_dim, patch_size=4, embed_dim=128):
        super().__init__()
        assert input_dim % patch_size == 0, "Input dim must be divisible by patch size"
        self.n_patches = input_dim // patch_size
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size, embed_dim)

    def forward(self, x):
        B, D = x.shape
        x = x.view(B, self.n_patches, self.patch_size)
        return self.proj(x)


class TransformerPatchAE_CLS(nn.Module):
    def __init__(self, input_dim,orig_dim=None,  patch_size=4, embed_dim=128, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        assert input_dim % patch_size == 0, f"input_dim ({input_dim}) must be divisible by patch_size ({patch_size})"
        self.n_patches = input_dim // patch_size
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbedding(input_dim, patch_size, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches + 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True,
            dropout=dropout, norm_first=True, activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.recon_linear = nn.Linear(embed_dim, patch_size)
        #self.final_proj = nn.Linear(self.n_patches * patch_size, input_dim)

        self.orig_dim = orig_dim if orig_dim is not None else input_dim
        self.final_proj = nn.Linear(self.n_patches * patch_size, self.orig_dim)

    def forward(self, x):
        B = x.size(0)
        x_patch = self.patch_embed(x)  # (B, N, E)
        x_patch = self.norm(x_patch)
        x_patch = self.dropout(x_patch)

        cls_tokens = self.cls_token.expand(B, 1, self.embed_dim)  # (B, 1, E)
        x_input = torch.cat((cls_tokens, x_patch), dim=1)  # (B, N+1, E)
        x_input = x_input + self.pos_embed[:, :x_input.size(1), :]

        x_encoded = self.transformer_encoder(x_input)  # (B, N+1, E)
        x_encoded = self.dropout(x_encoded)

        cls_feature = x_encoded[:, 0, :]  # 取CLS Token特征

        recon_patches = self.recon_linear(x_encoded[:, 1:, :])  # 只重建patch
        recon = recon_patches.flatten(start_dim=1)
        recon = self.final_proj(recon)
        return cls_feature, recon


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = (anchor - positive).pow(2).sum(1)
        neg_dist = (anchor - negative).pow(2).sum(1)
        losses = torch.relu(pos_dist - neg_dist + self.margin)
        return losses.mean()


def load_data_with_pca(n_pca=28):

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


def calc_statistics(encoded, recon, raw_input, mean, inv_cov):
    SPE = np.sum((raw_input - recon) ** 2, axis=1)
    T2 = np.einsum('ij,jk,ik->i', encoded - mean, inv_cov, encoded - mean)
    return T2, SPE


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

def write_data_1(file_name, data):
    """
    """
    fi = os.path.join('data/{}.dat'.format(file_name))
    np.savetxt(fi, data, fmt='%f', delimiter='\t')

def write_data(file_name, data):
    save_dir = 'F:/0002025_bishe/code/Neural-Component-Analysis-master/Neural-Component-Analysis-master/NCA/NCA/data'
    os.makedirs(save_dir, exist_ok=True)  # ✅ 确保 data/ 目录存在
    fi = os.path.join(save_dir, f'{file_name}.dat')
    np.savetxt(fi, data, fmt='%.6f', delimiter='\t')


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train, X_train_pca, X_test, X_test_pca, y_test = load_data_with_pca(n_pca=28)
    X_train_fused = np.concatenate([X_train, X_train_pca], axis=1)
    X_test_fused = np.concatenate([X_test, X_test_pca], axis=1)

    x_train_tensor = torch.tensor(X_train_fused, dtype=torch.float32).to(device)
    x_test_tensor = torch.tensor(X_test_fused, dtype=torch.float32).to(device)
    y_train_tensor = torch.zeros(len(X_train), dtype=torch.long).to(device)

    model = TransformerPatchAE_CLS(input_dim=80, orig_dim=52, patch_size=4).to(device)
    #center_loss_fn = CenterLoss(num_classes=1, feat_dim=128, device=device)
    triplet_loss_fn = TripletLoss(margin=1.0)
    recon_loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(list(model.parameters()) + list(triplet_loss_fn.parameters()), lr=1e-3)

    #triplet_loss_fn = TripletLoss(margin=1.0)

    for epoch in range(20):
        model.train()
        feat, recon = model(x_train_tensor)

        # 1. 计算重构损失
        loss_recon = nn.L1Loss()(recon, torch.tensor(X_train, dtype=torch.float32).to(device))

        # 2. 构造三元组 (anchor, positive, negative) —— 保证三个数量一致
        batch_size = len(x_train_tensor) // 3
        indices = torch.randperm(len(x_train_tensor))

        feat_anchor = feat[indices[:batch_size]]
        feat_positive = feat[indices[batch_size:2 * batch_size]]
        feat_negative = feat[indices[2 * batch_size:3 * batch_size]]

        # 3. Triplet Loss
        loss_triplet = triplet_loss_fn(feat_anchor, feat_positive, feat_negative)

        # 4. 总损失
        loss = loss_recon + 0.1 * loss_triplet
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}, Recon: {loss_recon.item():.4f}, Triplet: {loss_triplet.item():.4f}")

    model.eval()
    feat_train, recon_train = model(x_train_tensor)
    feat_test, recon_test = model(x_test_tensor)
    feat_train, recon_train = feat_train.cpu().detach().numpy(), recon_train.cpu().detach().numpy()
    feat_test, recon_test = feat_test.cpu().detach().numpy(), recon_test.cpu().detach().numpy()

    cov = EmpiricalCovariance().fit(feat_train)
    T2_train, SPE_train = calc_statistics(feat_train, recon_train, X_train, cov.location_, cov.precision_)


    T2_test, SPE_test = calc_statistics(feat_test, recon_test, X_test, cov.location_, cov.precision_)

    # 保存结果
    write_data('4_spe_train_ft', SPE_train)
    write_data('4_spe_test_ft', SPE_test)
    write_data('4_T2_train_ft', T2_train)
    write_data('4_T2_test_ft', T2_test)

    #t2_limit = np.quantile(T2_train, 0.99)
    #spe_limit = np.quantile(SPE_train, 0.99)

    #print(f"T² Limit: {t2_limit:.2f}, SPE Limit: {spe_limit:.2f}")
    #plot_T2_SPE(T2_test, SPE_test, t2_limit, spe_limit, split_index=160)


if __name__ == '__main__':
    main()
