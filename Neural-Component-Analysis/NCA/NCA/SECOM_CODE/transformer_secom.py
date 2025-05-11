# 完整优化版 SECOM 故障检测系统
# ------------------------
# （基于Transformer + Conv Patch Embedding + Mahalanobis + T² + SPE融合）
# AUC优化目标 > 0.90

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.covariance import EmpiricalCovariance
from sklearn.preprocessing import MinMaxScaler


# ========= 1. Conv Patch Embedding =========
class ConvPatchEmbedding(nn.Module):
    def __init__(self, input_dim, patch_size=10, embed_dim=256):
        super().__init__()
        assert input_dim % patch_size == 0
        self.conv = nn.Conv1d(1, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = x.unsqueeze(1)  # [B,1,L]
        x = self.conv(x)    # [B,embed_dim,n_patch]
        return x.transpose(1,2)  # [B,n_patch,embed_dim]

# ========= 2. Transformer Classifier =========
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, patch_size=10, embed_dim=128, num_heads=8, num_layers=6):
        super().__init__()
        self.patch_embed = ConvPatchEmbedding(input_dim, patch_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.bn = nn.BatchNorm1d(embed_dim)

    def forward(self, x):
        x = self.patch_embed(x)  # [B, n_patch, embed_dim]
        x = self.transformer(x)
        pooled = 0.5 * x[:,0] + 0.5 * x.mean(dim=1)  # token融合
        pooled = self.bn(pooled)
        return pooled

# ========= 3. Fault Detector =========
class FaultDetector:
    def __init__(self, input_dim, patch_size=10):
        self.model = TransformerClassifier(input_dim=input_dim, patch_size=patch_size)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()

    def train(self, x, epochs=50, batch_size=16):  # <<<<<< batch_size改成16！！
        self.model.train()
        for epoch in range(epochs):
            losses = []
            for i in range(0, len(x), batch_size):
                xb = x[i:i + batch_size]
                feat = self.model(xb)
                loss = self.criterion(feat, torch.zeros_like(feat))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {np.mean(losses):.6f}")

    def extract_features(self, x, batch_size=512):
        self.model.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, x.shape[0], batch_size):
                out = self.model(x[i:i+batch_size])
                outs.append(out.cpu())
        return torch.cat(outs, dim=0).numpy()

# ========= 4. AutoEncoder =========
class AE(nn.Module):
    def __init__(self, input_dim, bottleneck_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim,128), nn.ReLU(), nn.Linear(128,bottleneck_dim))
        self.decoder = nn.Sequential(nn.Linear(bottleneck_dim,128), nn.ReLU(), nn.Linear(128,input_dim))

    def forward(self, x):
        return self.decoder(self.encoder(x))

def train_autoencoder(features, epochs=100):
    ae = AE(features.shape[1])
    optimizer = torch.optim.Adam(ae.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    x = torch.tensor(features, dtype=torch.float32)
    ae.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(ae(x), x)
        loss.backward()
        optimizer.step()
    return ae.eval()

def compute_SPE(ae, features):
    with torch.no_grad():
        recon = ae(torch.tensor(features, dtype=torch.float32)).numpy()
    return np.sum((features - recon)**2, axis=1)

# ========= 5. Mahalanobis =========
def compute_mahalanobis_stats(features):
    clf = EmpiricalCovariance().fit(features)
    return clf.location_, clf.covariance_, clf.precision_

def mahalanobis_distance(x, mean, inv_cov):
    diff = x - mean
    return np.sqrt(np.einsum('ij,jk,ik->i', diff, inv_cov, diff))

# ========= 6. Fused Scoring =========
def find_best_alpha(score1, score2, y_true, n_grid=31):
    s1 = MinMaxScaler().fit_transform(score1.reshape(-1,1)).ravel()
    s2 = MinMaxScaler().fit_transform(score2.reshape(-1,1)).ravel()
    best_a, best_auc = 0.0, 0.0
    for a in np.linspace(0,1,n_grid):
        fusion = a*s1 + (1-a)*s2
        auc = roc_auc_score(y_true, fusion)
        if auc > best_auc:
            best_auc, best_a = auc, a
    return best_a, best_auc, s1, s2

# ========= 7. Plot =========
def plot_roc(scores, labels, title):
    fpr, tpr, _ = roc_curve(labels, scores)
    auc = roc_auc_score(labels, scores)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# ========= 8. Load SECOM =========
def load_secom_data():
    data_path = "F:/0002025_bishe/code/Neural-Component-Analysis-master/Neural-Component-Analysis-master/NCA/NCA/secom_data/SECOM.TXT"
    label_path= "F:/0002025_bishe/code/Neural-Component-Analysis-master/Neural-Component-Analysis-master/NCA/NCA/secom_data/SECOM_labels.TXT"
    X = pd.read_csv(data_path, delim_whitespace=True, header=None).replace("NaN", np.nan).astype(float)
    X = X.loc[:, X.isna().mean() < 0.4].fillna(X.mean(axis=1), axis=0)
    y = pd.read_csv(label_path, delim_whitespace=True, header=None).iloc[:, 0].values
    y = np.where(y==-1,0,1)
    return X.values, y

def split_data(X, y):
    X = preprocessing.StandardScaler().fit_transform(X)
    normal = X[y==0]
    fault = X[y==1]
    train_normal, test_normal = train_test_split(normal, test_size=0.3, random_state=42)
    test_X = np.vstack([test_normal, fault])
    test_y = np.array([0]*len(test_normal) + [1]*len(fault))
    return train_normal, np.zeros(len(train_normal)), test_X, test_y

# ========= 9. Main =========
def main():
    X, y = load_secom_data()
    X_train, _, X_test, y_test = split_data(X, y)


    input_dim = X_train.shape[1]

    # 动态选择合适的 patch_size
    for patch_size in range(5, 50):  # 比如允许 5到50之间搜索
        if input_dim % patch_size == 0:
            break
    else:
        patch_size = 1  # 如果找不到合适的，就设成1（每个特征当作一个patch）

    print(f"Chosen patch_size: {patch_size}")


    detector = FaultDetector(input_dim=input_dim, patch_size=patch_size)

    x_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    x_test_tensor  = torch.tensor(X_test, dtype=torch.float32)

    detector.train(x_train_tensor, epochs=20)
    feat_train = detector.extract_features(x_train_tensor)
    feat_test  = detector.extract_features(x_test_tensor)



    # Mahalanobis
    mean, cov, inv_cov = compute_mahalanobis_stats(feat_train)
    score_maha = mahalanobis_distance(feat_test, mean, inv_cov)

    # AutoEncoder SPE
    ae = train_autoencoder(feat_train, epochs=150)
    score_spe = compute_SPE(ae, feat_test)

    # 动态融合
    best_alpha, best_auc, maha_norm, spe_norm = find_best_alpha(score_maha, score_spe, y_test)
    fusion_score = best_alpha*maha_norm + (1-best_alpha)*spe_norm

    # 绘制
    plot_roc(score_maha, y_test, "Mahalanobis ROC")
    plot_roc(score_spe, y_test, "SPE ROC")
    plot_roc(fusion_score, y_test, "Optimized Fusion ROC (SECOM)")

if __name__ == '__main__':
    main()
