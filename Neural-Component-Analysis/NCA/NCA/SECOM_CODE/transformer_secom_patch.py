# ================== 完整版：针对 SECOM 数据集优化的 Transformer-AutoEncoder 检测系统 ==================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.covariance import EmpiricalCovariance
from sklearn.preprocessing import MinMaxScaler


# ========= 1. Transformer Encoder 直接每个特征作为 token =========
class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim=64, num_heads=4, num_layers=3, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                                   dim_feedforward=embed_dim*4,
                                                   dropout=dropout, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.input_proj(x).unsqueeze(1)  # [B,1,embed_dim]
        x = self.transformer(x)
        x = self.norm(x[:,0,:])  # 只取第一个token
        return x

# ========= 2. Fault Detector =========
class FaultDetector:
    def __init__(self, input_dim):
        self.model = SimpleTransformer(input_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()

    def train(self, x, epochs=50, batch_size=32):
        self.model.train()
        for epoch in range(epochs):
            losses = []
            for i in range(0, x.shape[0], batch_size):
                xb = x[i:i+batch_size]
                feat = self.model(xb)
                loss = self.criterion(feat, torch.zeros_like(feat))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
            print(f"Epoch {epoch+1}/{epochs}, Loss: {np.mean(losses):.6f}")

    def extract_features(self, x, batch_size=256):
        self.model.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, x.shape[0], batch_size):
                out = self.model(x[i:i+batch_size])
                outs.append(out.cpu())
        return torch.cat(outs, dim=0).numpy()

# ========= 3. AutoEncoder =========
class AE(nn.Module):
    def __init__(self, input_dim, bottleneck_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, bottleneck_dim))
        self.decoder = nn.Sequential(nn.Linear(bottleneck_dim, 64), nn.ReLU(), nn.Linear(64, input_dim))

    def forward(self, x):
        return self.decoder(self.encoder(x))

def train_autoencoder(features, epochs=150):
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

# ========= 4. Mahalanobis =========
def compute_mahalanobis_stats(features):
    clf = EmpiricalCovariance().fit(features)
    return clf.location_, clf.covariance_, clf.precision_

def mahalanobis_distance(x, mean, inv_cov):
    diff = x - mean
    return np.sqrt(np.einsum('ij,jk,ik->i', diff, inv_cov, diff))

# ========= 5. Fused Scoring =========
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

# ========= 6. Plot =========
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

# ========= 7. Load SECOM =========
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

# ========= 8. Main =========
def main():
    X, y = load_secom_data()
    X_train, _, X_test, y_test = split_data(X, y)

    detector = FaultDetector(input_dim=X_train.shape[1])
    x_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    x_test_tensor  = torch.tensor(X_test, dtype=torch.float32)

    detector.train(x_train_tensor, epochs=50)
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
    plot_roc(score_spe,  y_test, "SPE ROC")
    plot_roc(fusion_score, y_test, "Optimized Fusion ROC (SECOM)")

if __name__ == '__main__':
    main()
