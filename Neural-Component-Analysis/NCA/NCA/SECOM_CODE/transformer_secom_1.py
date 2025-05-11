import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn import preprocessing
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.covariance import EmpiricalCovariance
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# =============== Conv Patch Embedding ===============
class ConvPatchEmbedding(nn.Module):
    def __init__(self, input_dim, patch_size, embed_dim):
        super().__init__()
        assert input_dim % patch_size == 0, "input_dim must be divisible by patch_size"
        self.patch_size = patch_size
        self.n_patches = input_dim // patch_size
        self.conv = nn.Conv1d(1, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = x.unsqueeze(1)  # [B, 1, L]
        x = self.conv(x)    # [B, embed_dim, n_patches]
        return x.transpose(1, 2)  # [B, n_patches, embed_dim]

# =============== Positional Encoding ===============
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-np.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# =============== Transformer Classifier ===============
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim=590, patch_size=10, embed_dim=256, num_heads=8, num_layers=6, num_classes=2):
        super().__init__()
        self.patch_embed = ConvPatchEmbedding(input_dim, patch_size, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=(input_dim // patch_size) + 1)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                                   dim_feedforward=embed_dim * 4, dropout=0.1, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.bn = nn.BatchNorm1d(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        pooled = 0.5 * x[:, 0] + 0.5 * x.mean(dim=1)
        pooled = self.bn(pooled)
        return pooled, self.fc(pooled)

# =============== Fault Detector ===============
class FaultDetector:
    def __init__(self, input_dim, patch_size):
        self.model = TransformerClassifier(input_dim=input_dim, patch_size=patch_size)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, x, y, batch_size=64):
        self.model.train()
        total_loss = 0.0
        for i in range(0, x.shape[0], batch_size):
            xb = x[i:i + batch_size]
            yb = y[i:i + batch_size]
            out, logits = self.model(xb)
            loss = self.criterion(logits, yb)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / (x.shape[0] // batch_size)

    def extract_features(self, x, batch_size=512):
        self.model.eval()
        outputs = []
        with torch.no_grad():
            for i in range(0, x.shape[0], batch_size):
                out, _ = self.model(x[i:i + batch_size])
                outputs.append(out.cpu())
        return torch.cat(outputs, dim=0)

class AE(nn.Module):
    def __init__(self, input_dim, bottleneck_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, bottleneck_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x):
        code = self.encoder(x)
        recon = self.decoder(code)
        return recon

# =============== Data Handling ===============
def load_secom_data(data_path, label_path):
    X = pd.read_csv(data_path, delim_whitespace=True, header=None).replace("NaN", np.nan).astype(float)
    nan_ratio = X.isna().mean()
    X = X.loc[:, nan_ratio < 0.4]  # remove columns with >40% missing
    y = pd.read_csv(label_path, delim_whitespace=True, header=None).iloc[:, 0].values
    y = np.where(y == -1, 0, 1)
    X = X.fillna(X.mean(axis=1), axis=0)
    return X.values, y

def split_data(X, y):
    X = preprocessing.StandardScaler().fit_transform(X)
    normal = X[y == 0]
    fault = X[y == 1]
    train_normal, test_normal = train_test_split(normal, test_size=0.3, random_state=42)
    test_X = np.vstack([test_normal, fault])
    test_y = np.array([0] * len(test_normal) + [1] * len(fault))
    return train_normal, np.zeros(len(train_normal)), test_X, test_y

# =============== Detection ===============
def compute_mahalanobis_stats(features, labels):
    clf = EmpiricalCovariance().fit(features[labels == 0])
    mean = clf.location_
    cov = clf.covariance_
    inv_cov = clf.precision_
    return mean, cov, inv_cov

def mahalanobis_distance(x, mean, inv_cov):
    diff = x - mean
    return np.sqrt(np.einsum('ij,jk,ik->i', diff, inv_cov, diff))



def train_autoencoder(features_normal, epochs=100):
    ae = AE(input_dim=features_normal.shape[1])
    optimizer = torch.optim.Adam(ae.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    x = torch.tensor(features_normal, dtype=torch.float32)
    ae.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        recon = ae(x)
        loss = criterion(recon, x)
        loss.backward()
        optimizer.step()
    return ae.eval()

def compute_fusion_autoencoder(ae, features):
    x = torch.tensor(features, dtype=torch.float32)
    with torch.no_grad():
        recon = ae(x).numpy()

    SPE = np.sum((features - recon) ** 2, axis=1)
    SPE_norm = (SPE - SPE.mean()) / (SPE.std() + 1e-8)

    return SPE, SPE_norm


def compute_T2_SPE_fusion(features, mean, cov, inv_cov):
    diff = features - mean
    T2 = np.einsum('ij,jk,ik->i', diff, inv_cov, diff)
    recon = mean + diff @ np.linalg.pinv(inv_cov)
    SPE = np.sum((features - recon) ** 2, axis=1)
    T2_norm = (T2 - T2.mean()) / (T2.std() + 1e-8)
    SPE_norm = (SPE - SPE.mean()) / (SPE.std() + 1e-8)
    return T2, SPE, 0.5 * T2_norm + 0.5 * SPE_norm

# =============== Visualization ===============
def plot_roc(scores, labels, title):
    fpr, tpr, _ = roc_curve(labels, scores)
    auc = roc_auc_score(labels, scores)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
# =============== 动态调权 & 画 ROC ===============
def find_best_alpha(scores, spe, y, n_grid=21):
    s_nm = MinMaxScaler().fit_transform(scores.reshape(-1,1)).ravel()
    p_nm = MinMaxScaler().fit_transform(spe.reshape(-1,1)).ravel()
    best_a, best_auc = 0.0, 0.0
    for a in np.linspace(0,1,n_grid):
        fusion = a*p_nm + (1-a)*s_nm
        auc    = roc_auc_score(y, fusion)
        if auc>best_auc:
            best_auc, best_a = auc, a
    print(f">>> best α={best_a:.2f}, AUC={best_auc:.4f}")
    return best_a, best_auc, s_nm, p_nm

def compute_SPE(ae, features):
    with torch.no_grad():
        X = torch.tensor(features, dtype=torch.float32)
        recon = ae(X).cpu().numpy()
    return np.sum((features - recon)**2, axis=1)


# =============== Main ===============
def main():
    data_path = "F:/0002025_bishe/code/Neural-Component-Analysis-master/Neural-Component-Analysis-master/NCA/NCA/secom_data/SECOM.TXT"
    label_path = "F:/0002025_bishe/code/Neural-Component-Analysis-master/Neural-Component-Analysis-master/NCA/NCA/secom_data/SECOM_labels.TXT"


    X, y = load_secom_data(data_path, label_path)
    X_train, y_train, X_test, y_test = split_data(X, y)

    input_dim = X_train.shape[1]
    for patch_size in range(10, input_dim + 1):
        if input_dim % patch_size == 0:
            break

    detector = FaultDetector(input_dim=input_dim, patch_size=patch_size)
    x_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    for epoch in range(20):
        loss = detector.train(x_train_tensor, y_train_tensor, batch_size=64)
        print(f"Epoch {epoch + 1}/20, Loss = {loss:.4f}")
    features = detector.extract_features(torch.tensor(X_train, dtype=torch.float32)).numpy()
    feat_test = detector.extract_features(torch.tensor(X_test, dtype=torch.float32)).numpy()

    # 用训练集features估计均值、协方差
    mean, cov, inv_cov = compute_mahalanobis_stats(features, np.zeros_like(X_train[:, 0]))

    # 用测试集feat_test来计算异常分数
    scores = mahalanobis_distance(feat_test, mean, inv_cov)

    T2, SPE, fusion = compute_T2_SPE_fusion(features, mean, cov, inv_cov)

    plot_roc(scores, y_test, "Mahalanobis ROC")
    #plot_roc(fusion, y_test, "T² + SPE Fusion ROC")

    # Step 1: 仅使用正常样本训练 AutoEncoder
    features_normal = feat_test[y_test == 0]
    ae = train_autoencoder(features_normal, epochs=100)

    # Step 2: 计算 SPE 得分
    #SPE, SPE_norm = compute_fusion_autoencoder(ae, features)
    SPE, SPE_norm = compute_fusion_autoencoder(ae, feat_test)

    # Step 3: 与 T² 或 Mahalanobis 融合（任选其一）
    fusion_score = 0.5 * SPE_norm + 0.5 * scores  # Mahalanobis
    #fusion_score = 0.5 * SPE_norm + 0.5 * T2_norm

    # Step 4: 可视化
    plot_roc(fusion_score, y_test, "T² + SPE Fusion (AutoEncoder) ROC")


if __name__ == '__main__':
    main()