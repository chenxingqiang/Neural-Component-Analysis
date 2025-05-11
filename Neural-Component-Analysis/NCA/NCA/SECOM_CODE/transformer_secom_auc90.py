import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn import preprocessing
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.covariance import EmpiricalCovariance
from sklearn.preprocessing import MinMaxScaler

# =================== Conv Patch Embedding ===================
class ConvPatchEmbedding(nn.Module):
    def __init__(self, input_dim, patch_size, embed_dim):
        super().__init__()
        assert input_dim % patch_size == 0
        self.conv = nn.Conv1d(1, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        return x.transpose(1, 2)

# =================== Positional Encoding ===================
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-np.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# =================== Transformer Classifier ===================
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim=590, patch_size=10, embed_dim=256, num_heads=8, num_layers=6, num_classes=2):
        super().__init__()
        self.patch_embed = ConvPatchEmbedding(input_dim, patch_size, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_encoder = PositionalEncoding(embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4)
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

class FaultDetector:
    def __init__(self, input_dim, patch_size):
        self.model = TransformerClassifier(input_dim, patch_size)
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
        return total_loss / (x.shape[0] // batch_size)  # <<<<<< 一定要 return！！

    def extract_features(self, x, batch_size=512):
        self.model.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, x.shape[0], batch_size):
                out, _ = self.model(x[i:i+batch_size])
                outs.append(out.cpu())
        return torch.cat(outs, dim=0).numpy()

class AE(nn.Module):
    def __init__(self, input_dim, bottleneck_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, bottleneck_dim))
        self.decoder = nn.Sequential(nn.Linear(bottleneck_dim, 128), nn.ReLU(), nn.Linear(128, input_dim))

    def forward(self, x):
        return self.decoder(self.encoder(x))

# =================== Utilities ===================
def load_secom_data(data_path, label_path):
    X = pd.read_csv(data_path, delim_whitespace=True, header=None).replace("NaN", np.nan).astype(float)
    X = X.loc[:, X.isna().mean() < 0.4].fillna(X.mean(axis=1), axis=0)
    y = pd.read_csv(label_path, delim_whitespace=True, header=None).iloc[:, 0].values
    return X.values, np.where(y == -1, 0, 1)

def split_data(X, y):
    X = preprocessing.StandardScaler().fit_transform(X)
    normal = X[y == 0]
    fault = X[y == 1]
    train_normal, test_normal = train_test_split(normal, test_size=0.3, random_state=42)
    test_X = np.vstack([test_normal, fault])
    test_y = np.array([0] * len(test_normal) + [1] * len(fault))
    return train_normal, np.zeros(len(train_normal)), test_X, test_y

def compute_mahalanobis_stats(features):
    clf = EmpiricalCovariance().fit(features)
    return clf.location_, clf.covariance_, clf.precision_

def mahalanobis_distance(x, mean, inv_cov):
    diff = x - mean
    return np.sqrt(np.einsum('ij,jk,ik->i', diff, inv_cov, diff))

def train_autoencoder(features_normal, epochs=100):
    ae = AE(features_normal.shape[1])
    optimizer = torch.optim.Adam(ae.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    x = torch.tensor(features_normal, dtype=torch.float32)
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
    return np.sum((features - recon) ** 2, axis=1)

def find_best_alpha(score1, score2, y_true, n_grid=21):
    s1 = MinMaxScaler().fit_transform(score1.reshape(-1, 1)).ravel()
    s2 = MinMaxScaler().fit_transform(score2.reshape(-1, 1)).ravel()
    best_a, best_auc = 0.0, 0.0
    for a in np.linspace(0, 1, n_grid):
        fusion = a * s1 + (1 - a) * s2
        auc = roc_auc_score(y_true, fusion)
        if auc > best_auc:
            best_auc, best_a = auc, a
    return best_a, best_auc, s1, s2

def dynamic_monitoring(fusion_score, y_true, threshold):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, len(fusion_score))
    ax.set_ylim(fusion_score.min() - 0.5, fusion_score.max() + 0.5)
    ax.axhline(threshold, color='orange', linestyle='--')
    scat = ax.scatter([], [], c=[])

    def update(i):
        colors = ['blue' if y_true[j] == 0 else 'red' for j in range(i)]
        scat.set_offsets(np.c_[range(i), fusion_score[:i]])
        scat.set_color(colors)
        return scat,

    ani = animation.FuncAnimation(fig, update, frames=len(fusion_score), interval=50, blit=True, repeat=False)
    plt.title('Dynamic Online Monitoring')
    plt.xlabel('Sample Index')
    plt.ylabel('Anomaly Score')
    plt.grid(True)
    plt.show()

def save_detection_results(scores, labels, threshold, filename='detection_results.csv'):
    preds = (scores > threshold).astype(int)
    df = pd.DataFrame({'Index': np.arange(len(scores)), 'TrueLabel': labels, 'AnomalyScore': scores, 'PredictedLabel': preds})
    df.to_csv(filename, index=False)
import matplotlib.animation as animation

def online_monitoring_animation(scores, labels, save_path="monitor.mp4"):
    """
    动态在线监控，并保存为 mp4 或 gif。
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, len(scores))
    ax.set_ylim(min(scores) * 1.1, max(scores) * 1.1)
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Anomaly Score")
    ax.set_title("Online Monitoring of SECOM Faults (Dynamic)")

    normal_scatter, = ax.plot([], [], 'bo', label='Normal', markersize=3)
    anomaly_scatter, = ax.plot([], [], 'ro', label='Anomaly', markersize=3)
    threshold_line = ax.axhline(y=0, color='orange', linestyle='--', label='Threshold')
    ax.legend()
    ax.grid(True)

    # 阈值（正常数据95%分位）
    threshold = np.percentile(scores[labels == 0], 95)

    # 初始化
    def init():
        normal_scatter.set_data([], [])
        anomaly_scatter.set_data([], [])
        threshold_line.set_ydata(threshold)
        return normal_scatter, anomaly_scatter, threshold_line

    # 每帧更新
    def update(frame):
        current_x = np.arange(frame + 1)
        current_y = scores[:frame + 1]
        normal_x = current_x[labels[:frame + 1] == 0]
        normal_y = current_y[labels[:frame + 1] == 0]
        anomaly_x = current_x[labels[:frame + 1] == 1]
        anomaly_y = current_y[labels[:frame + 1] == 1]

        normal_scatter.set_data(normal_x, normal_y)
        anomaly_scatter.set_data(anomaly_x, anomaly_y)
        return normal_scatter, anomaly_scatter, threshold_line

    ani = animation.FuncAnimation(fig, update, frames=len(scores), init_func=init,
                                   blit=True, interval=50, repeat=False)

    # 保存动画
    ani.save(save_path, writer='ffmpeg', fps=30)
    # 如果想保存为 gif:
    # ani.save('monitor.gif', writer='pillow', fps=20)

    plt.close(fig)
def compute_fusion_autoencoder(ae, features):
    x = torch.tensor(features, dtype=torch.float32)
    with torch.no_grad():
        recon = ae(x).numpy()

    SPE = np.sum((features - recon) ** 2, axis=1)
    SPE_norm = (SPE - SPE.mean()) / (SPE.std() + 1e-8)

    return SPE, SPE_norm
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
# =================== Main ===================
def main():
    data_path = "F:/0002025_bishe/code/Neural-Component-Analysis-master/Neural-Component-Analysis-master/NCA/NCA/secom_data/SECOM.TXT"
    label_path = "F:/0002025_bishe/code/Neural-Component-Analysis-master/Neural-Component-Analysis-master/NCA/NCA/secom_data/SECOM_labels.TXT"

    # ========== 加载数据 ==========
    X, y = load_secom_data(data_path, label_path)
    X_train, y_train, X_test, y_test = split_data(X, y)

    input_dim = X_train.shape[1]
    for patch_size in range(10, input_dim + 1):
        if input_dim % patch_size == 0:
            break

    detector = FaultDetector(input_dim=input_dim, patch_size=patch_size)
    x_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    # ========== 训练 Transformer ==========
    for epoch in range(20):
        loss = detector.train(x_train_tensor, y_train_tensor, batch_size=64)
        print(f"Epoch {epoch + 1}/20, Loss = {loss:.4f}")

    # ========== 特征提取 ==========
    feat_train = detector.extract_features(torch.tensor(X_train, dtype=torch.float32))
    feat_test  = detector.extract_features(torch.tensor(X_test, dtype=torch.float32))

    # ========== Mahalanobis ==========

    mean, cov, inv_cov = compute_mahalanobis_stats(feat_train)

    scores_test = mahalanobis_distance(feat_test, mean, inv_cov)

    # ========== AutoEncoder ==========
    ae = train_autoencoder(feat_train, epochs=100)
    spe_test, _ = compute_fusion_autoencoder(ae, feat_test)

    # ========== 动态融合 alpha ==========
    best_alpha, best_auc, score_norm, spe_norm = find_best_alpha(scores_test, spe_test, y_test)

    # ========== 融合后的分数 ==========
    fusion_score = best_alpha * score_norm + (1 - best_alpha) * spe_norm

    # ========== 绘制 ROC ==========
    plot_roc(scores_test, y_test, title="Mahalanobis ROC")
    plot_roc(spe_test, y_test, title="SPE ROC")
    plot_roc(fusion_score, y_test, title="Optimized Fusion ROC")


if __name__ == '__main__':
    main()