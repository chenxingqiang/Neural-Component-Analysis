# TEP Fault Detection using Transformer + Mahalanobis Distance (Rewritten Structure)
# Based on: "A transformer-based approach for novel fault detection and fault classification"

from datetime import datetime
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn import preprocessing
import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_curve, roc_auc_score

class TransformerClassifier(torch.nn.Module):
    def __init__(self, input_dim, embed_dim=128, num_heads=8, num_layers=6, num_classes=22):
        super(TransformerClassifier, self).__init__()
        self.embed = torch.nn.Linear(input_dim, embed_dim)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=512, dropout=0.1, activation='gelu')
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = torch.nn.Parameter(torch.randn(1, 2, embed_dim))
        self.fc = torch.nn.Linear(embed_dim, num_classes)

    def forward(self, x):

        B, L = x.shape
        x = self.embed(x).unsqueeze(1)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1) + self.pos_embed[:, :x.shape[1], :]

        # 替换 forward 返回部分：
        x = self.transformer(x)
        return x.mean(dim=1), self.fc(x.mean(dim=1))



class FaultDetector:
    def __init__(self, input_dim=52, embed_dim=128, num_heads=8, num_layers=6):
        self.model = TransformerClassifier(input_dim, embed_dim, num_heads, num_layers)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = torch.nn.CrossEntropyLoss()

    def train(self, x, y):
        self.model.train()
        out, logits = self.model(x)
        loss = self.criterion(logits, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()



    def extract_features(self, x, batch_size=512):
        self.model.eval()
        outputs = []
        with torch.no_grad():
            for i in range(0, x.shape[0], batch_size):
                batch = x[i:i + batch_size]
                out, _ = self.model(batch)
                outputs.append(out.cpu())
        return torch.cat(outputs, dim=0)


def get_train_data():
    train_path = 'F:/0002025_bishe/code/Neural-Component-Analysis-master/Neural-Component-Analysis-master/NCA/NCA/data/d00.dat'
    X = np.fromfile(train_path, dtype=np.float32, sep='   ')
    X = X.reshape(-1, 52)
    X = preprocessing.StandardScaler().fit_transform(X)
    y = np.zeros(X.shape[0], dtype=int)
    return X, y

def get_test_data():
    base_path = 'F:/0002025_bishe/code/Neural-Component-Analysis-master/Neural-Component-Analysis-master/NCA/NCA/data'
    test_X, test_y = [], []

    # 加载正常测试数据 d00_te.dat（全部为正常）
    normal = np.fromfile(os.path.join(base_path, 'd00_te.dat'), dtype=np.float32, sep='  ').reshape(-1, 52)
    test_X.append(normal)
    test_y.extend([0] * len(normal))  # 全部标记为正常

    # 加载故障测试数据 d01_te.dat ~ d21_te.dat（前160为正常，后800为故障）
    for i in range(1, 22):
        path = os.path.join(base_path, f'd{i:02d}_te.dat')
        fault = np.fromfile(path, dtype=np.float32, sep='  ').reshape(-1, 52)

        test_X.append(fault)

        # 标注：前160条为正常（0），后800条为对应故障类型 i
        test_y.extend([0] * 160 + [i] * (fault.shape[0] - 160))

    test_X = np.vstack(test_X)
    test_X = preprocessing.StandardScaler().fit_transform(test_X)
    test_y = np.array(test_y)
    return test_X, test_y

def compute_mahalanobis_stats(features, labels):
    mean = features[labels == 0].mean(axis=0)

    cov = np.cov(features[labels == 0].T)
    cov += 1e-3 * np.eye(cov.shape[0])

    inv_cov = np.linalg.pinv(cov)
    return mean, inv_cov

def mahalanobis_distance(x, mean, inv_cov):
    diff = x - mean
    return np.sqrt(np.einsum('ij,jk,ik->i', diff, inv_cov, diff))

def visualize_tsne(features, labels):
    tsne = TSNE(n_components=2, random_state=42)
    reduced = tsne.fit_transform(features)
    plt.figure(figsize=(8, 6))
    for cls in np.unique(labels):
        idx = labels == cls
        plt.scatter(reduced[idx, 0], reduced[idx, 1], s=10, label=f"Class {cls}")
    plt.legend()
    plt.title("Transformer Feature Embedding (t-SNE)")
    plt.show()

def plot_score_line(scores, labels):
    plt.figure(figsize=(12, 4))
    plt.plot(scores, label='Mahalanobis Score')
    plt.axhline(np.percentile(scores[labels == 0], 99), color='r', linestyle='--', label='99% Threshold (Normal)')
    plt.title("Mahalanobis Score Over Time")
    plt.xlabel("Sample Index")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    X_train, y_train = get_train_data()
    X_test, y_test = get_test_data()
    detector = FaultDetector(input_dim=X_train.shape[1])
    x_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    print(X_test.shape)  # 应该是 [N, 52]

    for epoch in range(10):
        loss = detector.train(x_train_tensor, y_train_tensor)
        print(f"Epoch {epoch + 1}/10, Loss: {loss:.4f}")

    features = detector.extract_features(torch.tensor(X_test, dtype=torch.float32)).numpy()
    mean, inv_cov = compute_mahalanobis_stats(features, y_test)
    scores = mahalanobis_distance(features, mean, inv_cov)

    visualize_tsne(features, y_test)
    plt.figure()
    plt.hist(scores[y_test == 0], bins=100, alpha=0.6, label='Normal')
    plt.hist(scores[y_test != 0], bins=100, alpha=0.6, label='Fault')
    plt.title("Mahalanobis Novelty Score Distribution")
    plt.legend()
    plt.show()



    y_true = (y_test != 0).astype(int)  # 0 表示正常，1 表示故障
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    auc = roc_auc_score(y_true, scores)
    print(f"AUC = {auc:.4f}")

    # 找最佳阈值
    best_idx = np.argmax(tpr - fpr)
    best_thresh = thresholds[best_idx]
    print(f"Best threshold = {best_thresh:.4f}")

    # 绘制 ROC 曲线
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], '--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Mahalanobis Novelty Detection")
    plt.legend()
    plt.grid()
    plt.show()


    plot_score_line(scores, y_test)


if __name__ == '__main__':
    main()
