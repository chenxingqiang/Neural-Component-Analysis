import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import os

class TransformerClassifier(torch.nn.Module):
    def __init__(self, input_dim, embed_dim=128, num_heads=8, num_layers=6, num_classes=22):
        super(TransformerClassifier, self).__init__()
        self.embed = torch.nn.Linear(input_dim, embed_dim)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=512, dropout=0.1, activation='gelu')
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = torch.nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B, L = x.shape
        x = self.embed(x).unsqueeze(1)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return x, self.fc(x)

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
                batch = x[i:i+batch_size]
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
    normal = np.fromfile(os.path.join(base_path, 'd00_te.dat'), dtype=np.float32, sep='  ').reshape(-1, 52)
    test_X.append(normal)
    test_y.extend([0] * len(normal))


    i=7
    path = os.path.join(base_path, f'd{i:02d}_te.dat')
    fault = np.fromfile(path, dtype=np.float32, sep='  ').reshape(-1, 52)
    test_X.append(fault)
    test_y.extend([i] * len(fault))

    test_X = np.vstack(test_X)
    test_X = preprocessing.StandardScaler().fit_transform(test_X)
    test_y = np.array(test_y)
    return test_X, test_y

def compute_mahalanobis_stats(features, labels):
    mean = features[labels == 0].mean(axis=0)
    cov = np.cov(features[labels == 0].T)
    cov += 1e-3 * np.eye(cov.shape[0])
    inv_cov = np.linalg.pinv(cov)
    return mean, cov, inv_cov

def mahalanobis_distance(x, mean, inv_cov):
    diff = x - mean
    return np.sqrt(np.einsum('ij,jk,ik->i', diff, inv_cov, diff))

def compute_T2_SPE_fusion(features, mean, cov, inv_cov):
    diff = features - mean
    T2 = np.einsum('ij,jk,ik->i', diff, inv_cov, diff)
    recon = mean + diff @ np.linalg.pinv(inv_cov)
    SPE = np.sum((features - recon) ** 2, axis=1)
    T2_norm = (T2 - T2.mean()) / (T2.std() + 1e-8)
    SPE_norm = (SPE - SPE.mean()) / (SPE.std() + 1e-8)
    fused_score = 0.5 * T2_norm + 0.5 * SPE_norm
    return T2, SPE, fused_score

def plot_fusion_roc(fused_score, labels, title="Fused ROC (T² + SPE)"):
    y_true = (labels != 0).astype(int)
    fpr, tpr, thresholds = roc_curve(y_true, fused_score)
    auc = roc_auc_score(y_true, fused_score)
    best_idx = np.argmax(tpr - fpr)
    best_thresh = thresholds[best_idx]
    print(f"[{title}] AUC = {auc:.4f}, Best threshold = {best_thresh:.4f}")
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def visualize_tsne(features, labels):
    tsne = TSNE(n_components=2, random_state=42, perplexity=50, n_iter=1000)
    reduced = tsne.fit_transform(features)
    plt.figure(figsize=(10, 7))
    for cls in np.unique(labels):
        idx = labels == cls
        plt.scatter(reduced[idx, 0], reduced[idx, 1], s=5, label=f"Class {cls}")
    plt.legend(markerscale=2)
    plt.title("Transformer Feature Embedding (t-SNE)")
    plt.grid(True)
    plt.show()

def plot_mahalanobis_hist(scores, labels):
    plt.figure()
    plt.hist(scores[labels == 0], bins=100, alpha=0.6, label='Normal')
    plt.hist(scores[labels != 0], bins=100, alpha=0.6, label='Fault')
    plt.title("Mahalanobis Novelty Score Distribution")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_roc_curve(scores, labels):
    y_true = (labels != 0).astype(int)
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    auc = roc_auc_score(y_true, scores)
    best_idx = np.argmax(tpr - fpr)
    best_thresh = thresholds[best_idx]
    print(f"AUC = {auc:.4f}, Best threshold = {best_thresh:.4f}")
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Mahalanobis Detection")
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    X_train, y_train = get_train_data()
    X_test, y_test = get_test_data()
    detector = FaultDetector(input_dim=X_train.shape[1])
    x_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    for epoch in range(10):
        loss = detector.train(x_train_tensor, y_train_tensor)
        print(f"Epoch {epoch + 1}/10, Loss: {loss:.4f}")

    features = detector.extract_features(torch.tensor(X_test, dtype=torch.float32)).numpy()
    mean, cov, inv_cov = compute_mahalanobis_stats(features, y_test)
    scores = mahalanobis_distance(features, mean, inv_cov)

    visualize_tsne(features, y_test)
    plot_mahalanobis_hist(scores, y_test)
    plot_roc_curve(scores, y_test)

    # --- 新增：T2 + SPE 融合检测 ---
    T2, SPE, fused_score = compute_T2_SPE_fusion(features, mean, cov, inv_cov)
    plot_fusion_roc(fused_score, y_test)

if __name__ == '__main__':
    main()
