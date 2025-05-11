import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, confusion_matrix
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

    def predict_classes(self, x):
        self.model.eval()
        with torch.no_grad():
            _, logits = self.model(x)
            preds = torch.argmax(logits, dim=1)
        return preds.cpu().numpy()

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

    for i in range(1, 22):
        path = os.path.join(base_path, f'd{i:02d}_te.dat')
        fault = np.fromfile(path, dtype=np.float32, sep='  ').reshape(-1, 52)
        test_X.append(fault)
        test_y.extend([i] * len(fault))

    test_X = np.vstack(test_X)
    test_X = preprocessing.StandardScaler().fit_transform(test_X)
    test_y = np.array(test_y)
    return test_X, test_y

def compute_statistics(features):
    mean = features.mean(axis=0)
    cov = np.cov(features.T)
    cov += 1e-6 * np.eye(cov.shape[0])
    inv_cov = np.linalg.pinv(cov)
    return mean, cov, inv_cov

def compute_T2_SPE(features, mean, inv_cov):
    diff = features - mean
    T2 = np.einsum('ij,jk,ik->i', diff, inv_cov, diff)
    recon = mean + diff @ np.linalg.pinv(inv_cov)
    SPE = np.sum((features - recon) ** 2, axis=1)
    return T2, SPE

def plot_tSNE(features, labels):
    tsne = TSNE(n_components=2, random_state=42, perplexity=50)
    reduced = tsne.fit_transform(features)
    plt.figure(figsize=(10, 7))
    for cls in np.unique(labels):
        idx = labels == cls
        plt.scatter(reduced[idx, 0], reduced[idx, 1], s=5, label=f"Class {cls}")
    plt.title("Transformer Feature Embedding (t-SNE)")
    plt.legend(markerscale=2)
    plt.grid(True)
    plt.show()

def evaluate_classification(true_labels, pred_labels):
    print("\nConfusion Matrix:")
    print(confusion_matrix(true_labels, pred_labels))
    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels, digits=4))

def evaluate_per_class_auc(fusion_score, labels):
    class_ids = np.unique(labels)
    aucs = {}
    for cid in class_ids:
        if cid == 0:
            continue
        binary_labels = (labels == cid).astype(int)
        auc = roc_auc_score(binary_labels, fusion_score)
        aucs[f"Class {cid:02d}"] = auc
    print("\nPer-Class AUC (Fused Score):")
    for k, v in aucs.items():
        print(f"{k}: AUC = {v:.4f}")

def main():
    X_train, y_train = get_train_data()
    X_test, y_test = get_test_data()
    detector = FaultDetector(input_dim=52)

    x_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    x_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    for epoch in range(10):
        loss = detector.train(x_train_tensor, y_train_tensor)
        print(f"Epoch {epoch + 1}/10, Loss: {loss:.4f}")

    features_train = detector.extract_features(x_train_tensor).numpy()
    features_test = detector.extract_features(x_test_tensor).numpy()

    mean, cov, inv_cov = compute_statistics(features_train)
    T2, SPE = compute_T2_SPE(features_test, mean, inv_cov)

    T2_norm = (T2 - np.mean(T2)) / np.std(T2)
    SPE_norm = (SPE - np.mean(SPE)) / np.std(SPE)
    fusion_score = T2_norm + SPE_norm

    y_binary = (y_test != 0).astype(int)
    fpr, tpr, _ = roc_curve(y_binary, fusion_score)
    auc = roc_auc_score(y_binary, fusion_score)

    print(f"Fusion Score AUC = {auc:.4f}")
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - T² + SPE Fusion Detection")
    plt.legend()
    plt.grid(True)
    plt.show()

    # t-SNE Visualization
    plot_tSNE(features_test, y_test)

    # Classification Evaluation
    pred_labels = detector.predict_classes(x_test_tensor)
    evaluate_classification(y_test, pred_labels)

    # Per-Class AUC
    evaluate_per_class_auc(fusion_score, y_test)

if __name__ == '__main__':
    main()
