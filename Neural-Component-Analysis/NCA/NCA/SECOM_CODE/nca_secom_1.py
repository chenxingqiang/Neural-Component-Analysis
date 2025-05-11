import torch
from sklearn import preprocessing
import pandas as pd
import matplotlib
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
matplotlib.use('TkAgg')
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
# MLP Model
class MLP(torch.nn.Module):
    def __init__(self, i, h, o):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(i, h)
        self.bn = torch.nn.BatchNorm1d(h)
        self.act = torch.nn.LeakyReLU(True)
        self.fc2 = torch.nn.Linear(h, o)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

# NCA Class
class NCA:
    def __init__(self, i, h, o, B):
        self.model = MLP(i, h, o)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)
        self.B = B

    def train(self, x):
        self.model.train()
        o = self.model(x)
        o = o.float()
        self.B = self.B.float()
        loss = torch.mean(torch.pow(torch.mm(o, self.B.t()) - x, 2))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        U, _, V = torch.svd(torch.mm(x.t().data, o.data))
        self.B = torch.autograd.Variable(torch.mm(U, V.t()))
        return loss

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            out = self.model(x)
            loss = torch.mean(torch.pow(torch.mm(out, self.B.t()) - x, 2))
            loss_single = torch.mean(torch.pow(torch.mm(out, self.B.t()) - x, 2), dim=1)
        return loss, out, loss_single

# Load SECOM data
def load_secom_data():
    path_to_secom = "F:/0002025_bishe/code/Neural-Component-Analysis-master/Neural-Component-Analysis-master/NCA/NCA/secom_data/SECOM.TXT"
    path_to_secom_labels = "F:/0002025_bishe/code/Neural-Component-Analysis-master/Neural-Component-Analysis-master/NCA/NCA/secom_data/SECOM_labels.TXT"

    features = pd.read_csv(path_to_secom, delim_whitespace=True, header=None)
    labels = pd.read_csv(path_to_secom_labels, delim_whitespace=True, header=None, names=["Label", "Timestamp"])

    features.fillna(features.mean(), inplace=True)
    labels = labels["Label"].astype(int)

    normal_data = features[labels == -1].values
    abnormal_data = features[labels == 1].values

    scaler = preprocessing.StandardScaler()
    normal_data = scaler.fit_transform(normal_data)
    abnormal_data = scaler.transform(abnormal_data)

    return normal_data, abnormal_data

# Compute control limits
def compute_control_limits(spe, quantile=0.95):
    return np.quantile(spe, quantile)

# Plot results
def plot_results(spe_normal, spe_abnormal, t2_normal, t2_abnormal, spe_cl, t2_cl):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # SPE Plot
    #axes[0].plot(spe_normal, label="SPE Normal", color="blue", alpha=0.7)
    axes[0].plot(spe_abnormal, label="SPE Abnormal", color="green", alpha=0.7)
    axes[0].axhline(y=spe_cl, color="red", linestyle="--", label="SPE Control Limit")
    axes[0].set_title("SPE Monitoring")
    axes[0].set_xlabel("Sample Index")
    axes[0].set_ylabel("SPE Value")
    axes[0].legend()

    # T² Plot
    #axes[1].plot(t2_normal, label="T² Normal", color="blue", alpha=0.7)
    axes[1].plot(t2_abnormal, label="T² Abnormal", color="green", alpha=0.7)
    axes[1].axhline(y=t2_cl, color="red", linestyle="--", label="T² Control Limit")
    axes[1].set_title("T² Monitoring")
    axes[1].set_xlabel("Sample Index")
    axes[1].set_ylabel("T² Value")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

# Main function
def main():
    normal_data, abnormal_data = load_secom_data()

    # Train/Test split for normal data
    train_data, test_data = train_test_split(normal_data, test_size=0.3, random_state=42)

    # Convert data to Torch tensors
    train_tensor = torch.tensor(train_data, dtype=torch.float32)
    test_tensor = torch.tensor(test_data, dtype=torch.float32)
    abnormal_tensor = torch.tensor(abnormal_data, dtype=torch.float32)

    # PCA initialization for B
    pca = PCA(n_components=27).fit(train_data)
    B = torch.autograd.Variable(torch.from_numpy(pca.components_.T).float())

    # NCA model
    nca = NCA(train_data.shape[1], 27, 27, B)

    # Train NCA model
    for epoch in range(500):
        train_loss = nca.train(train_tensor)
        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Training Loss: {train_loss:.4f}")

    # Predict for normal and abnormal data
    _, _, spe_normal = nca.predict(test_tensor)
    _, _, spe_abnormal = nca.predict(abnormal_tensor)

    # Compute control limits from normal data
    spe_cl = compute_control_limits(spe_normal.numpy())
    t2_cl = compute_control_limits(spe_normal.numpy())  # Placeholder; replace with T² calculation

    # Visualize results
    plot_results(spe_normal.numpy(), spe_abnormal.numpy(), spe_normal.numpy(), spe_abnormal.numpy(), spe_cl, t2_cl)

if __name__ == "__main__":
    main()