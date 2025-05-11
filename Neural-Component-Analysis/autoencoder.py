import datetime
import numpy as np
import torch
import torch.utils.data
from sklearn import preprocessing
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from joblib import dump, load
import util
import os


class MLP(torch.nn.Module):
    def __init__(self, i, h, o):
        super(MLP, self).__init__()
        self.h = torch.nn.Linear(i, h)
        self.a = torch.nn.Tanh()
        self.o = torch.nn.Linear(h, o)

    def forward(self, x):
        x = self.o(self.a(self.h(x)))
        return x

    def predict(self, data_loader):
        pred = []
        for data in data_loader:
            x = torch.autograd.Variable(data)
            if torch.cuda.is_available():
                x = x.cuda()
            o = self.h(x)
            pred.append(o.data.cpu().numpy())
        return np.vstack(pred)


class Metirc(object):

    def __init__(self):
        self._sum = 0
        self._count = 0

    def update(self, pred, target):
        self._sum += np.sum(np.mean(np.power(target - pred, 2), axis=1))
        self._count += pred.shape[0]

    def get(self):
        return self._sum / self._count


def train(model, optimizer, train_loader, device):
    model.train()
    l2norm = Metirc()
    for data in train_loader:
        x = torch.autograd.Variable(data.to(device))
        y = model(x)

        loss = torch.mean(torch.pow(x - y, 2))
        l2norm.update(x.data.cpu().numpy(), y.data.cpu().numpy())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return l2norm.get()


def validate(model, val_loader, device):
    model.eval()
    l2norm = Metirc()
    for data in val_loader:
        x = torch.autograd.Variable(data.to(device))
        y = model(x)
        l2norm.update(y.data.cpu().numpy(), x.data.cpu().numpy())
    return l2norm.get()


def get_mock_data(num_samples=500, num_features=52):
    """Generate mock data when real data files are not available"""
    print("WARNING: Using mock data as real data files were not found")
    return np.random.randn(num_samples, num_features).astype(np.float32)


def get_train_data():
    try:
        train_data, _ = util.read_data(error=0, is_train=True)
        train_data = preprocessing.StandardScaler().fit_transform(train_data)
    except FileNotFoundError:
        print("Training data file not found. Using mock data.")
        train_data = get_mock_data()
    return train_data


def get_test_data():
    try:
        test_data = []
        for i in range(22):
            data, _ = util.read_data(error=i, is_train=False)
            test_data.append(data)
        test_data = np.concatenate(test_data)
        train_data, _ = util.read_data(error=0, is_train=True)
        scaler = preprocessing.StandardScaler().fit(train_data)
        test_data = scaler.transform(test_data)
    except FileNotFoundError:
        print("Test data file not found. Using mock data.")
        test_data = get_mock_data(num_samples=200)
    return test_data


def check_data_dir():
    """Check if data directory exists and create it if not"""
    os.makedirs('./data/train', exist_ok=True)
    print("Note: Data directory './data/train/' has been created. "
          "You need to add data files for real results.")


def main():
    # Check data directory
    check_data_dir()
    
    # Set device to GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_data = get_train_data()
    test_data = get_test_data()
    
    train_dataset = torch.from_numpy(train_data).float()
    test_dataset = torch.from_numpy(test_data).float()

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = MLP(52, 27, 52)
    model.to(device)
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)

    for i in range(500):
        train_acc = train(model, optimizer, train_loader, device)
        test_acc = validate(model, test_loader, device)
        print('{}\tepoch = {}\ttrain accuracy: {:0.3f}\ttest accuracy: {:0.3f}' \
              .format(datetime.datetime.now(), i, train_acc, test_acc))
    
    return model


if __name__ == '__main__':
    try:
        model = main()
        
        # Train a classifier on top of the autoencoder features
        train_data = get_train_data()
        
        try:
            train_labels, _ = util.read_data(error=0, is_train=True, return_label=True)
        except (FileNotFoundError, TypeError):
            # If no data or return_label not supported
            print("Using mock labels for classifier")
            train_labels = np.zeros(len(train_data))
        
        # Create pipeline with NCA and MLP classifier
        nca = NeighborhoodComponentsAnalysis(n_components=10, random_state=42)
        nn = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42)
        pipeline = Pipeline([('nca', nca), ('mlp', nn)])
        
        # Save the model
        dump(pipeline, 'tep_nca_mlp_model.joblib')
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure all dependencies are installed and data files are available.")
