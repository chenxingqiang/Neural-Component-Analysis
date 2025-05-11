import numpy as np
import torch
import torch.utils.data
from sklearn import preprocessing
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from joblib import dump, load
import os
import util


class MLP(torch.nn.Module):
    def __init__(self, i, h, o):
        super(MLP, self).__init__()
        self.h = torch.nn.Linear(i, h)
        self.a = torch.nn.Tanh()
        self.o = torch.nn.Linear(h, o)

    def forward(self, x):
        x = self.o(self.a(self.h(x)))
        return x


def get_train_data():
    train_data, _ = util.read_data(error=0, is_train=True)
    train_data = preprocessing.StandardScaler().fit_transform(train_data)
    return train_data


def get_test_data():
    test_data = []
    for i in range(22):
        data, _ = util.read_data(error=i, is_train=False)
        test_data.append(data)
    test_data = np.concatenate(test_data)
    train_data, _ = util.read_data(error=0, is_train=True)
    scaler = preprocessing.StandardScaler().fit(train_data)
    test_data = scaler.transform(test_data)
    return test_data


def train(model, optimizer, train_loader):
    """
    Training loop for a single epoch.
    Arguments:
    - model: The neural network model (MLP).
    - optimizer: The optimizer to use for parameter updates (e.g., SGD).
    - train_loader: The DataLoader that loads batches of training data.

    Returns:
    - loss: The average loss over the current epoch.
    """
    model.train()  # Set the model to training mode
    total_loss = 0  # Variable to accumulate the loss for the epoch
    for data in train_loader:  # Iterate over batches of training data
        x = torch.autograd.Variable(data.cuda())  # Move data to GPU and wrap it in a Variable
        y = model(x)  # Pass the input through the model to get predictions

        # Compute the loss (Mean Squared Error loss)
        loss = torch.mean(torch.pow(x - y, 2))
        total_loss += loss.item()  # Accumulate the loss for the epoch

        optimizer.zero_grad()  # Zero the gradients before backward pass
        loss.backward()  # Perform backpropagation to compute gradients
        optimizer.step()  # Update the model's parameters based on the gradients

    # Return the average loss for the epoch
    return total_loss / len(train_loader)


def train_and_save_models():
    # Load datasets
    train_data = get_train_data()
    test_data = get_test_data()

    # Define and train NCA
    nca = NeighborhoodComponentsAnalysis(random_state=42)
    train_reduced = nca.fit_transform(train_data, np.zeros(train_data.shape[0]))

    # Train PyTorch MLP
    train_dataset = torch.from_numpy(train_reduced).float().cuda()
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = MLP(train_reduced.shape[1], 27, train_reduced.shape[1])
    model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)

    for i in range(500):  # Reduced epochs for example
        loss = train(model, optimizer, train_loader)  # Call the train function

    # Save models
    torch.save(model.state_dict(), 'mlp_model.pth')
    dump(nca, 'nca_model.joblib')



def predict_with_models(data):
    # Load saved models
    nca = load('nca_model.joblib')
    model = MLP(data.shape[1], 27, data.shape[1])  # Adjust dimensions to match trained model
    model.load_state_dict(torch.load('mlp_model.pth'))
    model.eval().cuda()

    # Process data
    reduced_data = nca.transform(data)
    dataset = torch.from_numpy(reduced_data).float().cuda()
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    # Get predictions
    predictions = validate(model, data_loader)
    return predictions


def validate(model, val_loader):
    model.eval()
    predictions = []
    for data in val_loader:
        x = torch.autograd.Variable(data.cuda())
        y = model(x)
        predictions.append(y.data.cpu().numpy())
    return np.vstack(predictions)


if __name__ == '__main__':
    # Ensure the .npy files exist before running
    train_data = get_train_data()
    test_data = get_test_data()
    if not os.path.exists("train_data") or not os.path.exists("test_data"):
        raise FileNotFoundError("Ensure 'd00.dat' and 'd00_te.dat' exist in the working directory.")

    # Train and save models
    train_and_save_models()

