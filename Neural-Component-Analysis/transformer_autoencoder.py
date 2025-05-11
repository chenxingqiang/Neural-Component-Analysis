import datetime
import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
from sklearn import preprocessing
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from joblib import dump, load
import os
import util
import matplotlib.pyplot as plt


class PositionalEncoding(nn.Module):
    """
    Positional encoding for the input features
    """
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        # Removed the fixed buffer and made it dynamic to handle variable sizes
        self.d_model = d_model
        
    def _get_positional_encoding(self, seq_len):
        pe = torch.zeros(seq_len, self.d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-np.log(10000.0) / self.d_model))
        
        # Handle even/odd dimensions properly
        if self.d_model % 2 == 0:
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            # For odd dimensions, handle carefully
            pe[:, 0::2] = torch.sin(position * div_term[:self.d_model//2 + 1])
            pe[:, 1::2] = torch.cos(position * div_term[:self.d_model//2])
        
        pe = pe.unsqueeze(0)
        return pe

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        pe = self._get_positional_encoding(x.size(1))
        # Ensure positional encoding is on the same device as input
        pe = pe.to(x.device)
        return x + pe[:, :x.size(1), :x.size(2)]


class TransformerAutoencoder(nn.Module):
    """
    Transformer-based autoencoder
    The encoder uses transformer encoder layers to create a compressed representation
    The decoder uses a simple MLP to reconstruct the original input
    """
    def __init__(self, input_dim, hidden_dim, nhead=4, num_layers=2, dropout=0.1):
        super(TransformerAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Input linear layer to match transformer dimensions
        self.input_linear = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
        # Ensure nhead divides hidden_dim evenly
        self._adjust_nhead(hidden_dim, nhead)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=self.nhead, 
            dim_feedforward=hidden_dim*4, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Output linear layer to reconstruct input
        self.output_linear = nn.Linear(hidden_dim, input_dim)
        
        # Pooling layer to get fixed-size representation for classification
        self.pooling = nn.AdaptiveAvgPool1d(1)
    
    def _adjust_nhead(self, hidden_dim, nhead):
        """Ensure number of heads divides hidden dimension evenly"""
        if hidden_dim % nhead != 0:
            # Find largest divisor of hidden_dim less than nhead
            for i in range(nhead, 0, -1):
                if hidden_dim % i == 0:
                    self.nhead = i
                    print(f"Adjusted number of attention heads to {i} to match hidden dimension {hidden_dim}")
                    return
        self.nhead = nhead
    
    def forward(self, x):
        # x shape: [batch_size, input_dim]
        
        # Reshape to [batch_size, seq_len=1, input_dim]
        x = x.unsqueeze(1)
        
        # Project to hidden dimension
        x = self.input_linear(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Pass through transformer encoder
        x = self.transformer_encoder(x)
        
        # Get encoder representation (for feature extraction)
        encoder_features = x
        
        # Reconstruct the input
        x = self.output_linear(x)
        
        # Reshape back to [batch_size, input_dim]
        x = x.squeeze(1)
        
        return x, encoder_features
    
    def extract_features(self, x):
        """Extract features for external use (like classification)"""
        # Forward pass but only return the encoder features
        x = x.unsqueeze(1)
        x = self.input_linear(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        
        # Pool across sequence dimension to get a fixed-size representation
        # Shape: [batch_size, hidden_dim, 1]
        pooled = self.pooling(x.transpose(1, 2))
        
        # Shape: [batch_size, hidden_dim]
        return pooled.squeeze(-1)
    
    def predict(self, data_loader, device):
        """Generate predictions for a data loader"""
        self.eval()
        features = []
        with torch.no_grad():
            for data in data_loader:
                data = data.to(device)
                feature = self.extract_features(data)
                features.append(feature.cpu().numpy())
        return np.vstack(features)


class Metric(object):
    """Metric calculator for model evaluation"""
    def __init__(self):
        self._sum = 0
        self._count = 0

    def update(self, pred, target):
        self._sum += np.sum(np.mean(np.power(target - pred, 2), axis=1))
        self._count += pred.shape[0]

    def get(self):
        return self._sum / self._count


def train(model, optimizer, train_loader, device):
    """Train the model for one epoch"""
    model.train()
    l2norm = Metric()
    for data in train_loader:
        data = data.to(device)
        
        # Forward pass
        reconstructed, _ = model(data)
        
        # Compute loss (reconstruction error)
        loss = torch.mean(torch.pow(data - reconstructed, 2))
        
        # Update metrics
        l2norm.update(data.cpu().numpy(), reconstructed.detach().cpu().numpy())
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return l2norm.get()


def validate(model, val_loader, device):
    """Validate the model on a validation set"""
    model.eval()
    l2norm = Metric()
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            reconstructed, _ = model(data)
            l2norm.update(reconstructed.cpu().numpy(), data.cpu().numpy())
    return l2norm.get()


def get_mock_data(num_samples=500, num_features=52):
    """Generate mock data when real data files are not available"""
    print("WARNING: Using mock data as real data files were not found")
    return np.random.randn(num_samples, num_features).astype(np.float32)


def get_train_data():
    """Get training data, generating mock data if real data is unavailable"""
    try:
        train_data, _ = util.read_data(error=0, is_train=True)
        train_data = preprocessing.StandardScaler().fit_transform(train_data)
    except FileNotFoundError:
        print("Training data file not found. Using mock data.")
        train_data = get_mock_data()
    return train_data


def get_test_data():
    """Get test data, generating mock data if real data is unavailable"""
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


def plot_loss_curve(train_losses, test_losses):
    """Plot the training and testing loss curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Reconstruction Error)')
    plt.title('Training and Testing Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig('transformer_autoencoder_loss.png')
    plt.close()


def main():
    # Check data directory
    check_data_dir()
    
    # Set device to GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load training and testing data
    train_data = get_train_data()
    test_data = get_test_data()
    
    # Verify data shape
    print(f"Training data shape: {train_data.shape}")
    print(f"Testing data shape: {test_data.shape}")
    
    # Convert to PyTorch tensors
    train_dataset = torch.from_numpy(train_data).float()
    test_dataset = torch.from_numpy(test_data).float()

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize the model
    input_dim = train_data.shape[1]  # Use actual data dimension
    hidden_dim = min(27, input_dim - 1)  # Same compression as original autoencoder, but ensure <= input_dim
    
    print(f"Using input dimension: {input_dim}, hidden dimension: {hidden_dim}")
    model = TransformerAutoencoder(input_dim, hidden_dim)
    model.to(device)
    
    # Set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    epochs = 500
    train_losses = []
    test_losses = []
    
    for i in range(epochs):
        train_loss = train(model, optimizer, train_loader, device)
        test_loss = validate(model, test_loader, device)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        print('{}\tepoch = {}\ttrain loss: {:0.3f}\ttest loss: {:0.3f}' \
              .format(datetime.datetime.now(), i, train_loss, test_loss))
        
        # Optional: Early stopping
        if i > 50 and i % 10 == 0:
            if test_losses[-10] - test_losses[-1] < 0.0001:
                print("Early stopping at epoch", i)
                break
    
    # Plot loss curves
    plot_loss_curve(train_losses, test_losses)
    
    # Save the model
    torch.save(model.state_dict(), 'transformer_autoencoder.pth')
    
    return model


if __name__ == '__main__':
    try:
        model = main()
        
        # Extract features and train a classifier (similar to original autoencoder.py)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Get training data and prepare feature extraction
        train_data = get_train_data()
        train_tensor = torch.from_numpy(train_data).float().to(device)
        
        # Extract features using transformer encoder
        model.eval()
        with torch.no_grad():
            # Process in batches if data is large
            if len(train_tensor) > 1000:
                batch_size = 100
                train_features = []
                for i in range(0, len(train_tensor), batch_size):
                    batch = train_tensor[i:i+batch_size]
                    features = model.extract_features(batch).cpu().numpy()
                    train_features.append(features)
                train_features = np.vstack(train_features)
            else:
                train_features = model.extract_features(train_tensor).cpu().numpy()
        
        # Get labels (if available)
        try:
            train_labels, _ = util.read_data(error=0, is_train=True, return_label=True)
        except (FileNotFoundError, TypeError):
            print("Using mock labels for classifier")
            train_labels = np.zeros(len(train_data))
        
        # Create pipeline with NCA and MLP classifier
        nca = NeighborhoodComponentsAnalysis(n_components=min(10, train_features.shape[1]-1), random_state=42)
        nn_classifier = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42)
        pipeline = Pipeline([('nca', nca), ('mlp', nn_classifier)])
        
        # Fit the pipeline on the extracted features
        pipeline.fit(train_features, train_labels)
        
        # Save the pipeline
        dump(pipeline, 'transformer_nca_mlp_model.joblib')
        
        print("Transformer autoencoder and classifier pipeline trained successfully.")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        print("Please ensure all dependencies are installed and data files are available.") 