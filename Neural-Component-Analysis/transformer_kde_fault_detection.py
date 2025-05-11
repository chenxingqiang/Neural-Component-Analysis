import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
from scipy import stats
from sklearn.preprocessing import StandardScaler
from scipy import signal
import time
from transformer_autoencoder import TransformerAutoencoder
from joblib import load


def simple_kde(data, n_points=1000):
    """
    A simplified kernel density estimation implementation using scipy's gaussian_kde
    """
    # Handle case where all values are the same
    if np.std(data) < 1e-8:
        x = np.linspace(np.min(data) - 0.1, np.max(data) + 0.1, n_points)
        density = np.zeros_like(x)
        density[n_points//2] = 1.0
        # Simple step function for CDF
        cdf = np.zeros_like(x)
        cdf[n_points//2:] = 1.0
        return 0.01, density, x, cdf
    
    # Use scipy's gaussian_kde for more robust KDE
    kde = stats.gaussian_kde(data)
    
    # Create a grid for evaluation
    x_min = np.min(data) - 0.1 * (np.max(data) - np.min(data))
    x_max = np.max(data) + 0.1 * (np.max(data) - np.min(data))
    x = np.linspace(x_min, x_max, n_points)
    
    # Calculate density
    density = kde(x)
    
    # Calculate CDF by integrating the density
    cdf = np.zeros_like(x)
    for i in range(1, len(x)):
        cdf[i] = cdf[i-1] + density[i-1] * (x[i] - x[i-1])
    
    # Normalize CDF
    if cdf[-1] > 0:
        cdf = cdf / cdf[-1]
    
    # Bandwidth estimation is the default used by scipy
    bandwidth = kde.factor
    
    return bandwidth, density, x, cdf


def load_data(is_mock=False):
    """Load the data, create mock data if real data is unavailable"""
    if is_mock or not os.path.exists('./data/train/d00.dat'):
        print("Using mock data for demonstration")
        np.random.seed(42)
        # Create mock normal operation data
        d00 = np.random.randn(500, 52).astype(np.float32)
        # Create mock test data (fault condition)
        d01_te = np.random.randn(500, 52).astype(np.float32)
        # Add a fault signature after sample 160
        d01_te[160:, :10] += 2.0  # Add offset to first 10 variables after sample 160
        d01_te[160:, 5:15] *= 1.5  # Add variance to some variables
        happen = 160
    else:
        try:
            # Load actual data
            d00 = np.loadtxt('./data/train/d00.dat')
            d01_te = np.loadtxt('./data/test/d01_te.dat')
            happen = 160  # Same as in MATLAB code
        except:
            print("Error loading data, using mock data instead")
            return load_data(is_mock=True)
    
    # No need to transpose anymore since we're working with row-oriented data directly
    X = d00  # Shape: (num_samples, num_features)
    XT = d01_te  # Shape: (num_samples, num_features)
    
    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    XT_scaled = scaler.transform(XT)
    
    return X_scaled, XT_scaled, happen


def calculate_limits(t2, spe, confidence=0.99):
    """Calculate the control limits using KDE"""
    # Calculate T2 limit
    _, _, xmesh_t2, cdf_t2 = simple_kde(t2)
    T2limit = None
    for i in range(len(cdf_t2)):
        if cdf_t2[i] >= confidence:
            T2limit = xmesh_t2[i]
            break
    
    if T2limit is None:
        T2limit = np.max(t2) * 1.1  # Fallback if KDE fails
    
    # Calculate SPE limit
    _, _, xmesh_spe, cdf_spe = simple_kde(spe)
    SPElimit = None
    for i in range(len(cdf_spe)):
        if cdf_spe[i] >= confidence:
            SPElimit = xmesh_spe[i]
            break
    
    if SPElimit is None:
        SPElimit = np.max(spe) * 1.1  # Fallback if KDE fails
    
    return T2limit, SPElimit


def calculate_metrics(model, data, latent_dim):
    """Calculate T2 and SPE metrics using the transformer model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    batch_size = 32
    t2_values = []
    spe_values = []
    
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = torch.tensor(data[i:i+batch_size], dtype=torch.float32).to(device)
            # Get reconstruction and latent features
            reconstructed, _ = model(batch)
            
            # Extract features (equivalent to PCA scores)
            features = model.extract_features(batch)
            
            # Calculate T2 statistic for each sample
            for j in range(len(batch)):
                # Use mahalanobis distance as T2 statistic
                feature_j = features[j].unsqueeze(0)
                # Simplified T2 calculation for demonstration
                t2 = torch.sum((feature_j)**2 / (torch.var(features, dim=0) + 1e-8))  # Add epsilon to avoid division by zero
                t2_values.append(t2.cpu().numpy())
                
                # Calculate SPE (reconstruction error)
                spe = torch.sum((batch[j] - reconstructed[j])**2).cpu().numpy()
                spe_values.append(spe)
    
    return np.array(t2_values), np.array(spe_values)


def calculate_alarm_rates(t2_test, spe_test, t2_limit, spe_limit, happen):
    """Calculate false alarm and miss alarm rates"""
    # False alarm rate (normal samples incorrectly identified as faulty)
    false_t2 = sum(t2_test[:happen] > t2_limit)
    false_spe = sum(spe_test[:happen] > spe_limit)
    
    false_rate_t2 = 100 * false_t2 / happen
    false_rate_spe = 100 * false_spe / happen
    
    # Miss alarm rate (faulty samples not detected)
    miss_t2 = sum(t2_test[happen:] < t2_limit)
    miss_spe = sum(spe_test[happen:] < spe_limit)
    
    miss_rate_t2 = 100 * miss_t2 / (len(t2_test) - happen)
    miss_rate_spe = 100 * miss_spe / (len(spe_test) - happen)
    
    return (false_rate_t2, false_rate_spe), (miss_rate_t2, miss_rate_spe)


def calculate_detection_time(t2_test, spe_test, t2_limit, spe_limit, happen):
    """Calculate detection time (consecutive samples above threshold)"""
    consecutive_required = 6  # Same as MATLAB code
    
    # T2 detection time
    detection_time_t2 = None
    for i in range(happen, len(t2_test) - consecutive_required + 1):
        if all(t2_test[i:i+consecutive_required] > t2_limit):
            detection_time_t2 = i - happen
            break
    
    # SPE detection time
    detection_time_spe = None
    for i in range(happen, len(spe_test) - consecutive_required + 1):
        if all(spe_test[i:i+consecutive_required] > spe_limit):
            detection_time_spe = i - happen
            break
    
    return detection_time_t2, detection_time_spe


def plot_metrics(t2_test, spe_test, t2_limit, spe_limit, happen, title_prefix="Transformer"):
    """Create plots similar to the MATLAB code"""
    plt.figure(figsize=(12, 8))
    
    # T2 plot
    plt.subplot(2, 1, 1)
    plt.plot(range(1, happen+1), t2_test[:happen], 'g', label='Normal')
    plt.plot(range(happen+1, len(t2_test)+1), t2_test[happen:], 'm', label='Faulty')
    plt.axhline(y=t2_limit, color='k', linestyle='--', label='Threshold')
    plt.title(f'{title_prefix} - T2 for TE data')
    plt.xlabel('Sample')
    plt.ylabel('T2')
    plt.legend()
    
    # SPE plot
    plt.subplot(2, 1, 2)
    plt.plot(range(1, happen+1), spe_test[:happen], 'g', label='Normal')
    plt.plot(range(happen+1, len(spe_test)+1), spe_test[happen:], 'm', label='Faulty')
    plt.axhline(y=spe_limit, color='k', linestyle='--', label='Threshold')
    plt.title(f'{title_prefix} - SPE for TE data')
    plt.xlabel('Sample')
    plt.ylabel('SPE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{title_prefix.lower()}_fault_detection.png')
    print(f"Plot saved as {title_prefix.lower()}_fault_detection.png")
    plt.close()


def main():
    """Main function to run the transformer-based fault detection"""
    start_time = time.time()
    
    # Load data
    X_train, X_test, happen = load_data(is_mock=True)  # Use mock data for demonstration
    print(f"Data loaded. Training shape: {X_train.shape}, Testing shape: {X_test.shape}")
    
    # Initialize and load the transformer model
    input_dim = X_train.shape[1]
    hidden_dim = min(27, input_dim - 1)  # Same compression as in transformer_autoencoder.py
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerAutoencoder(input_dim, hidden_dim)
    
    # Try to load pre-trained model
    try:
        model.load_state_dict(torch.load('transformer_autoencoder.pth', map_location=device))
        print("Loaded pre-trained transformer model")
    except:
        print("Pre-trained model not found. Training a new model...")
        # Create a simplified training process
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        train_tensor = torch.tensor(X_train, dtype=torch.float32)
        train_loader = torch.utils.data.DataLoader(train_tensor, batch_size=32, shuffle=True)
        
        model.to(device)
        model.train()
        for epoch in range(10):  # Just train for a few epochs for demonstration
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                reconstructed, _ = model(batch)
                loss = torch.mean((batch - reconstructed)**2)
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch+1}/10, Loss: {loss.item():.4f}")
        
        # Save the model
        torch.save(model.state_dict(), 'transformer_autoencoder.pth')
        print("Model trained and saved.")
    
    model.to(device)
    
    # Calculate T2 and SPE metrics for training data
    print("Calculating metrics for training data...")
    t2_train, spe_train = calculate_metrics(model, X_train, hidden_dim)
    
    # Calculate control limits
    t2_limit, spe_limit = calculate_limits(t2_train, spe_train)
    print(f"Control limits: T2 = {t2_limit:.2f}, SPE = {spe_limit:.2f}")
    
    # Calculate metrics for test data
    print("Calculating metrics for test data...")
    t2_test, spe_test = calculate_metrics(model, X_test, hidden_dim)
    
    # Calculate alarm rates
    false_rates, miss_rates = calculate_alarm_rates(
        t2_test, spe_test, t2_limit, spe_limit, happen)
    
    print("---- Transformer -- False alarm rate ----")
    print(f"T2: {false_rates[0]:.2f}%, SPE: {false_rates[1]:.2f}%")
    
    print("---- Transformer -- Miss alarm rate ----")
    print(f"T2: {miss_rates[0]:.2f}%, SPE: {miss_rates[1]:.2f}%")
    
    # Calculate detection time
    det_time_t2, det_time_spe = calculate_detection_time(
        t2_test, spe_test, t2_limit, spe_limit, happen)
    
    print(f"Detection time T2: {det_time_t2 if det_time_t2 is not None else 'Not detected'}")
    print(f"Detection time SPE: {det_time_spe if det_time_spe is not None else 'Not detected'}")
    
    # Plot results
    plot_metrics(t2_test, spe_test, t2_limit, spe_limit, happen)
    
    runtime = time.time() - start_time
    print(f"Runtime: {runtime:.2f} seconds")
    
    return {
        'model': model,
        't2_train': t2_train,
        'spe_train': spe_train,
        't2_test': t2_test,
        'spe_test': spe_test,
        't2_limit': t2_limit,
        'spe_limit': spe_limit,
        'false_rates': false_rates,
        'miss_rates': miss_rates,
        'detection_time_t2': det_time_t2,
        'detection_time_spe': det_time_spe,
        'runtime': runtime
    }


if __name__ == "__main__":
    main() 