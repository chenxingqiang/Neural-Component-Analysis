import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from scipy import stats
from sklearn.preprocessing import StandardScaler
import time
from enhanced_transformer_autoencoder import (
    EnhancedTransformerAutoencoder, 
    calculate_weighted_spe,
    adaptive_control_limits,
    train_enhanced_model
)


def simple_kde(data, n_points=1000):
    """
    Improved kernel density estimation using adaptive bandwidth for better precision
    """
    # Handle extreme case: all values are the same
    if np.std(data) < 1e-8:
        x = np.linspace(np.min(data) - 0.1, np.max(data) + 0.1, n_points)
        density = np.zeros_like(x)
        density[n_points//2] = 1.0
        cdf = np.zeros_like(x)
        cdf[n_points//2:] = 1.0
        return 0.01, density, x, cdf
    
    # Calculate adaptive bandwidth using Scott's rule
    bandwidth = 1.06 * np.std(data) * (len(data) ** (-1/5))
    
    # Use gaussian_kde with adaptive bandwidth
    try:
        kde = stats.gaussian_kde(data, bw_method=bandwidth)
    except:
        # If failed, fall back to default bandwidth
        kde = stats.gaussian_kde(data)
    
    # Create evaluation grid
    margin = 0.2 * (np.max(data) - np.min(data))
    x_min = np.min(data) - margin
    x_max = np.max(data) + margin
    x = np.linspace(x_min, x_max, n_points)
    
    # Calculate density
    density = kde(x)
    
    # Calculate CDF
    cdf = np.zeros_like(x)
    dx = x[1] - x[0]
    for i in range(1, len(x)):
        cdf[i] = cdf[i-1] + density[i-1] * dx
    
    # Normalize CDF
    if cdf[-1] > 0:
        cdf = cdf / cdf[-1]
    
    return bandwidth, density, x, cdf


def load_data(is_mock=False):
    """Load data, create mock data if real data unavailable"""
    if is_mock or not os.path.exists('./data/train/d00.dat'):
        print("Using mock data for demonstration")
        np.random.seed(42)
        # Create mock normal operation data
        d00 = np.random.randn(500, 52).astype(np.float32)
        # Create mock test data (fault conditions)
        d01_te = np.random.randn(500, 52).astype(np.float32)
        # Add fault features after sample 160
        d01_te[160:, :10] += 2.0  # Add offset to first 10 variables after sample 160
        d01_te[160:, 5:15] *= 1.5  # Increase variance in some variables
        happen = 160
    else:
        try:
            # Load actual data
            d00 = np.loadtxt('./data/train/d00.dat')
            d01_te = np.loadtxt('./data/test/d01_te.dat')
            happen = 160  # Same as MATLAB code
        except:
            print("Error loading data, using mock data instead")
            return load_data(is_mock=True)
    
    # Use row-oriented data directly
    X = d00  # Shape: (num_samples, num_features)
    XT = d01_te  # Shape: (num_samples, num_features)
    
    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    XT_scaled = scaler.transform(XT)
    
    return X_scaled, XT_scaled, happen


def calculate_variable_importance(model, X_train, device):
    """Calculate variable importance weights for weighted SPE calculation"""
    model.eval()
    
    # Convert data to tensor
    X_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    
    # Get original reconstruction errors
    with torch.no_grad():
        reconstructed, _ = model(X_tensor)
        original_errors = torch.mean((X_tensor - reconstructed)**2, dim=0)
    
    # Initialize importance weights
    importance_weights = torch.ones(X_train.shape[1], device=device)
    
    # Calculate perturbation impact for each variable
    batch_size = min(100, len(X_train))
    sample_indices = np.random.choice(len(X_train), batch_size, replace=False)
    X_batch = X_tensor[sample_indices]
    
    for i in range(X_train.shape[1]):
        # Create perturbed data
        perturbed_batch = X_batch.clone()
        perturbed_batch[:, i] += torch.randn(batch_size, device=device) * 0.1
        
        # Reconstruct perturbed data
        with torch.no_grad():
            perturbed_reconstructed, _ = model(perturbed_batch)
            perturbed_errors = torch.mean((perturbed_batch - perturbed_reconstructed)**2, dim=0)
        
        # Calculate perturbation impact
        impact = torch.abs(perturbed_errors - original_errors).sum()
        importance_weights[i] = 1.0 + impact.item()
    
    # Normalize importance weights
    importance_weights = importance_weights / importance_weights.sum() * len(importance_weights)
    
    return importance_weights


def calculate_t2_statistics(model, data, device, Sigma_inv=None):
    """Calculate T2 statistics using Mahalanobis distance"""
    model.eval()
    batch_size = 32
    features_list = []
    
    # Extract features
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = torch.tensor(data[i:i+batch_size], dtype=torch.float32).to(device)
            features = model.extract_features(batch)
            features_list.append(features.cpu().numpy())
    
    features = np.vstack(features_list)
    
    # Calculate inverse covariance matrix if not provided
    if Sigma_inv is None:
        # Calculate feature covariance matrix
        Sigma = np.cov(features, rowvar=False)
        # Add regularization to ensure matrix is invertible
        Sigma = Sigma + np.eye(Sigma.shape[0]) * 1e-6
        Sigma_inv = np.linalg.inv(Sigma)
    
    # Calculate T2 statistic for each sample
    t2_values = []
    for feature in features:
        # Calculate Mahalanobis distance
        t2 = np.dot(np.dot(feature, Sigma_inv), feature)
        t2_values.append(t2)
    
    return np.array(t2_values), Sigma_inv


def calculate_control_limits(values, method='kde', confidence=0.99, false_alarm_target=0.01):
    """Calculate control limits, supporting multiple methods"""
    if method == 'kde':
        # Use KDE method to calculate control limit
        _, _, xmesh, cdf = simple_kde(values)
        limit = None
        for i in range(len(cdf)):
            if cdf[i] >= confidence:
                limit = xmesh[i]
                break
        
        if limit is None:
            limit = np.max(values) * 1.1  # Fallback when KDE fails
    
    elif method == 'adaptive':
        # Use adaptive method to calculate control limit
        limit = adaptive_control_limits(values, false_alarm_target)
    
    else:  # 'percentile'
        # Use percentile method to calculate control limit
        limit = np.percentile(values, 100 * confidence)
    
    return limit


def calculate_alarm_rates(t2_test, spe_test, t2_limit, spe_limit, happen):
    """Calculate false alarm rates and miss rates"""
    # False alarm rate (normal samples incorrectly identified as faulty)
    false_t2 = sum(t2_test[:happen] > t2_limit)
    false_spe = sum(spe_test[:happen] > spe_limit)
    
    false_rate_t2 = 100 * false_t2 / happen
    false_rate_spe = 100 * false_spe / happen
    
    # Miss rate (faulty samples not detected)
    miss_t2 = sum(t2_test[happen:] < t2_limit)
    miss_spe = sum(spe_test[happen:] < spe_limit)
    
    miss_rate_t2 = 100 * miss_t2 / (len(t2_test) - happen)
    miss_rate_spe = 100 * miss_spe / (len(spe_test) - happen)
    
    return (false_rate_t2, false_rate_spe), (miss_rate_t2, miss_rate_spe)


def calculate_detection_time(t2_test, spe_test, t2_limit, spe_limit, happen, consecutive_required=3):
    """
    Calculate detection time (consecutive samples exceeding threshold)
    Reduced consecutive_required to improve detection speed
    """
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


def plot_enhanced_metrics(t2_test, spe_test, t2_limit, spe_limit, happen, title_prefix="Enhanced Transformer"):
    """Enhanced plotting function, showing normal/fault boundary"""
    plt.figure(figsize=(12, 8))
    
    # T2 plot
    plt.subplot(2, 1, 1)
    plt.plot(range(1, happen+1), t2_test[:happen], 'g', label='Normal')
    plt.plot(range(happen+1, len(t2_test)+1), t2_test[happen:], 'm', label='Fault')
    plt.axhline(y=t2_limit, color='k', linestyle='--', label='Control Limit')
    plt.axvline(x=happen, color='r', linestyle='-', label='Fault Occurrence')
    plt.title(f'{title_prefix} - T² Statistics')
    plt.xlabel('Sample')
    plt.ylabel('T²')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # SPE plot
    plt.subplot(2, 1, 2)
    plt.plot(range(1, happen+1), spe_test[:happen], 'g', label='Normal')
    plt.plot(range(happen+1, len(spe_test)+1), spe_test[happen:], 'm', label='Fault')
    plt.axhline(y=spe_limit, color='k', linestyle='--', label='Control Limit')
    plt.axvline(x=happen, color='r', linestyle='-', label='Fault Occurrence')
    plt.title(f'{title_prefix} - SPE Statistics')
    plt.xlabel('Sample')
    plt.ylabel('SPE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{title_prefix.lower().replace(" ", "_")}_fault_detection.png')
    print(f"Chart saved as {title_prefix.lower().replace(' ', '_')}_fault_detection.png")
    plt.close()


def contribution_plot(model, faulty_sample, normal_data, device, n_top=10):
    """Generate contribution plot showing which variables contribute most to fault"""
    model.eval()
    
    # Convert data to tensors
    faulty_tensor = torch.tensor(faulty_sample, dtype=torch.float32).unsqueeze(0).to(device)
    normal_tensor = torch.tensor(normal_data, dtype=torch.float32).to(device)
    
    # Reconstruct samples
    with torch.no_grad():
        faulty_reconstructed, _ = model(faulty_tensor)
        normal_batch_reconstructed, _ = model(normal_tensor)
    
    # Calculate reconstruction error for each variable
    faulty_errors = (faulty_tensor - faulty_reconstructed)**2
    normal_errors = torch.mean((normal_tensor - normal_batch_reconstructed)**2, dim=0)
    
    # Calculate error difference between faulty and normal samples
    error_diff = (faulty_errors - normal_errors).squeeze().cpu().numpy()
    
    # Find variables with highest contribution
    top_indices = np.argsort(error_diff)[-n_top:][::-1]
    top_contributions = error_diff[top_indices]
    
    # Plot contribution chart
    plt.figure(figsize=(10, 6))
    plt.bar(range(n_top), top_contributions)
    plt.xticks(range(n_top), [f'Var {i+1}' for i in top_indices], rotation=45)
    plt.title('Variable Contribution Plot - Fault Diagnosis')
    plt.xlabel('Variable')
    plt.ylabel('Contribution')
    plt.tight_layout()
    plt.savefig('fault_contribution_plot.png')
    plt.close()
    
    return top_indices, top_contributions


def main():
    """Main function for fault detection using enhanced Transformer autoencoder"""
    start_time = time.time()
    
    # Load data
    X_train, X_test, happen = load_data(is_mock=True)
    print(f"Data loaded. Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize and load model
    input_dim = X_train.shape[1]
    hidden_dim = min(27, input_dim - 1)
    
    # Create model instance
    model = EnhancedTransformerAutoencoder(input_dim, hidden_dim)
    
    # Try to load pretrained model, train new model if not found
    try:
        model.load_state_dict(torch.load('enhanced_transformer_autoencoder.pth', map_location=device))
        print("Loaded pretrained enhanced Transformer model")
    except:
        print("Pretrained model not found. Training new model...")
        model, _, _ = train_enhanced_model(
            X_train, 
            epochs=50,
            batch_size=32,
            lr=0.001,
            hidden_dim=hidden_dim,
            validation_split=0.2
        )
    
    model.to(device)
    
    # Calculate variable importance weights
    print("Calculating variable importance weights...")
    importance_weights = calculate_variable_importance(model, X_train, device)
    
    # Calculate T2 statistics for training data
    print("Calculating T2 statistics for training data...")
    t2_train, Sigma_inv = calculate_t2_statistics(model, X_train, device)
    
    # Calculate weighted SPE
    print("Calculating weighted SPE for training data...")
    spe_train = calculate_weighted_spe(model, X_train, device, importance_weights)
    
    # Calculate adaptive control limits
    print("Calculating adaptive control limits...")
    t2_limit = calculate_control_limits(t2_train, method='adaptive', false_alarm_target=0.01)
    spe_limit = calculate_control_limits(spe_train, method='adaptive', false_alarm_target=0.01)
    print(f"Control limits: T2 = {t2_limit:.2f}, SPE = {spe_limit:.2f}")
    
    # Calculate metrics for test data
    print("Calculating metrics for test data...")
    t2_test, _ = calculate_t2_statistics(model, X_test, device, Sigma_inv)
    spe_test = calculate_weighted_spe(model, X_test, device, importance_weights)
    
    # Calculate alarm rates
    false_rates, miss_rates = calculate_alarm_rates(
        t2_test, spe_test, t2_limit, spe_limit, happen)
    
    print("\n===== Enhanced Transformer -- False Alarm Rates =====")
    print(f"T2: {false_rates[0]:.2f}%, SPE: {false_rates[1]:.2f}%")
    
    print("\n===== Enhanced Transformer -- Miss Rates =====")
    print(f"T2: {miss_rates[0]:.2f}%, SPE: {miss_rates[1]:.2f}%")
    
    # Calculate detection time (using fewer consecutive samples requirement)
    det_time_t2, det_time_spe = calculate_detection_time(
        t2_test, spe_test, t2_limit, spe_limit, happen, consecutive_required=3)
    
    print(f"\nDetection Time T2: {det_time_t2 if det_time_t2 is not None else 'Not Detected'}")
    print(f"Detection Time SPE: {det_time_spe if det_time_spe is not None else 'Not Detected'}")
    
    # Plot results
    plot_enhanced_metrics(t2_test, spe_test, t2_limit, spe_limit, happen)
    
    # Create contribution plot if fault detected
    if det_time_spe is not None:
        fault_sample_index = happen + det_time_spe + 5  # Select a sample after detection
        if fault_sample_index < len(X_test):
            print("\nGenerating fault contribution plot...")
            top_vars, _ = contribution_plot(model, X_test[fault_sample_index], X_train, device)
            print(f"Variables with highest fault contribution: {top_vars}")
    
    runtime = time.time() - start_time
    print(f"\nTotal runtime: {runtime:.2f} seconds")
    
    # Return results dictionary
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
        'importance_weights': importance_weights.cpu().numpy(),
        'runtime': runtime
    }


if __name__ == "__main__":
    main() 