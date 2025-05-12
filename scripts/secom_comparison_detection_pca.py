import numpy as np
import matplotlib.pyplot as plt
import os
import time
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from run_secom_fault_detection import load_secom_data

def calculate_pca_metrics(data, pca_model, projected_train, latent_dim):
    """Calculate T2 and SPE metrics using PCA model"""
    # Project data to PCA space
    projected = pca_model.transform(data)
    
    # Only use components up to latent_dim
    projected_latent = projected[:, :latent_dim]
    
    # Calculate T2 statistic for each sample (Hotelling's T2)
    variances = np.var(projected_train[:, :latent_dim], axis=0) + 1e-8
    t2_values = np.sum((projected_latent**2) / variances, axis=1)
    
    # Calculate SPE (Q statistic) - reconstruction error
    reconstructed = pca_model.inverse_transform(projected)
    spe_values = np.sum((data - reconstructed)**2, axis=1)
    
    return t2_values, spe_values


def calculate_control_limits(values, method='kde', confidence=0.99):
    """Calculate control limits using various methods"""
    if method == 'kde':
        # Use Kernel Density Estimation (KDE) for adaptive control limit
        from scipy import stats
        kde = stats.gaussian_kde(values)
        x_grid = np.linspace(np.min(values), np.max(values), 1000)
        pdf = kde(x_grid)
        cdf = np.cumsum(pdf) / np.sum(pdf)
        idx = np.argmin(np.abs(cdf - confidence))
        limit = x_grid[idx]
    elif method == 'chi2':
        # Chi-square method for T2
        from scipy import stats
        dof = min(20, len(values)) # use degrees of freedom limited to 20 to avoid extreme values
        limit = stats.chi2.ppf(confidence, dof) * np.mean(values)
    elif method == 'percentile':
        # Simple percentile method
        limit = np.percentile(values, confidence * 100)
    else:
        # Default: use percentile method
        limit = np.percentile(values, confidence * 100)
    
    return limit


def plot_pca_comparison(t2_pca, spe_pca, t2_limit_pca, spe_limit_pca, happen):
    """Plot comparison of PCA T2 and SPE statistics"""
    plt.figure(figsize=(15, 10))
    
    # Ensure arrays are 1D
    t2_pca = np.ravel(t2_pca)
    spe_pca = np.ravel(spe_pca)
    
    # T2 PCA plot (top)
    plt.subplot(2, 1, 1)
    plt.plot(range(1, happen+1), t2_pca[:happen], 'b', label='Normal')
    plt.plot(range(happen+1, len(t2_pca)+1), t2_pca[happen:], 'r', label='Fault')
    plt.axhline(y=t2_limit_pca, color='k', linestyle='--', label='Threshold')
    plt.axvline(x=happen, color='m', linestyle='-', label='Fault Time')
    plt.title('PCA - T² Statistics')
    plt.xlabel('Sample')
    plt.ylabel('T²')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # SPE PCA plot (bottom)
    plt.subplot(2, 1, 2)
    plt.plot(range(1, happen+1), spe_pca[:happen], 'b', label='Normal')
    plt.plot(range(happen+1, len(spe_pca)+1), spe_pca[happen:], 'r', label='Fault')
    plt.axhline(y=spe_limit_pca, color='k', linestyle='--', label='Threshold')
    plt.axvline(x=happen, color='m', linestyle='-', label='Fault Time')
    plt.title('PCA - SPE Statistics')
    plt.xlabel('Sample')
    plt.ylabel('SPE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("results/plots/secom_pca_comparison.png")
    print("Plot saved as secom_pca_comparison.png")
    plt.close()


def main():
    """Main function to run PCA-based fault detection for SECOM data"""
    print("============================================================")
    print("SECOM PCA Fault Detection")
    print("============================================================")
    start_time = time.time()
    
    # Load data
    print("Loading SECOM data...")
    X_train, X_test, happen, y_test, normal_indices, fault_indices = load_secom_data()
    if X_train is None:
        print("Error loading data. Exiting.")
        return
    
    print(f"Data loaded. Training shape: {X_train.shape}, Testing shape: {X_test.shape}")
    print(f"Fault occurrence at sample: {happen}")
    
    # 1. Implement PCA as baseline
    print("\nSetting up PCA model...")
    # Set up PCA model
    n_components = min(64, X_train.shape[1] - 1)  # Similar to transformer models
    print(f"Using {n_components} principal components for PCA")
    pca_model = PCA(n_components=X_train.shape[1])  # Keep all components for reconstruction
    pca_model.fit(X_train)
    
    # Project training data
    projected_train = pca_model.transform(X_train)
    
    # Calculate % variance explained by components
    explained_variance_ratio = pca_model.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    # Find number of components for desired variance (85%)
    latent_dim = 0
    for i, var in enumerate(cumulative_variance):
        if var >= 0.85:
            latent_dim = i + 1
            break
    
    print(f"Selected {latent_dim} principal components explaining 85% of variance")
    
    # Calculate PCA metrics
    print("Calculating PCA metrics...")
    t2_train_pca, spe_train_pca = calculate_pca_metrics(
        X_train, pca_model, projected_train, latent_dim)
    t2_test_pca, spe_test_pca = calculate_pca_metrics(
        X_test, pca_model, projected_train, latent_dim)
    
    # Calculate PCA limits
    t2_limit_pca = calculate_control_limits(t2_train_pca, method='kde')
    spe_limit_pca = calculate_control_limits(spe_train_pca, method='kde')
    print(f"PCA control limits: T² = {t2_limit_pca:.2f}, SPE = {spe_limit_pca:.2f}")
    
    # Calculate PCA alarm rates
    t2_pca_alarms = t2_test_pca > t2_limit_pca
    spe_pca_alarms = spe_test_pca > spe_limit_pca
    
    # False alarm rates (% of normal samples that trigger alarm)
    t2_pca_false_alarms = np.sum(t2_pca_alarms[:happen])
    spe_pca_false_alarms = np.sum(spe_pca_alarms[:happen])
    t2_pca_false_rate = (t2_pca_false_alarms / happen) * 100.0
    spe_pca_false_rate = (spe_pca_false_alarms / happen) * 100.0
    
    # Miss rates (% of fault samples not detected)
    t2_pca_misses = np.sum(~t2_pca_alarms[happen:])
    spe_pca_misses = np.sum(~spe_pca_alarms[happen:])
    t2_pca_miss_rate = (t2_pca_misses / (len(t2_test_pca) - happen)) * 100.0
    spe_pca_miss_rate = (spe_pca_misses / (len(spe_test_pca) - happen)) * 100.0
    
    print(f"PCA T² False Alarm Rate: {t2_pca_false_rate:.2f}%, Miss Rate: {t2_pca_miss_rate:.2f}%")
    print(f"PCA SPE False Alarm Rate: {spe_pca_false_rate:.2f}%, Miss Rate: {spe_pca_miss_rate:.2f}%")
    
    # Detection times
    t2_pca_detection_time = None
    spe_pca_detection_time = None
    consecutive_required = 3  # Number of consecutive alarms to confirm detection
    
    # Find T² detection time
    consecutive_count = 0
    for i in range(happen, len(t2_pca_alarms)):
        if t2_pca_alarms[i]:
            consecutive_count += 1
            if consecutive_count >= consecutive_required:
                t2_pca_detection_time = i - happen
                break
        else:
            consecutive_count = 0
    
    # Find SPE detection time
    consecutive_count = 0
    for i in range(happen, len(spe_pca_alarms)):
        if spe_pca_alarms[i]:
            consecutive_count += 1
            if consecutive_count >= consecutive_required:
                spe_pca_detection_time = i - happen
                break
        else:
            consecutive_count = 0
    
    print(f"PCA Detection Time - T²: {t2_pca_detection_time if t2_pca_detection_time is not None else 'Not detected'}")
    print(f"PCA Detection Time - SPE: {spe_pca_detection_time if spe_pca_detection_time is not None else 'Not detected'}")
    
    # Print comparison table
    print("\n" + "="*90)
    print("{:<20} | {:<12} {:<12} | {:<12} {:<12}".format(
        "Method", "T² False(%)", "SPE False(%)", "T² Miss(%)", "SPE Miss(%)"))
    print("-"*90)
    print("{:<20} | {:<12.2f} {:<12.2f} | {:<12.2f} {:<12.2f}".format(
        "PCA", t2_pca_false_rate, spe_pca_false_rate, 
        t2_pca_miss_rate, spe_pca_miss_rate))
    print("-"*90)
    print("{:<20} | {:<27} | {:<25}".format(
        "Method", "Detection Time T²", "Detection Time SPE"))
    print("-"*90)
    det_t2_str = str(t2_pca_detection_time) if t2_pca_detection_time is not None else "Not detected"
    det_spe_str = str(spe_pca_detection_time) if spe_pca_detection_time is not None else "Not detected"
    print("{:<20} | {:<27} | {:<25}".format(
        "PCA", det_t2_str, det_spe_str))
    print("="*90)
    
    # Plot PCA comparisons
    plot_pca_comparison(t2_test_pca, spe_test_pca, t2_limit_pca, spe_limit_pca, happen)
    
    runtime = time.time() - start_time
    print(f"Total Runtime: {runtime:.2f} seconds")
    
    return {
        'pca_results': {
            't2_test': t2_test_pca,
            'spe_test': spe_test_pca,
            't2_limit': t2_limit_pca,
            'spe_limit': spe_limit_pca,
            'false_rates': [t2_pca_false_rate, spe_pca_false_rate],
            'miss_rates': [t2_pca_miss_rate, spe_pca_miss_rate],
            'detection_times': [t2_pca_detection_time, spe_pca_detection_time]
        },
        'runtime': runtime
    }


if __name__ == "__main__":
    main() 