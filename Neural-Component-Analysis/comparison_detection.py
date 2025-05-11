import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import time
from enhanced_transformer_autoencoder import (
    EnhancedTransformerAutoencoder, 
    calculate_weighted_spe,
    adaptive_control_limits,
    train_enhanced_model
)
from enhanced_transformer_detection import (
    simple_kde, load_data, calculate_variable_importance,
    calculate_t2_statistics, calculate_control_limits,
    calculate_alarm_rates, calculate_detection_time
)


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


def plot_comparison(t2_pca, spe_pca, t2_transformer, spe_transformer, 
                   t2_limit_pca, spe_limit_pca, t2_limit_transformer, spe_limit_transformer, happen):
    """Create comparison plots between PCA and Enhanced Transformer-based detection"""
    plt.figure(figsize=(15, 10))
    
    # T2 PCA plot
    plt.subplot(2, 2, 1)
    plt.plot(range(1, happen+1), t2_pca[:happen], 'b', label='Normal')
    plt.plot(range(happen+1, len(t2_pca)+1), t2_pca[happen:], 'r', label='Fault')
    plt.axhline(y=t2_limit_pca, color='k', linestyle='--', label='Threshold')
    plt.axvline(x=happen, color='m', linestyle='-', label='Fault Time')
    plt.title('PCA - T² Statistics')
    plt.xlabel('Sample')
    plt.ylabel('T²')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # T2 Transformer plot
    plt.subplot(2, 2, 2)
    plt.plot(range(1, happen+1), t2_transformer[:happen], 'g', label='Normal')
    plt.plot(range(happen+1, len(t2_transformer)+1), t2_transformer[happen:], 'purple', label='Fault')
    plt.axhline(y=t2_limit_transformer, color='k', linestyle='--', label='Threshold')
    plt.axvline(x=happen, color='m', linestyle='-', label='Fault Time')
    plt.title('Enhanced Transformer - T² Statistics')
    plt.xlabel('Sample')
    plt.ylabel('T²')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # SPE PCA plot
    plt.subplot(2, 2, 3)
    plt.plot(range(1, happen+1), spe_pca[:happen], 'b', label='Normal')
    plt.plot(range(happen+1, len(spe_pca)+1), spe_pca[happen:], 'r', label='Fault')
    plt.axhline(y=spe_limit_pca, color='k', linestyle='--', label='Threshold')
    plt.axvline(x=happen, color='m', linestyle='-', label='Fault Time')
    plt.title('PCA - SPE Statistics')
    plt.xlabel('Sample')
    plt.ylabel('SPE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # SPE Transformer plot
    plt.subplot(2, 2, 4)
    plt.plot(range(1, happen+1), spe_transformer[:happen], 'g', label='Normal')
    plt.plot(range(happen+1, len(spe_transformer)+1), spe_transformer[happen:], 'purple', label='Fault')
    plt.axhline(y=spe_limit_transformer, color='k', linestyle='--', label='Threshold')
    plt.axvline(x=happen, color='m', linestyle='-', label='Fault Time')
    plt.title('Enhanced Transformer - SPE Statistics')
    plt.xlabel('Sample')
    plt.ylabel('SPE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comparison_fault_detection.png')
    print("Plot saved as comparison_fault_detection.png")
    plt.close()


def print_comparison_table(pca_false_rates, pca_miss_rates, transformer_false_rates, transformer_miss_rates,
                          pca_detection_times, transformer_detection_times):
    """Print comparison table for the different methods"""
    print("\n" + "="*80)
    print("{:<15} | {:<12} {:<12} | {:<12} {:<12}".format(
        "Method", "T² False(%)", "SPE False(%)", "T² Miss(%)", "SPE Miss(%)"))
    print("-"*80)
    print("{:<15} | {:<12.2f} {:<12.2f} | {:<12.2f} {:<12.2f}".format(
        "PCA", pca_false_rates[0], pca_false_rates[1], pca_miss_rates[0], pca_miss_rates[1]))
    print("{:<15} | {:<12.2f} {:<12.2f} | {:<12.2f} {:<12.2f}".format(
        "Enhanced Transf.", transformer_false_rates[0], transformer_false_rates[1], 
        transformer_miss_rates[0], transformer_miss_rates[1]))
    print("-"*80)
    print("{:<15} | {:<27} | {:<25}".format("Method", "Detection Time T²", "Detection Time SPE"))
    print("-"*80)
    det_t2_pca = pca_detection_times[0] if pca_detection_times[0] is not None else "Not detected"
    det_spe_pca = pca_detection_times[1] if pca_detection_times[1] is not None else "Not detected"
    det_t2_trans = transformer_detection_times[0] if transformer_detection_times[0] is not None else "Not detected"
    det_spe_trans = transformer_detection_times[1] if transformer_detection_times[1] is not None else "Not detected"
    print("{:<15} | {:<27} | {:<25}".format("PCA", det_t2_pca, det_spe_pca))
    print("{:<15} | {:<27} | {:<25}".format("Enhanced Transf.", det_t2_trans, det_spe_trans))
    print("="*80)


def main():
    """Main function to run the comparison between PCA and enhanced transformer-based fault detection"""
    start_time = time.time()
    
    # Load data
    X_train, X_test, happen = load_data(is_mock=True)  # Use mock data for demonstration
    print(f"Data loaded. Training shape: {X_train.shape}, Testing shape: {X_test.shape}")
    
    # Set up PCA model
    n_components = min(27, X_train.shape[1] - 1)  # Same compression as the transformer model
    print(f"Using {n_components} principal components for PCA")
    pca_model = PCA(n_components=X_train.shape[1])  # Keep all components for reconstruction
    pca_model.fit(X_train)
    
    # Project training data
    projected_train = pca_model.transform(X_train)
    
    # Calculate % variance explained by components
    explained_variance_ratio = pca_model.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    # Find number of components for desired variance (85% like in MATLAB code)
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
    
    # Initialize and load the enhanced transformer model
    print("Setting up enhanced transformer model...")
    input_dim = X_train.shape[1]
    hidden_dim = min(27, input_dim - 1)  
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EnhancedTransformerAutoencoder(input_dim, hidden_dim)
    
    # Try to load pre-trained model or train a new one
    try:
        model.load_state_dict(torch.load('enhanced_transformer_autoencoder.pth', map_location=device))
        print("Loaded pre-trained enhanced transformer model")
    except:
        print("Pre-trained model not found. Training a new model...")
        model, _, _ = train_enhanced_model(
            X_train, 
            epochs=50,
            batch_size=32,
            lr=0.001,
            hidden_dim=hidden_dim,
            validation_split=0.2
        )
        
        torch.save(model.state_dict(), 'enhanced_transformer_autoencoder.pth')
        print("Enhanced model trained and saved.")
    
    model.to(device)
    
    # Calculate variable importance weights for weighted SPE
    print("Calculating variable importance weights...")
    importance_weights = calculate_variable_importance(model, X_train, device)
    
    # Calculate transformer metrics with improved methods
    print("Calculating enhanced transformer metrics...")
    t2_train_transformer, Sigma_inv = calculate_t2_statistics(model, X_train, device)
    spe_train_transformer = calculate_weighted_spe(model, X_train, device, importance_weights)
    
    t2_test_transformer, _ = calculate_t2_statistics(model, X_test, device, Sigma_inv)
    spe_test_transformer = calculate_weighted_spe(model, X_test, device, importance_weights)
    
    # Calculate transformer limits with adaptive method
    t2_limit_transformer = calculate_control_limits(t2_train_transformer, method='adaptive', false_alarm_target=0.01)
    spe_limit_transformer = calculate_control_limits(spe_train_transformer, method='adaptive', false_alarm_target=0.01)
    print(f"Enhanced Transformer control limits: T² = {t2_limit_transformer:.2f}, SPE = {spe_limit_transformer:.2f}")
    
    # Calculate alarm rates
    pca_false_rates, pca_miss_rates = calculate_alarm_rates(
        t2_test_pca, spe_test_pca, t2_limit_pca, spe_limit_pca, happen)
    
    transformer_false_rates, transformer_miss_rates = calculate_alarm_rates(
        t2_test_transformer, spe_test_transformer, t2_limit_transformer, spe_limit_transformer, happen)
    
    # Calculate detection times with fewer consecutive samples required (3 instead of 5)
    pca_detection_times = calculate_detection_time(
        t2_test_pca, spe_test_pca, t2_limit_pca, spe_limit_pca, happen, consecutive_required=3)
    
    transformer_detection_times = calculate_detection_time(
        t2_test_transformer, spe_test_transformer, t2_limit_transformer, spe_limit_transformer, happen, consecutive_required=3)
    
    # Print comparison table
    print_comparison_table(
        pca_false_rates, pca_miss_rates, 
        transformer_false_rates, transformer_miss_rates,
        pca_detection_times, transformer_detection_times
    )
    
    # Plot comparison
    plot_comparison(
        t2_test_pca, spe_test_pca, 
        t2_test_transformer, spe_test_transformer,
        t2_limit_pca, spe_limit_pca, 
        t2_limit_transformer, spe_limit_transformer,
        happen
    )
    
    runtime = time.time() - start_time
    print(f"Total Runtime: {runtime:.2f} seconds")
    
    return {
        'pca_model': pca_model,
        'transformer_model': model,
        'pca_metrics': {
            't2_train': t2_train_pca,
            'spe_train': spe_train_pca,
            't2_test': t2_test_pca,
            'spe_test': spe_test_pca,
            't2_limit': t2_limit_pca,
            'spe_limit': spe_limit_pca,
            'false_rates': pca_false_rates,
            'miss_rates': pca_miss_rates,
            'detection_times': pca_detection_times
        },
        'transformer_metrics': {
            't2_train': t2_train_transformer,
            'spe_train': spe_train_transformer,
            't2_test': t2_test_transformer,
            'spe_test': spe_test_transformer,
            't2_limit': t2_limit_transformer,
            'spe_limit': spe_limit_transformer,
            'false_rates': transformer_false_rates,
            'miss_rates': transformer_miss_rates,
            'detection_times': transformer_detection_times,
            'importance_weights': importance_weights.cpu().numpy() if hasattr(importance_weights, 'cpu') else importance_weights
        },
        'runtime': runtime
    }


if __name__ == "__main__":
    main() 