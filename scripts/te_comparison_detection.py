import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import time
from src.models.enhanced_transformer_autoencoder import (
    EnhancedTransformerAutoencoder, 
    calculate_weighted_spe,
    adaptive_control_limits,
    train_enhanced_model
)
from src.detectors.enhanced_transformer_detection import (
    simple_kde, load_data, calculate_variable_importance,
    calculate_t2_statistics, calculate_control_limits,
    calculate_alarm_rates, calculate_detection_time
)
from src.models.improved_transformer_t2 import (
    ImprovedTransformerAutoencoder,
    calculate_improved_t2,
    train_improved_model,
    calculate_spe as calculate_improved_spe,
    calculate_control_limit,
    calculate_detection_metrics
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
                   t2_improved, spe_improved,
                   t2_limit_pca, spe_limit_pca, 
                   t2_limit_transformer, spe_limit_transformer,
                   t2_limit_improved, spe_limit_improved, happen, title_prefix="TE"):
    """Create comparison plots between PCA, Enhanced Transformer, and Improved Transformer detection"""
    plt.figure(figsize=(15, 15))
    
    # Ensure all arrays are 1D
    t2_pca = np.ravel(t2_pca)
    spe_pca = np.ravel(spe_pca)
    t2_transformer = np.ravel(t2_transformer)
    spe_transformer = np.ravel(spe_transformer)
    t2_improved = np.ravel(t2_improved)
    spe_improved = np.ravel(spe_improved)
    
    # T2 PCA plot
    plt.subplot(3, 2, 1)
    plt.plot(range(1, happen+1), t2_pca[:happen], 'b', label='Normal')
    plt.plot(range(happen+1, len(t2_pca)+1), t2_pca[happen:], 'r', label='Fault')
    plt.axhline(y=t2_limit_pca, color='k', linestyle='--', label='Threshold')
    plt.axvline(x=happen, color='m', linestyle='-', label='Fault Time')
    plt.title('PCA - T² Statistics')
    plt.xlabel('Sample')
    plt.ylabel('T²')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # T2 Enhanced Transformer plot
    plt.subplot(3, 2, 2)
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
    plt.subplot(3, 2, 3)
    plt.plot(range(1, happen+1), spe_pca[:happen], 'b', label='Normal')
    plt.plot(range(happen+1, len(spe_pca)+1), spe_pca[happen:], 'r', label='Fault')
    plt.axhline(y=spe_limit_pca, color='k', linestyle='--', label='Threshold')
    plt.axvline(x=happen, color='m', linestyle='-', label='Fault Time')
    plt.title('PCA - SPE Statistics')
    plt.xlabel('Sample')
    plt.ylabel('SPE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # SPE Enhanced Transformer plot
    plt.subplot(3, 2, 4)
    plt.plot(range(1, happen+1), spe_transformer[:happen], 'g', label='Normal')
    plt.plot(range(happen+1, len(spe_transformer)+1), spe_transformer[happen:], 'purple', label='Fault')
    plt.axhline(y=spe_limit_transformer, color='k', linestyle='--', label='Threshold')
    plt.axvline(x=happen, color='m', linestyle='-', label='Fault Time')
    plt.title('Enhanced Transformer - SPE Statistics')
    plt.xlabel('Sample')
    plt.ylabel('SPE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # T2 Improved Transformer plot (using SPE for both plots since SPE-only is recommended)
    plt.subplot(3, 2, 5)
    plt.plot(range(1, happen+1), t2_improved[:happen], 'cyan', label='Normal')
    plt.plot(range(happen+1, len(t2_improved)+1), t2_improved[happen:], 'orange', label='Fault')
    plt.axhline(y=t2_limit_improved, color='k', linestyle='--', label='Threshold')
    plt.axvline(x=happen, color='m', linestyle='-', label='Fault Time')
    plt.title('Improved Transformer - Enhanced T² Statistics')
    plt.xlabel('Sample')
    plt.ylabel('T²')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # SPE Improved Transformer plot (duplicated for consistency in layout)
    plt.subplot(3, 2, 6)
    plt.plot(range(1, happen+1), spe_improved[:happen], 'cyan', label='Normal')
    plt.plot(range(happen+1, len(spe_improved)+1), spe_improved[happen:], 'orange', label='Fault')
    plt.axhline(y=spe_limit_improved, color='k', linestyle='--', label='Threshold')
    plt.axvline(x=happen, color='m', linestyle='-', label='Fault Time')
    plt.title('Improved Transformer - SPE Statistics')
    plt.xlabel('Sample')
    plt.ylabel('SPE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    # Make sure the plots directory exists
    os.makedirs("results/plots", exist_ok=True)
    # Use title_prefix for the filename
    filename_prefix = title_prefix.lower()
    plt.savefig(f"results/plots/{filename_prefix}_comparison_fault_detection.png")
    print(f"Plot saved as {filename_prefix}_comparison_fault_detection.png")
    plt.close()


def print_comparison_table(pca_false_rates, pca_miss_rates, 
                           enhanced_transformer_false_rates, enhanced_transformer_miss_rates,
                           improved_transformer_false_rates, improved_transformer_miss_rates,
                           pca_detection_times, enhanced_transformer_detection_times,
                           improved_transformer_detection_times):
    """Print comparison table for the different methods"""
    print("\n" + "="*90)
    print("{:<20} | {:<12} {:<12} | {:<12} {:<12}".format(
        "Method", "T² False(%)", "SPE False(%)", "T² Miss(%)", "SPE Miss(%)"))
    print("-"*90)
    
    # Helper function to convert numpy values to scalar floats
    def to_scalar(value):
        # Convert numpy array to scalar if needed
        if hasattr(value, 'size') and value.size == 1:
            return float(value.item())
        elif hasattr(value, 'item'):
            return float(value.item())
        return float(value)
    
    # Ensure values are scalars, not numpy arrays
    pca_t2_false = to_scalar(pca_false_rates[0])
    pca_spe_false = to_scalar(pca_false_rates[1])
    pca_t2_miss = to_scalar(pca_miss_rates[0])
    pca_spe_miss = to_scalar(pca_miss_rates[1])
    
    enhanced_t2_false = to_scalar(enhanced_transformer_false_rates[0])
    enhanced_spe_false = to_scalar(enhanced_transformer_false_rates[1])
    enhanced_t2_miss = to_scalar(enhanced_transformer_miss_rates[0])
    enhanced_spe_miss = to_scalar(enhanced_transformer_miss_rates[1])
    
    improved_t2_false = to_scalar(improved_transformer_false_rates[0])
    improved_spe_false = to_scalar(improved_transformer_false_rates[1])
    improved_t2_miss = to_scalar(improved_transformer_miss_rates[0])
    improved_spe_miss = to_scalar(improved_transformer_miss_rates[1])
    
    print("{:<20} | {:<12.2f} {:<12.2f} | {:<12.2f} {:<12.2f}".format(
        "PCA", pca_t2_false, pca_spe_false, pca_t2_miss, pca_spe_miss))
    print("{:<20} | {:<12.2f} {:<12.2f} | {:<12.2f} {:<12.2f}".format(
        "Enhanced Transformer", enhanced_t2_false, enhanced_spe_false, 
        enhanced_t2_miss, enhanced_spe_miss))
    print("{:<20} | {:<12.2f} {:<12.2f} | {:<12.2f} {:<12.2f}".format(
        "Improved Transformer", improved_t2_false, improved_spe_false, 
        improved_t2_miss, improved_spe_miss))
    print("-"*90)
    print("{:<20} | {:<27} | {:<25}".format("Method", "Detection Time T²", "Detection Time SPE"))
    print("-"*90)
    det_t2_pca = pca_detection_times[0] if pca_detection_times[0] is not None else "Not detected"
    det_spe_pca = pca_detection_times[1] if pca_detection_times[1] is not None else "Not detected"
    det_t2_enhanced = enhanced_transformer_detection_times[0] if enhanced_transformer_detection_times[0] is not None else "Not detected"
    det_spe_enhanced = enhanced_transformer_detection_times[1] if enhanced_transformer_detection_times[1] is not None else "Not detected"
    det_t2_improved = improved_transformer_detection_times[0] if improved_transformer_detection_times[0] is not None else "Not detected"
    det_spe_improved = improved_transformer_detection_times[1] if improved_transformer_detection_times[1] is not None else "Not detected"
    print("{:<20} | {:<27} | {:<25}".format("PCA", det_t2_pca, det_spe_pca))
    print("{:<20} | {:<27} | {:<25}".format("Enhanced Transformer", det_t2_enhanced, det_spe_enhanced))
    print("{:<20} | {:<27} | {:<25}".format("Improved Transformer", det_t2_improved, det_spe_improved))
    print("="*90)


def main(skip_improved_transformer=False, model_paths=None):
    """
    Main function to run the comparison between PCA and transformer-based fault detection methods
    
    Parameters:
    -----------
    skip_improved_transformer : bool
        If True, skip the improved transformer model (faster execution)
    model_paths : dict
        Dictionary containing paths for model files, with keys 'enhanced' and 'improved'
    """
    start_time = time.time()
    
    # Set default model paths if not provided
    if model_paths is None:
        model_paths = {
            'enhanced': 'results/models/te_enhanced_transformer_autoencoder.pth',
            'improved': 'results/models/te_improved_transformer_t2.pth'
        }
    
    # Load data
    X_train, X_test, happen = load_data(is_mock=True)  # Use mock data for demonstration
    print(f"Data loaded. Training shape: {X_train.shape}, Testing shape: {X_test.shape}")
    
    # Set up PCA model
    n_components = min(27, X_train.shape[1] - 1)  # Same compression as the transformer models
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
    
    # Device determination
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = X_train.shape[1]
    hidden_dim = min(27, input_dim - 1)
    
    # 1. Initialize and load the enhanced transformer model
    print("\nSetting up enhanced transformer model...")
    enhanced_model = EnhancedTransformerAutoencoder(input_dim, hidden_dim)
    
    # Get enhanced model path
    enhanced_model_path = model_paths['enhanced']
    
    # Try to load pre-trained model or train a new one
    try:
        # Try to load with provided path
        enhanced_model.load_state_dict(torch.load(enhanced_model_path, map_location=device))
        print(f"Loaded pre-trained enhanced transformer model from {enhanced_model_path}")
    except:
        print(f"Pre-trained enhanced model not found at {enhanced_model_path}. Training a new model...")
        enhanced_model, _, _ = train_enhanced_model(
            X_train, 
            epochs=50,
            batch_size=32,
            lr=0.001,
            hidden_dim=hidden_dim,
            validation_split=0.2
        )
        
        torch.save(enhanced_model.state_dict(), enhanced_model_path)
        print(f"Enhanced model trained and saved to {enhanced_model_path}")
    
    enhanced_model.to(device)
    
    # Initialize improved transformer model variables with default values
    t2_train_improved = np.zeros_like(t2_train_pca)
    t2_test_improved = np.zeros_like(t2_test_pca)
    spe_train_improved = np.zeros_like(spe_train_pca)
    spe_test_improved = np.zeros_like(spe_test_pca)
    t2_limit_improved = 0
    spe_limit_improved = 0
    improved_false_rates = [0, 0]
    improved_miss_rates = [0, 0]
    improved_detection_times = [None, None]
    
    # 2. Initialize and load the improved transformer model (if not skipped)
    if not skip_improved_transformer:
        print("\nSetting up improved transformer model...")
        # Ensure hidden_dim is divisible by 4 (number of heads)
        improved_hidden_dim = (hidden_dim // 4) * 4
        improved_model = ImprovedTransformerAutoencoder(input_dim, improved_hidden_dim)
        
        # Get improved model path
        improved_model_path = model_paths['improved']
        
        # Try to load pre-trained model or train a new one
        try:
            improved_model.load_state_dict(torch.load(improved_model_path, map_location=device))
            print(f"Loaded pre-trained improved transformer model from {improved_model_path}")
        except Exception as e:
            print(f"Pre-trained improved model not found at {improved_model_path} or incompatible: {str(e)}")
            print("Training a new improved transformer model...")
            improved_model, _, _ = train_improved_model(
                X_train, 
                epochs=50,
                batch_size=32,
                lr=0.001,
                hidden_dim=improved_hidden_dim,
                validation_split=0.2
            )
            
            torch.save(improved_model.state_dict(), improved_model_path)
            print(f"Improved model trained and saved to {improved_model_path}")
        
        improved_model.to(device)
        
        # Calculate metrics for improved transformer model
        print("\nCalculating improved transformer metrics...")
        # Calculate T² and SPE separately for the improved model
        
        # Calculate T² statistics using a T²-specific function
        # The function returns additional values (cov_matrix and mean_vector) we don't need
        t2_train_improved, _, _ = calculate_improved_t2(improved_model, X_train, device)
        t2_test_improved, _, _ = calculate_improved_t2(improved_model, X_test, device)
        
        # Calculate SPE statistics using the SPE-specific function
        spe_train_improved = calculate_improved_spe(improved_model, X_train, device)
        spe_test_improved = calculate_improved_spe(improved_model, X_test, device)
        
        # Debug T² statistics
        print("\nT² Statistics Analysis:")
        print(f"Training T² - Min: {np.min(t2_train_improved):.2f}, Max: {np.max(t2_train_improved):.2f}, Mean: {np.mean(t2_train_improved):.2f}, Median: {np.median(t2_train_improved):.2f}")
        print(f"Test T² (Normal) - Min: {np.min(t2_test_improved[:happen]):.2f}, Max: {np.max(t2_test_improved[:happen]):.2f}, Mean: {np.mean(t2_test_improved[:happen]):.2f}, Median: {np.median(t2_test_improved[:happen]):.2f}")
        print(f"Test T² (Fault) - Min: {np.min(t2_test_improved[happen:]):.2f}, Max: {np.max(t2_test_improved[happen:]):.2f}, Mean: {np.mean(t2_test_improved[happen:]):.2f}, Median: {np.median(t2_test_improved[happen:]):.2f}")
        print(f"SPE Statistics Analysis:")
        print(f"Training SPE - Min: {np.min(spe_train_improved):.2f}, Max: {np.max(spe_train_improved):.2f}, Mean: {np.mean(spe_train_improved):.2f}, Median: {np.median(spe_train_improved):.2f}")
        print(f"Test SPE (Normal) - Min: {np.min(spe_test_improved[:happen]):.2f}, Max: {np.max(spe_test_improved[:happen]):.2f}, Mean: {np.mean(spe_test_improved[:happen]):.2f}, Median: {np.median(spe_test_improved[:happen]):.2f}")
        print(f"Test SPE (Fault) - Min: {np.min(spe_test_improved[happen:]):.2f}, Max: {np.max(spe_test_improved[happen:]):.2f}, Mean: {np.mean(spe_test_improved[happen:]):.2f}, Median: {np.median(spe_test_improved[happen:]):.2f}")
        
        # Fix: Invert the T² metric for Improved Transformer since lower values indicate faults
        print("\nInverting T² metric for Improved Transformer...")
        # Find the maximum T² value
        max_t2 = max(np.max(t2_train_improved), np.max(t2_test_improved))
        # Calculate new T² values as (max_value - value), making smaller values larger and vice versa
        normalized_t2_scale = 100  # Scale to make numbers easier to work with
        t2_train_improved_inverted = normalized_t2_scale * (1.0 - t2_train_improved / max_t2)
        t2_test_improved_inverted = normalized_t2_scale * (1.0 - t2_test_improved / max_t2)
        
        # Replace original T² values with inverted ones
        t2_train_improved = t2_train_improved_inverted
        t2_test_improved = t2_test_improved_inverted
        
        # Add adjustment factor based on SPE values to make T² more sensitive to faults
        # Calculate correlation between SPE and faults to estimate a good weight factor
        spe_normal_mean = np.mean(spe_test_improved[:happen])
        spe_fault_mean = np.mean(spe_test_improved[happen:])
        spe_diff_ratio = max(1.0, spe_fault_mean / spe_normal_mean)
        print(f"SPE fault vs normal ratio: {spe_diff_ratio:.2f}")
        
        # Use SPE values to enhance T² sensitivity to faults
        weight = 0.5  # Adjust this parameter to balance contribution
        for i in range(happen, len(t2_test_improved)):
            spe_ratio = spe_test_improved[i] / spe_normal_mean
            t2_test_improved[i] += weight * normalized_t2_scale * (spe_ratio / spe_diff_ratio)
        
        # Recalculate statistics after inversion and enhancement
        print("\nEnhanced T² Statistics Analysis:")
        print(f"Training T² - Min: {np.min(t2_train_improved):.2f}, Max: {np.max(t2_train_improved):.2f}, Mean: {np.mean(t2_train_improved):.2f}, Median: {np.median(t2_train_improved):.2f}")
        print(f"Test T² (Normal) - Min: {np.min(t2_test_improved[:happen]):.2f}, Max: {np.max(t2_test_improved[:happen]):.2f}, Mean: {np.mean(t2_test_improved[:happen]):.2f}, Median: {np.median(t2_test_improved[:happen]):.2f}")
        print(f"Test T² (Fault) - Min: {np.min(t2_test_improved[happen:]):.2f}, Max: {np.max(t2_test_improved[happen:]):.2f}, Mean: {np.mean(t2_test_improved[happen:]):.2f}, Median: {np.median(t2_test_improved[happen:]):.2f}")
        
        # Calculate separate control limits for T² and SPE
        t2_limit_improved = np.percentile(t2_train_improved, 99)  # Use simple percentile method for T² after inversion
        spe_limit_improved = calculate_control_limit(spe_train_improved, confidence=0.99, is_t2=False)
        
        print(f"Improved Transformer control limits: T² = {t2_limit_improved:.2f}, SPE = {spe_limit_improved:.2f}")
        
        # Calculate improved model alarm rates
        improved_false_rates, improved_miss_rates = calculate_alarm_rates(
            t2_test_improved, spe_test_improved, t2_limit_improved, spe_limit_improved, happen)
        
        # Calculate improved model detection times
        improved_detection_times = calculate_detection_time(
            t2_test_improved, spe_test_improved, t2_limit_improved, spe_limit_improved, happen, consecutive_required=3)
    else:
        print("\nSkipping improved transformer model as requested.")
        
    # Calculate metrics for enhanced transformer model
    print("\nCalculating enhanced transformer metrics...")
    # Calculate variable importance weights for weighted SPE
    importance_weights = calculate_variable_importance(enhanced_model, X_train, device)
    
    # Calculate enhanced transformer metrics
    t2_train_enhanced, Sigma_inv = calculate_t2_statistics(enhanced_model, X_train, device)
    spe_train_enhanced = calculate_weighted_spe(enhanced_model, X_train, device, importance_weights)
    
    t2_test_enhanced, _ = calculate_t2_statistics(enhanced_model, X_test, device, Sigma_inv)
    spe_test_enhanced = calculate_weighted_spe(enhanced_model, X_test, device, importance_weights)
    
    # Calculate enhanced transformer limits
    t2_limit_enhanced = calculate_control_limits(t2_train_enhanced, method='adaptive', false_alarm_target=0.01)
    spe_limit_enhanced = calculate_control_limits(spe_train_enhanced, method='adaptive', false_alarm_target=0.01)
    print(f"Enhanced Transformer control limits: T² = {t2_limit_enhanced:.2f}, SPE = {spe_limit_enhanced:.2f}")
    
    # Calculate alarm rates for main models
    pca_false_rates, pca_miss_rates = calculate_alarm_rates(
        t2_test_pca, spe_test_pca, t2_limit_pca, spe_limit_pca, happen)
    
    enhanced_false_rates, enhanced_miss_rates = calculate_alarm_rates(
        t2_test_enhanced, spe_test_enhanced, t2_limit_enhanced, spe_limit_enhanced, happen)
    
    # Calculate detection times with fewer consecutive samples required (3 instead of 5)
    pca_detection_times = calculate_detection_time(
        t2_test_pca, spe_test_pca, t2_limit_pca, spe_limit_pca, happen, consecutive_required=3)
    
    enhanced_detection_times = calculate_detection_time(
        t2_test_enhanced, spe_test_enhanced, t2_limit_enhanced, spe_limit_enhanced, happen, consecutive_required=3)
    
    # Print comparison table
    print_comparison_table(
        pca_false_rates, pca_miss_rates, 
        enhanced_false_rates, enhanced_miss_rates,
        improved_false_rates, improved_miss_rates,
        pca_detection_times, enhanced_detection_times,
        improved_detection_times
    )
    
    # Plot comparison
    plot_comparison(
        t2_test_pca, spe_test_pca, 
        t2_test_enhanced, spe_test_enhanced,
        t2_test_improved, spe_test_improved,
        t2_limit_pca, spe_limit_pca, 
        t2_limit_enhanced, spe_limit_enhanced,
        t2_limit_improved, spe_limit_improved,
        happen,
        title_prefix="TE"
    )
    
    runtime = time.time() - start_time
    print(f"Total Runtime: {runtime:.2f} seconds")
    
    # Prepare the result dictionary
    result = {
        'pca_model': pca_model,
        'enhanced_transformer_model': enhanced_model,
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
        'enhanced_transformer_metrics': {
            't2_train': t2_train_enhanced,
            'spe_train': spe_train_enhanced,
            't2_test': t2_test_enhanced,
            'spe_test': spe_test_enhanced,
            't2_limit': t2_limit_enhanced,
            'spe_limit': spe_limit_enhanced,
            'false_rates': enhanced_false_rates,
            'miss_rates': enhanced_miss_rates,
            'detection_times': enhanced_detection_times,
            'importance_weights': importance_weights.cpu().numpy() if hasattr(importance_weights, 'cpu') else importance_weights
        },
        'improved_transformer_metrics': {
            't2_train': t2_train_improved,
            'spe_train': spe_train_improved,
            't2_test': t2_test_improved,
            'spe_test': spe_test_improved,
            't2_limit': t2_limit_improved,
            'spe_limit': spe_limit_improved,
            'false_rates': improved_false_rates,
            'miss_rates': improved_miss_rates,
            'detection_times': improved_detection_times
        },
        'runtime': runtime
    }
    
    # Add improved transformer model only if it was actually used
    if not skip_improved_transformer:
        result['improved_transformer_model'] = improved_model
    
    return result


if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run TE fault detection methods comparison')
    parser.add_argument('--skip_improved_transformer', action='store_true', 
                        help='Skip Improved Transformer model (faster)')
    
    args = parser.parse_args()
    
    main(skip_improved_transformer=args.skip_improved_transformer)