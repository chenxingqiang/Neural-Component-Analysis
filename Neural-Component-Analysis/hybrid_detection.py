import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from enhanced_transformer_autoencoder import (
    EnhancedTransformerAutoencoder, 
    calculate_weighted_spe,
)
from enhanced_transformer_detection import (
    load_data, calculate_variable_importance,
    calculate_control_limits, calculate_alarm_rates,
    calculate_detection_time
)

def calculate_pca_t2(data, pca_model, projected_train, latent_dim):
    """Calculate T2 statistic using PCA model"""
    # Project data to PCA space
    projected = pca_model.transform(data)
    
    # Only use components up to latent_dim
    projected_latent = projected[:, :latent_dim]
    
    # Calculate T2 statistic for each sample (Hotelling's T2)
    variances = np.var(projected_train[:, :latent_dim], axis=0) + 1e-8
    t2_values = np.sum((projected_latent**2) / variances, axis=1)
    
    return t2_values

def plot_hybrid_metrics(t2_test, spe_test, t2_limit, spe_limit, happen, title="Hybrid Detection Model"):
    """Plot T² and SPE metrics for the hybrid model"""
    plt.figure(figsize=(12, 8))
    
    # T2 plot (PCA-based)
    plt.subplot(2, 1, 1)
    plt.plot(range(1, happen+1), t2_test[:happen], 'g', label='Normal')
    plt.plot(range(happen+1, len(t2_test)+1), t2_test[happen:], 'r', label='Fault')
    plt.axhline(y=t2_limit, color='k', linestyle='--', label='Control Limit')
    plt.axvline(x=happen, color='m', linestyle='-', label='Fault Occurrence')
    plt.title(f'{title} - T² Statistics (PCA optimized)')
    plt.xlabel('Sample')
    plt.ylabel('T²')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # SPE plot (Transformer-based)
    plt.subplot(2, 1, 2)
    plt.plot(range(1, happen+1), spe_test[:happen], 'g', label='Normal')
    plt.plot(range(happen+1, len(spe_test)+1), spe_test[happen:], 'r', label='Fault')
    plt.axhline(y=spe_limit, color='k', linestyle='--', label='Control Limit')
    plt.axvline(x=happen, color='m', linestyle='-', label='Fault Occurrence')
    plt.title(f'{title} - SPE Statistics (Transformer optimized)')
    plt.xlabel('Sample')
    plt.ylabel('SPE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}_fault_detection.png')
    print(f"Chart saved as {title.lower().replace(' ', '_')}_fault_detection.png")
    plt.close()

def main():
    """Hybrid approach combining PCA for T² and enhanced transformer for SPE"""
    print("Loading data...")
    X_train, X_test, happen = load_data(is_mock=True)
    
    # ===================== PCA for T² calculation ======================
    print("Setting up PCA model for T² calculation...")
    n_components = min(27, X_train.shape[1] - 1)
    pca_model = PCA(n_components=X_train.shape[1])
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
    
    # Calculate T² statistics using PCA
    print("Calculating T² statistics using PCA...")
    t2_train = calculate_pca_t2(X_train, pca_model, projected_train, latent_dim)
    t2_test = calculate_pca_t2(X_test, pca_model, projected_train, latent_dim)
    
    # Calculate T² limit
    t2_limit = calculate_control_limits(t2_train, method='kde')
    print(f"PCA-based T² control limit: {t2_limit:.2f}")
    
    # ================= Enhanced Transformer for SPE ===================
    print("Setting up enhanced transformer model for SPE calculation...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model parameters
    input_dim = X_train.shape[1]
    hidden_dim = min(27, input_dim - 1)
    
    # Load the enhanced transformer model
    model = EnhancedTransformerAutoencoder(input_dim, hidden_dim)
    try:
        model.load_state_dict(torch.load('enhanced_transformer_autoencoder.pth', map_location=device))
        print("Enhanced transformer model loaded successfully")
    except:
        print("Failed to load model. Please run enhanced_transformer_detection.py first")
        return
    
    model.to(device)
    
    # Calculate variable importance weights for weighted SPE
    print("Calculating variable importance weights...")
    importance_weights = calculate_variable_importance(model, X_train, device)
    
    # Calculate SPE metrics
    print("Calculating weighted SPE metrics...")
    spe_train = calculate_weighted_spe(model, X_train, device, importance_weights)
    spe_test = calculate_weighted_spe(model, X_test, device, importance_weights)
    
    # Calculate SPE limit using adaptive method
    spe_limit = calculate_control_limits(spe_train, method='adaptive', false_alarm_target=0.01)
    print(f"Transformer-based SPE control limit: {spe_limit:.2f}")
    
    # =================== Performance Evaluation ======================
    # Calculate alarm rates
    false_rates, miss_rates = calculate_alarm_rates(
        t2_test, spe_test, t2_limit, spe_limit, happen)
    
    print("\n===== Hybrid Model Performance =====")
    print(f"T² False Alarm Rate: {false_rates[0]:.2f}%")
    print(f"SPE False Alarm Rate: {false_rates[1]:.2f}%")
    print(f"T² Miss Rate: {miss_rates[0]:.2f}%")
    print(f"SPE Miss Rate: {miss_rates[1]:.2f}%")
    
    # Calculate detection time
    det_time_t2, det_time_spe = calculate_detection_time(
        t2_test, spe_test, t2_limit, spe_limit, happen, consecutive_required=3)
    
    print(f"\nT² Detection Time: {det_time_t2 if det_time_t2 is not None else 'Not detected'}")
    print(f"SPE Detection Time: {det_time_spe if det_time_spe is not None else 'Not detected'}")
    
    # Plot results
    plot_hybrid_metrics(t2_test, spe_test, t2_limit, spe_limit, happen, "Hybrid Model")
    
    # Create a consolidated performance table
    performance_data = {
        'Method': ['PCA only', 'Enhanced Transformer only', 'Hybrid Model'],
        'T² False Alarm (%)': [0.00, 3.75, false_rates[0]],
        'SPE False Alarm (%)': [0.00, 6.88, false_rates[1]],
        'T² Miss Rate (%)': [19.71, 97.94, miss_rates[0]],
        'SPE Miss Rate (%)': [100.00, 0.59, miss_rates[1]],
        'T² Detection Time': [2, 'Not detected', det_time_t2 if det_time_t2 is not None else 'Not detected'],
        'SPE Detection Time': ['Not detected', 0, det_time_spe if det_time_spe is not None else 'Not detected']
    }
    
    # Save performance data for future reference
    import json
    with open('hybrid_performance.json', 'w') as f:
        json_compatible_data = {
            'Method': performance_data['Method'],
            'T2_False_Alarm': performance_data['T² False Alarm (%)'],
            'SPE_False_Alarm': performance_data['SPE False Alarm (%)'],
            'T2_Miss_Rate': performance_data['T² Miss Rate (%)'],
            'SPE_Miss_Rate': performance_data['SPE Miss Rate (%)'],
            'T2_Detection_Time': [str(t) for t in performance_data['T² Detection Time']],
            'SPE_Detection_Time': [str(t) for t in performance_data['SPE Detection Time']]
        }
        json.dump(json_compatible_data, f, indent=2)
    
    print("\nPerformance comparison:")
    print("-" * 100)
    print(f"{'Method':<25} | {'T² False (%)':<12} {'SPE False (%)':<12} | {'T² Miss (%)':<12} {'SPE Miss (%)':<12} | {'T² Detection':<12} {'SPE Detection':<12}")
    print("-" * 100)
    
    for i in range(len(performance_data['Method'])):
        method = performance_data['Method'][i]
        t2_false = performance_data['T² False Alarm (%)'][i]
        spe_false = performance_data['SPE False Alarm (%)'][i]
        t2_miss = performance_data['T² Miss Rate (%)'][i]
        spe_miss = performance_data['SPE Miss Rate (%)'][i]
        t2_det = performance_data['T² Detection Time'][i]
        spe_det = performance_data['SPE Detection Time'][i]
        
        print(f"{method:<25} | {t2_false:<12.2f} {spe_false:<12.2f} | {t2_miss:<12.2f} {spe_miss:<12.2f} | {str(t2_det):<12} {str(spe_det):<12}")
    
    print("-" * 100)
    print("Hybrid Model combines: PCA-based T² + Transformer-based SPE for optimal fault detection performance")
    
    return {
        'pca_model': pca_model,
        'transformer_model': model,
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
    }

if __name__ == "__main__":
    main() 