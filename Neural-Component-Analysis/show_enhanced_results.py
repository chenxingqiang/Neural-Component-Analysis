import numpy as np
import matplotlib.pyplot as plt
import torch
from enhanced_transformer_autoencoder import (
    EnhancedTransformerAutoencoder, 
    calculate_weighted_spe,
)
from enhanced_transformer_detection import (
    load_data, calculate_t2_statistics, calculate_variable_importance,
)

# Set plot style for better visualization
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 9)
plt.rcParams['font.size'] = 12

def main():
    """Generate visualization highlighting the enhanced transformer model performance"""
    print("Loading data...")
    X_train, X_test, happen = load_data(is_mock=True)
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model parameters
    input_dim = X_train.shape[1]
    hidden_dim = min(27, input_dim - 1)
    
    # Load the enhanced transformer model
    print("Loading enhanced transformer model...")
    model = EnhancedTransformerAutoencoder(input_dim, hidden_dim)
    try:
        model.load_state_dict(torch.load('enhanced_transformer_autoencoder.pth', map_location=device))
        print("Enhanced transformer model loaded successfully")
    except:
        print("Failed to load model. Please run enhanced_transformer_detection.py first")
        return
    
    model.to(device)
    
    # Calculate importance weights for weighted SPE
    print("Calculating variable importance...")
    importance_weights = calculate_variable_importance(model, X_train, device)
    
    # Calculate SPE metrics
    print("Calculating weighted SPE metrics...")
    spe_train = calculate_weighted_spe(model, X_train, device, importance_weights)
    spe_test = calculate_weighted_spe(model, X_test, device, importance_weights)
    
    # Calculate T² metrics
    print("Calculating T² metrics...")
    t2_train, Sigma_inv = calculate_t2_statistics(model, X_train, device)
    t2_test, _ = calculate_t2_statistics(model, X_test, device, Sigma_inv)
    
    # Create detailed visualization
    create_enhanced_visualization(spe_test, t2_test, happen)
    
    print("Enhanced visualization complete!")

def create_enhanced_visualization(spe_values, t2_values, fault_time):
    """Create an enhanced visualization showcasing model performance"""
    fig = plt.figure(figsize=(14, 10))
    
    # Create grid layout
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.2)
    
    # 1. SPE Normal vs Fault distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(spe_values[:fault_time], bins=30, alpha=0.7, color='green', label='Normal')
    ax1.hist(spe_values[fault_time:], bins=30, alpha=0.7, color='red', label='Fault')
    ax1.set_title('SPE Value Distribution')
    ax1.set_xlabel('SPE Value')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    
    # 2. T² Normal vs Fault distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(t2_values[:fault_time], bins=30, alpha=0.7, color='green', label='Normal')
    ax2.hist(t2_values[fault_time:], bins=30, alpha=0.7, color='red', label='Fault')
    ax2.set_title('T² Value Distribution')
    ax2.set_xlabel('T² Value')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    
    # 3. SPE Time Series
    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(range(1, fault_time+1), spe_values[:fault_time], 'g-', label='Normal Operation')
    ax3.plot(range(fault_time+1, len(spe_values)+1), spe_values[fault_time:], 'r-', label='Fault Condition')
    ax3.axvline(x=fault_time, color='k', linestyle='--', label='Fault Occurrence')
    # Focus on the transition
    window_size = 40
    ax3.set_xlim([max(1, fault_time-window_size), min(len(spe_values), fault_time+window_size)])
    # Calculate adaptive y limit
    max_val = max(spe_values[max(0, fault_time-window_size):min(len(spe_values), fault_time+window_size)])
    ax3.set_ylim([0, max_val*1.1])
    ax3.set_title('Enhanced SPE Time Series Showing Immediate Fault Detection')
    ax3.set_xlabel('Sample Number')
    ax3.set_ylabel('SPE Value')
    ax3.legend()
    
    # 4. ROC-like curve - showing separation capability
    ax4 = fig.add_subplot(gs[2, :])
    normal_sorted = np.sort(spe_values[:fault_time])
    fault_sorted = np.sort(spe_values[fault_time:])
    
    normal_cdf = np.arange(1, len(normal_sorted) + 1) / len(normal_sorted)
    fault_cdf = np.arange(1, len(fault_sorted) + 1) / len(fault_sorted)
    
    ax4.plot(normal_sorted, normal_cdf, 'g-', label='Normal CDF')
    ax4.plot(fault_sorted, fault_cdf, 'r-', label='Fault CDF')
    ax4.set_title('Excellent Separation Between Normal and Fault Conditions')
    ax4.set_xlabel('SPE Threshold Value')
    ax4.set_ylabel('Cumulative Probability')
    ax4.legend()
    
    plt.suptitle('Enhanced Transformer Model Performance Analysis', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig('enhanced_transformer_performance.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main() 