#!/usr/bin/env python3
"""
Transformer-based Fault Detection Comparison Framework

This module implements a comparison framework specifically focused on transformer-based models
for fault detection on both SECOM and TE datasets, including:

1. Basic Transformer Autoencoder
   - Standard transformer structure with self-attention
   - Evaluates both T² and SPE (Q) statistics

2. Enhanced Transformer Autoencoder
   - Dual-path architecture (transformer + MLP)
   - Feature importance weighting for SPE calculation
   - Residual connections for better gradient flow

3. PCA (as baseline)
   - Implemented for performance reference
   - Uses kernel density estimation for control limits

The framework evaluates all methods on:
- False alarm rates (percentage of normal samples incorrectly flagged)
- Miss rates (percentage of fault samples not detected)
- Detection times (samples after fault occurrence until detection)

Each model is evaluated on both SECOM and TE datasets for comprehensive comparison.

Usage:
  python transformer_comparison_detection.py [--dataset secom|te] [--skip_basic]
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import DataLoader, TensorDataset
import os
import time
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats

# Import necessary modules
from neural_component_analysis.models.transformer_autoencoder import TransformerAutoencoder
from neural_component_analysis.models.enhanced_transformer_autoencoder import EnhancedTransformerAutoencoder


def load_secom_data(data_dir='data/secom'):
    """
    Load the processed SECOM data
    """
    print(f"Loading processed SECOM data from {data_dir}")
    
    try:
        # Load training data (normal samples only)
        X_train = np.loadtxt(f"{data_dir}/train/d00.dat")
        
        # Load test data
        X_test = np.loadtxt(f"{data_dir}/test/d01_te.dat")
        
        # Load fault occurrence sample if available
        try:
            with open(f"{data_dir}/test/happen.txt", 'r') as f:
                happen = int(f.read().strip())
        except FileNotFoundError:
            # Default fault occurrence is at sample 20
            print("Fault occurrence file not found, using default value of 20")
            happen = 20
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        print(f"Fault occurrence at sample: {happen}")
        
        # Create test labels based on fault occurrence
        test_labels = np.zeros(X_test.shape[0])
        test_labels[happen:] = 1  # Set labels after fault occurrence to 1
        
        # Standardize data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, test_labels, happen
    
    except Exception as e:
        print(f"Error loading data: {e}")
        raise e


def load_te_data(data_dir='data/TE'):
    """
    Load the processed Tennessee Eastman (TE) data
    """
    print(f"Loading processed TE data from {data_dir}")
    
    try:
        # Load training data (normal samples only)
        X_train = np.loadtxt(f"{data_dir}/train/d00.dat")
        
        # Load test data
        X_test = np.loadtxt(f"{data_dir}/test/d01_te.dat")
        
        # Load fault occurrence sample if available
        try:
            with open(f"{data_dir}/test/happen.txt", 'r') as f:
                happen = int(f.read().strip())
        except FileNotFoundError:
            # Default fault occurrence is at sample 160
            print("Fault occurrence file not found, using default value of 160")
            happen = 160
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        print(f"Fault occurrence at sample: {happen}")
        
        # Create test labels based on fault occurrence
        test_labels = np.zeros(X_test.shape[0])
        test_labels[happen:] = 1  # Set labels after fault occurrence to 1
        
        # Standardize data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, test_labels, happen
    
    except Exception as e:
        print(f"Error loading data: {e}")
        raise e


def train_model(model, train_data, batch_size=32, epochs=20, device="cpu"):
    """Train a model using the provided training data"""
    print(f"Training {model.__class__.__name__} on {device}...")

    # Create DataLoader
    train_tensor = torch.FloatTensor(train_data)
    train_dataset = TensorDataset(train_tensor, train_tensor)  # Input and target are the same for autoencoders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Move model to device
    model = model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Loss function
    criterion = nn.MSELoss()

    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_inputs, _ in train_loader:
            batch_inputs = batch_inputs.to(device)

            # Forward pass
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_inputs)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Print progress
        avg_loss = total_loss / len(train_loader)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")

    return model


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


def calculate_transformer_metrics(data, model, device="cpu"):
    """Calculate T2 and SPE statistics for transformer-based models"""
    model.eval()
    with torch.no_grad():
        # Convert to tensor
        data_tensor = torch.FloatTensor(data).to(device)

        # Get latent representation
        z = model._transform(data_tensor)

        # Reconstruct input
        x_recon = model._inverse_transform(z)

        # Move back to CPU and numpy
        z = z.cpu().numpy()
        x_recon = x_recon.cpu().numpy()
        data_tensor = data_tensor.cpu().numpy()

        # Calculate SPE (squared prediction error)
        spe = np.sum((data_tensor - x_recon) ** 2, axis=1)

        # Calculate T2 (Hotelling's T2 statistic)
        # For T2, we need covariance of training latent space, so this is approximate
        # Ideally, would pre-compute this from training data
        z_cov = np.cov(z.T) + np.eye(z.shape[1]) * 1e-8  # Add small value for numerical stability
        z_cov_inv = np.linalg.inv(z_cov)
        t2 = np.sum(z @ z_cov_inv * z, axis=1)

        return t2, spe


def calculate_control_limits(values, method='kde', confidence=0.99):
    """
    Calculate control limits for monitoring statistics

    Args:
        values: Array of statistic values
        method: Method for limit calculation ('kde' or 'percentile')
        confidence: Confidence level (default: 0.99)

    Returns:
        control_limit: Calculated control limit
    """
    if method == 'kde':
        # Use kernel density estimation
        kde = stats.gaussian_kde(values)
        x_grid = np.linspace(np.min(values), np.max(values) * 1.5, 1000)
        pdf = kde(x_grid)
        cdf = np.cumsum(pdf) / np.sum(pdf)
        limit_idx = np.argmin(np.abs(cdf - confidence))
        control_limit = x_grid[limit_idx]
    else:
        # Use percentile method
        control_limit = np.percentile(values, confidence * 100)

    return control_limit


def calculate_alarm_rates(statistics, control_limit, labels, fault_start_idx):
    """
    Calculate false alarm and miss rates

    Args:
        statistics: Monitoring statistic values (T2 or SPE)
        control_limit: Control limit for the statistic
        labels: Sample labels (0=normal, 1=fault)
        fault_start_idx: Index of first fault sample

    Returns:
        false_alarm_rate: False alarm rate (% of normal samples detected as faults)
        miss_rate: Miss rate (% of fault samples not detected)
    """
    # Generate alarms where statistic exceeds control limit
    alarms = statistics > control_limit

    # Normal samples are before fault_start_idx
    normal_alarms = alarms[:fault_start_idx]
    false_alarm_rate = (np.sum(normal_alarms) / len(normal_alarms)) * 100

    # Fault samples are from fault_start_idx onwards
    fault_alarms = alarms[fault_start_idx:]
    miss_rate = (len(fault_alarms) - np.sum(fault_alarms)) / len(fault_alarms) * 100

    return false_alarm_rate, miss_rate


def calculate_detection_time(statistics, control_limit, fault_start_idx, consecutive_violations=3):
    """
    Calculate fault detection time

    Args:
        statistics: Monitoring statistic values (T2 or SPE)
        control_limit: Control limit for the statistic
        fault_start_idx: Index of first fault sample
        consecutive_violations: Number of consecutive violations to trigger detection

    Returns:
        detection_time: Number of samples after fault start until detection
    """
    # Generate alarms where statistic exceeds control limit
    alarms = statistics > control_limit

    # Only look at alarms after fault start
    fault_alarms = alarms[fault_start_idx:]

    # Check for consecutive violations
    detection_time = None
    consecutive_count = 0

    for i, alarm in enumerate(fault_alarms):
        if alarm:
            consecutive_count += 1
            if consecutive_count >= consecutive_violations:
                detection_time = i - consecutive_violations + 1
                break
        else:
            consecutive_count = 0

    return detection_time


def plot_comparison(results, happen, dataset_name="SECOM"):
    """
    Plot comparison of detection statistics for all models

    Args:
        results: Dictionary of results from all methods
        happen: Index of first fault sample
        dataset_name: Name of the dataset for plot title
    """
    plt.figure(figsize=(15, 12))

    # Extract data for all methods
    methods = list(results.keys())
    num_methods = len(methods)

    # T2 statistics plot
    plt.subplot(2, 1, 1)
    for i, method in enumerate(methods):
        # Colors for normal and fault samples
        normal_color = plt.cm.tab10(i)
        fault_color = plt.cm.tab10((i+5) % 10)

        # Get T2 values and limit
        t2_values = results[method]["t2_test"]
        t2_limit = results[method]["t2_limit"]

        # Plot T2 values
        plt.plot(range(1, happen+1), t2_values[:happen],
                 color=normal_color, linestyle='-', label=f"{method} - Normal")
        plt.plot(range(happen+1, len(t2_values)+1), t2_values[happen:],
                 color=fault_color, linestyle='-', label=f"{method} - Fault")

        # Plot T2 limit
        plt.axhline(y=t2_limit, color=normal_color, linestyle='--')

    # Mark fault start
    plt.axvline(x=happen, color='black', linestyle='-', label="Fault Start")

    plt.title(f"{dataset_name} - T² Statistics Comparison")
    plt.xlabel("Sample")
    plt.ylabel("T²")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, alpha=0.3)

    # SPE statistics plot
    plt.subplot(2, 1, 2)
    for i, method in enumerate(methods):
        # Colors for normal and fault samples
        normal_color = plt.cm.tab10(i)
        fault_color = plt.cm.tab10((i+5) % 10)

        # Get SPE values and limit
        spe_values = results[method]["spe_test"]
        spe_limit = results[method]["spe_limit"]

        # Plot SPE values
        plt.plot(range(1, happen+1), spe_values[:happen],
                 color=normal_color, linestyle='-', label=f"{method} - Normal")
        plt.plot(range(happen+1, len(spe_values)+1), spe_values[happen:],
                 color=fault_color, linestyle='-', label=f"{method} - Fault")

        # Plot SPE limit
        plt.axhline(y=spe_limit, color=normal_color, linestyle='--')

    # Mark fault start
    plt.axvline(x=happen, color='black', linestyle='-', label="Fault Start")

    plt.title(f"{dataset_name} - SPE Statistics Comparison")
    plt.xlabel("Sample")
    plt.ylabel("SPE")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    os.makedirs("results/plots", exist_ok=True)
    plt.savefig(f"results/plots/{dataset_name.lower()}_transformer_comparison.png")
    print(f"Plot saved as {dataset_name.lower()}_transformer_comparison.png")
    plt.close()


def print_comparison_table(results, dataset_name="SECOM"):
    """
    Print a formatted comparison table for all methods

    Args:
        results: Dictionary of results from all methods
        dataset_name: Name of the dataset for table header
    """
    # Get all methods
    methods = list(results.keys())

    print("\n" + "="*80)
    print(f"{dataset_name} Fault Detection Comparison")
    print("="*80)

    # Performance metrics table
    print("{:<25} | {:<12} {:<12} | {:<12} {:<12} | {:<12}".format(
        "Method", "T² False(%)", "SPE False(%)", "T² Miss(%)", "SPE Miss(%)", "Detection (samples)"))
    print("-"*80)

    for method in methods:
        t2_false_rate = results[method]["t2_false_alarm_rate"]
        spe_false_rate = results[method]["spe_false_alarm_rate"]
        t2_miss_rate = results[method]["t2_miss_rate"]
        spe_miss_rate = results[method]["spe_miss_rate"]

        # Get detection time (use SPE by default, T2 as fallback)
        if results[method]["spe_detection_time"] is not None:
            detection_time = results[method]["spe_detection_time"]
        else:
            detection_time = results[method]["t2_detection_time"]

        # Format detection time string
        if detection_time is not None:
            detection_str = f"{detection_time}"
        else:
            detection_str = "Not detected"

        print("{:<25} | {:<12.2f} {:<12.2f} | {:<12.2f} {:<12.2f} | {:<12}".format(
            method, t2_false_rate, spe_false_rate, t2_miss_rate, spe_miss_rate, detection_str))

    print("-"*80)


def run_comparison_on_dataset(dataset_name, skip_basic=False):
    """
    Run transformer comparison on a specific dataset
    
    Args:
        dataset_name: Name of the dataset ('secom' or 'te')
        skip_basic: Whether to skip the basic transformer model
    """
    # Load data based on dataset name
    if dataset_name.lower() == 'secom':
        train_data, test_data, test_labels, fault_start_idx = load_secom_data()
        print(f"SECOM dataset: {train_data.shape[1]} variables, {train_data.shape[0]} training samples, {test_data.shape[0]} test samples")
        title_prefix = "SECOM"
    else:  # TE dataset
        train_data, test_data, test_labels, fault_start_idx = load_te_data()
        print(f"TE dataset: {train_data.shape[1]} variables, {train_data.shape[0]} training samples, {test_data.shape[0]} test samples")
        title_prefix = "TE"
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Results dictionary
    results = {}
    
    # Define latent dimension
    input_dim = train_data.shape[1]
    latent_dim = min(input_dim // 4, 20)  # Reduce to 1/4 or max 20 dimensions
    
    # Calculate nhead to ensure divisibility with input_dim
    # Find the largest divisor of input_dim that is <= 8
    for nhead in [8, 6, 5, 4, 3, 2, 1]:
        if input_dim % nhead == 0:
            break
    print(f"Using {nhead} attention heads for input dimension {input_dim}")
    
    # ========== PCA (Baseline) ==========
    print("\nRunning PCA baseline...")
    start_time = time.time()
    
    # Initialize and fit PCA
    pca = PCA(n_components=latent_dim)
    projected_train = pca.fit_transform(train_data)
    
    # Calculate T2 and SPE for training data
    t2_train, spe_train = calculate_pca_metrics(train_data, pca, projected_train, latent_dim)
    
    # Calculate control limits
    t2_limit = calculate_control_limits(t2_train, method='kde')
    spe_limit = calculate_control_limits(spe_train, method='kde')
    
    # Calculate T2 and SPE for test data
    t2_test, spe_test = calculate_pca_metrics(test_data, pca, projected_train, latent_dim)
    
    # Calculate alarm rates
    t2_false_alarm_rate, t2_miss_rate = calculate_alarm_rates(t2_test, t2_limit, test_labels, fault_start_idx)
    spe_false_alarm_rate, spe_miss_rate = calculate_alarm_rates(spe_test, spe_limit, test_labels, fault_start_idx)
    
    # Calculate detection times
    t2_detection_time = calculate_detection_time(t2_test, t2_limit, fault_start_idx)
    spe_detection_time = calculate_detection_time(spe_test, spe_limit, fault_start_idx)
    
    # Store results
    results["PCA"] = {
        "t2_train": t2_train,
        "spe_train": spe_train,
        "t2_test": t2_test,
        "spe_test": spe_test,
        "t2_limit": t2_limit,
        "spe_limit": spe_limit,
        "t2_false_alarm_rate": t2_false_alarm_rate,
        "spe_false_alarm_rate": spe_false_alarm_rate,
        "t2_miss_rate": t2_miss_rate,
        "spe_miss_rate": spe_miss_rate,
        "t2_detection_time": t2_detection_time,
        "spe_detection_time": spe_detection_time
    }
    
    print(f"PCA completed in {time.time() - start_time:.2f} seconds")
    
    # ========== Basic Transformer Autoencoder ==========
    if not skip_basic:
        print("\nRunning Basic Transformer Autoencoder...")
        start_time = time.time()
        
        # Initialize model
        basic_transformer = TransformerAutoencoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            num_encoder_layers=3,
            num_decoder_layers=3,
            nhead=nhead,  # Using the calculated nhead value
            dim_feedforward=256,
            dropout=0.1
        )
        
        # Check if model file exists, load or train
        model_file = f"results/models/{dataset_name.lower()}_basic_transformer.pth"
        if os.path.exists(model_file):
            print(f"Loading existing model from {model_file}")
            basic_transformer.load(model_file)
        else:
            print(f"Training new model...")
            basic_transformer = train_model(basic_transformer, train_data, batch_size=32, epochs=20, device=device)
            # Save model
            os.makedirs("results/models", exist_ok=True)
            basic_transformer.save(model_file)
            print(f"Model saved to {model_file}")
        
        # Calculate metrics on training data
        t2_train, spe_train = calculate_transformer_metrics(train_data, basic_transformer, device)
        
        # Calculate control limits
        t2_limit = calculate_control_limits(t2_train, method='kde')
        spe_limit = calculate_control_limits(spe_train, method='kde')
        
        # Calculate metrics on test data
        t2_test, spe_test = calculate_transformer_metrics(test_data, basic_transformer, device)
        
        # Calculate alarm rates
        t2_false_alarm_rate, t2_miss_rate = calculate_alarm_rates(t2_test, t2_limit, test_labels, fault_start_idx)
        spe_false_alarm_rate, spe_miss_rate = calculate_alarm_rates(spe_test, spe_limit, test_labels, fault_start_idx)
        
        # Calculate detection times
        t2_detection_time = calculate_detection_time(t2_test, t2_limit, fault_start_idx)
        spe_detection_time = calculate_detection_time(spe_test, spe_limit, fault_start_idx)
        
        # Store results
        results["Basic_Transformer"] = {
            "t2_train": t2_train,
            "spe_train": spe_train,
            "t2_test": t2_test,
            "spe_test": spe_test,
            "t2_limit": t2_limit,
            "spe_limit": spe_limit,
            "t2_false_alarm_rate": t2_false_alarm_rate,
            "spe_false_alarm_rate": spe_false_alarm_rate,
            "t2_miss_rate": t2_miss_rate,
            "spe_miss_rate": spe_miss_rate,
            "t2_detection_time": t2_detection_time,
            "spe_detection_time": spe_detection_time
        }
        
        print(f"Basic Transformer completed in {time.time() - start_time:.2f} seconds")
    
    # ========== Enhanced Transformer Autoencoder ==========
    print("\nRunning Enhanced Transformer Autoencoder...")
    start_time = time.time()
    
    # Initialize model
    enhanced_transformer = EnhancedTransformerAutoencoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        num_encoder_layers=4,
        num_decoder_layers=4,
        nhead=nhead,  # Using the calculated nhead value
        dim_feedforward=512,
        dropout=0.1,
        activation='relu',
        use_residual=True
    )
    
    # Check if model file exists, load or train
    model_file = f"results/models/{dataset_name.lower()}_enhanced_transformer.pth"
    try:
        if os.path.exists(model_file):
            print(f"Loading existing model from {model_file}")
            enhanced_transformer = EnhancedTransformerAutoencoder.load(model_file, map_location=device)
            print("Model loaded successfully")
        else:
            print(f"Model file not found at {model_file}, training new model...")
            enhanced_transformer = train_model(enhanced_transformer, train_data, batch_size=32, epochs=20, device=device)
            # Save model
            os.makedirs("results/models", exist_ok=True)
            enhanced_transformer.save(model_file)
            print(f"Model saved to {model_file}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Training new model instead...")
        enhanced_transformer = train_model(enhanced_transformer, train_data, batch_size=32, epochs=20, device=device)
        # Save model
        os.makedirs("results/models", exist_ok=True)
        enhanced_transformer.save(model_file)
        print(f"Model saved to {model_file}")

    # Calculate metrics on training data
    t2_train, spe_train = calculate_transformer_metrics(train_data, enhanced_transformer, device)

    # Calculate control limits
    t2_limit = calculate_control_limits(t2_train, method='kde')
    spe_limit = calculate_control_limits(spe_train, method='kde')

    # Calculate metrics on test data
    t2_test, spe_test = calculate_transformer_metrics(test_data, enhanced_transformer, device)

    # Calculate alarm rates
    t2_false_alarm_rate, t2_miss_rate = calculate_alarm_rates(t2_test, t2_limit, test_labels, fault_start_idx)
    spe_false_alarm_rate, spe_miss_rate = calculate_alarm_rates(spe_test, spe_limit, test_labels, fault_start_idx)

    # Calculate detection times
    t2_detection_time = calculate_detection_time(t2_test, t2_limit, fault_start_idx)
    spe_detection_time = calculate_detection_time(spe_test, spe_limit, fault_start_idx)

    # Store results
    results["Enhanced_Transformer"] = {
        "t2_train": t2_train,
        "spe_train": spe_train,
        "t2_test": t2_test,
        "spe_test": spe_test,
        "t2_limit": t2_limit,
        "spe_limit": spe_limit,
        "t2_false_alarm_rate": t2_false_alarm_rate,
        "spe_false_alarm_rate": spe_false_alarm_rate,
        "t2_miss_rate": t2_miss_rate,
        "spe_miss_rate": spe_miss_rate,
        "t2_detection_time": t2_detection_time,
        "spe_detection_time": spe_detection_time
    }

    print(f"Enhanced Transformer completed in {time.time() - start_time:.2f} seconds")

    # Plot results
    plot_comparison(results, fault_start_idx, dataset_name)

    # Print comparison table
    print_comparison_table(results, dataset_name)

    return results


def main(skip_basic=False, dataset='both'):
    """
    Main function to run comparison

    Args:
        skip_basic: Whether to skip the basic transformer model
        dataset: Dataset to use ('secom', 'te', or 'both')
    """
    # Ensure results directories exist
    os.makedirs("results/models", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("results/logs", exist_ok=True)

    # Run comparison based on selected dataset
    if dataset.lower() in ['secom', 'both']:
        print("\n" + "="*50)
        print("RUNNING COMPARISON ON SECOM DATASET")
        print("="*50)
        run_comparison_on_dataset('secom', skip_basic)

    if dataset.lower() in ['te', 'both']:
        print("\n" + "="*50)
        print("RUNNING COMPARISON ON TE DATASET")
        print("="*50)
        run_comparison_on_dataset('te', skip_basic)


if __name__ == "__main__":
    # Parse command line arguments when run directly
    parser = argparse.ArgumentParser(description='Run transformer-based fault detection comparison')
    parser.add_argument('--dataset', type=str, choices=['secom', 'te', 'both'], default='both',
                        help='Dataset to use: secom, te, or both')
    parser.add_argument('--skip_basic', action='store_true',
                        help='Skip Basic Transformer model (faster)')

    args = parser.parse_args()
    main(args.skip_basic, args.dataset)