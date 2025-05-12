import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import re

def load_secom_data(data_path='secom_data/secom.data', labels_path='secom_data/SECOM_labels.TXT'):
    """
    Load the SECOM dataset and labels
    """
    print(f"Loading SECOM dataset from {data_path} and {labels_path}")
    
    # Load the data
    try:
        # The SECOM data contains some missing values (NA) which we need to handle
        data = pd.read_csv(data_path, header=None, sep=' ', na_values='NA')
        data = data.fillna(0)  # Replace NA with 0
        
        # Read labels (-1 = pass/normal, 1 = fail/faulty)
        labels_df = pd.read_csv(labels_path, header=None, sep=' ', usecols=[0])
        labels = labels_df[0].values
        
        # Convert labels: -1 (pass) -> 0 (normal), 1 (fail) -> 1 (faulty)
        labels = (labels == 1).astype(int)
        
        print(f"Data shape: {data.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Number of faulty samples: {np.sum(labels == 1)}")
        
        return data.values, labels
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

def split_secom_data(data, labels, test_size=0.3, random_state=42):
    """
    Split the SECOM dataset into training and testing sets
    """
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=test_size, random_state=random_state,
        stratify=labels  # Ensure similar distribution of normal/faulty in both sets
    )
    
    # Get indices of normal and faulty samples in test set
    normal_indices = np.where(y_test == 0)[0]
    fault_indices = np.where(y_test == 1)[0]
    
    print(f"Training set: {X_train.shape}, Labels: {y_train.shape}")
    print(f"Testing set: {X_test.shape}, Labels: {y_test.shape}")
    print(f"Testing set - Normal samples: {len(normal_indices)}, Faulty samples: {len(fault_indices)}")
    
    return X_train, X_test, y_train, y_test, normal_indices, fault_indices

def save_processed_data(X_train, X_test, y_train, y_test, normal_indices, fault_indices, output_dir='data/secom'):
    """
    Save the processed SECOM data in the format expected by the fault detection models
    """
    # Create output directories if they don't exist
    os.makedirs(f"{output_dir}/train", exist_ok=True)
    os.makedirs(f"{output_dir}/test", exist_ok=True)
    
    # Save training data (normal samples only for training)
    normal_train = X_train[y_train == 0]
    np.savetxt(f"{output_dir}/train/d00.dat", normal_train)
    print(f"Saved normal training data: {normal_train.shape} to {output_dir}/train/d00.dat")
    
    # Save all test data together
    np.savetxt(f"{output_dir}/test/d01_te.dat", X_test)
    print(f"Saved test data: {X_test.shape} to {output_dir}/test/d01_te.dat")
    
    # Save test labels and indices
    np.savetxt(f"{output_dir}/test/labels.txt", y_test, fmt='%d')
    np.savetxt(f"{output_dir}/test/normal_indices.txt", normal_indices, fmt='%d')
    np.savetxt(f"{output_dir}/test/fault_indices.txt", fault_indices, fmt='%d')
    
    # Save information about fault happening at the first fault index
    happen = min(fault_indices) if len(fault_indices) > 0 else 0
    with open(f"{output_dir}/test/happen.txt", 'w') as f:
        f.write(str(happen))
    print(f"Fault occurrence at sample: {happen}")
    
    return happen

def visualize_data(X_train, X_test, y_test, happen, output_dir='data/secom'):
    """
    Visualize the processed data
    """
    # Standardize data for visualization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Visualize a few features to see the difference between normal and faulty
    plt.figure(figsize=(15, 10))
    
    # Plot first few features of normal training data
    for i in range(min(5, X_train_scaled.shape[1])):
        plt.subplot(2, 3, i+1)
        plt.hist(X_train_scaled[:, i], bins=20, alpha=0.5, label='Normal (Train)')
        
        # Plot test data separated by normal/faulty
        normal_test = X_test_scaled[y_test == 0, i]
        fault_test = X_test_scaled[y_test == 1, i]
        
        plt.hist(normal_test, bins=20, alpha=0.5, label='Normal (Test)')
        plt.hist(fault_test, bins=20, alpha=0.5, label='Faulty (Test)')
        plt.title(f'Feature {i+1}')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_distribution.png")
    plt.close()
    
    # Plot a time series of test data to show normal and faulty regions
    plt.figure(figsize=(12, 6))
    num_features = min(5, X_test_scaled.shape[1])
    
    for i in range(num_features):
        plt.plot(X_test_scaled[:, i], label=f'Feature {i+1}')
    
    plt.axvline(x=happen, color='r', linestyle='--', label='Fault Occurrence')
    plt.legend()
    plt.title('SECOM Test Data Time Series')
    plt.xlabel('Sample')
    plt.ylabel('Standardized Value')
    plt.savefig(f"{output_dir}/time_series.png")
    plt.close()
    
    print(f"Saved visualization to {output_dir}")

def main():
    # Load SECOM data
    data, labels = load_secom_data()
    if data is None:
        return
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test, normal_indices, fault_indices = split_secom_data(data, labels)
    
    # Save processed data
    output_dir = 'data/secom'
    happen = save_processed_data(X_train, X_test, y_train, y_test, normal_indices, fault_indices, output_dir)
    
    # Visualize data
    visualize_data(X_train, X_test, y_test, happen, output_dir)
    
    print("\nSECOM data processing complete.")
    print("You can now run the fault detection models with the processed SECOM data:")
    print(f"  1. Update the data path to '{output_dir}' in your model code")
    print(f"  2. Run the model with the processed SECOM data")

if __name__ == "__main__":
    main() 