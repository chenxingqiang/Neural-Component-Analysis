import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import glob

def load_te_data_category(category=1, data_dir='data/TE'):
    """
    Load the Tennessee Eastman (TE) data for a specific fault category
    
    Parameters:
    -----------
    category : int
        Fault category number (0-21)
        0 represents normal operation
        1-21 represent different fault conditions
    data_dir : str
        Path to the TE data directory
    
    Returns:
    --------
    X_train_scaled : ndarray
        Standardized training data
    X_test_scaled : ndarray
        Standardized test data
    test_labels : ndarray
        Labels for test data (0=normal, 1=fault)
    happen : int
        Index where fault occurs in test data
    """
    print(f"Loading TE data for fault category {category}")
    
    try:
        # Load training data (normal samples only)
        X_train = np.loadtxt(f"{data_dir}/train/d00.dat")
        
        # Load test data for specific category
        test_file = f"{data_dir}/test/d{category:02d}_te.dat"
        X_test = np.loadtxt(test_file)
        
        # Fix data shape if needed - ensure samples are rows and features are columns
        if X_train.shape[0] < X_train.shape[1]:  # If fewer rows than columns, transpose
            print(f"Transposing training data from shape {X_train.shape} to {X_train.T.shape}")
            X_train = X_train.T
        
        if X_test.shape[0] < X_test.shape[1]:  # If fewer rows than columns, transpose
            print(f"Transposing test data from shape {X_test.shape} to {X_test.T.shape}")
            X_test = X_test.T
        
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
        if category > 0:  # Only set fault labels for actual fault categories
            test_labels[happen:] = 1  # Set labels after fault occurrence to 1
        
        # Standardize data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, test_labels, happen
    
    except Exception as e:
        print(f"Error loading data for category {category}: {e}")
        raise e

def get_available_categories(data_dir='data/TE'):
    """
    Get list of available fault categories in the TE dataset
    
    Parameters:
    -----------
    data_dir : str
        Path to the TE data directory
    
    Returns:
    --------
    categories : list
        List of available fault category numbers
    """
    test_files = glob.glob(f"{data_dir}/test/d*_te.dat")
    categories = []
    
    for file in test_files:
        # Extract category number from filename (d##_te.dat)
        filename = os.path.basename(file)
        if filename.startswith('d') and filename.endswith('_te.dat'):
            category_str = filename[1:3]
            try:
                category = int(category_str)
                categories.append(category)
            except ValueError:
                continue
    
    return sorted(categories)

def load_te_data_all(data_dir='data/TE', categories=None):
    """
    Load the Tennessee Eastman (TE) data for all or selected fault categories
    
    Parameters:
    -----------
    data_dir : str
        Path to the TE data directory
    categories : list or None
        List of fault categories to load. If None, all available categories are loaded.
    
    Returns:
    --------
    dataset : dict
        Dictionary with category numbers as keys and tuples of 
        (X_train_scaled, X_test_scaled, test_labels, happen) as values
    """
    if categories is None:
        categories = get_available_categories(data_dir)
    
    dataset = {}
    for category in categories:
        try:
            result = load_te_data_category(category, data_dir)
            dataset[category] = result
        except Exception as e:
            print(f"Skipping category {category} due to error: {e}")
    
    return dataset 