import os
import torch
import time
from datetime import datetime

def save_model(model, model_name, dataset_name, category=None, version=None, additional_info=None):
    """
    Save a PyTorch model with proper naming and versioning
    
    Parameters:
    -----------
    model : torch.nn.Module
        The PyTorch model to save
    model_name : str
        Name of the model (e.g., 'enhanced_transformer', 'improved_transformer')
    dataset_name : str
        Name of the dataset used for training (e.g., 'TE', 'SECOM')
    category : int or None
        Category or fault type number (if applicable)
    version : str or None
        Version string to use in filename. If None, timestamp is used.
    additional_info : dict or None
        Additional information to save with the model
    
    Returns:
    --------
    model_path : str
        Path to the saved model file
    """
    # Create base directory
    base_dir = "results/models"
    os.makedirs(base_dir, exist_ok=True)
    
    # Create dataset-specific subdirectory
    dataset_dir = os.path.join(base_dir, dataset_name.lower())
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Create model-specific subdirectory
    model_dir = os.path.join(dataset_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # Generate version if not provided
    if version is None:
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Construct filename
    if category is not None:
        filename = f"{model_name}_{dataset_name}_cat{category}_v{version}.pth"
    else:
        filename = f"{model_name}_{dataset_name}_v{version}.pth"
    
    model_path = os.path.join(model_dir, filename)
    
    # Prepare save data - include model state dict and any additional info
    save_data = {
        'model_state_dict': model.state_dict(),
        'timestamp': time.time(),
        'model_name': model_name,
        'dataset_name': dataset_name
    }
    
    # Add category if provided
    if category is not None:
        save_data['category'] = category
        
    # Add any additional info
    if additional_info is not None:
        save_data.update(additional_info)
    
    # Save the model
    torch.save(save_data, model_path)
    print(f"Model saved to {model_path}")
    
    return model_path


def load_model(model, model_path):
    """
    Load a saved PyTorch model
    
    Parameters:
    -----------
    model : torch.nn.Module
        The model instance to load parameters into
    model_path : str
        Path to the saved model file
    
    Returns:
    --------
    model : torch.nn.Module
        The model with loaded parameters
    additional_info : dict
        Additional information saved with the model
    """
    # Check if file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load the saved data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    saved_data = torch.load(model_path, map_location=device)
    
    # Check if it's a simple state dict or a dictionary with additional info
    if 'model_state_dict' in saved_data:
        model.load_state_dict(saved_data['model_state_dict'])
        # Extract additional info
        additional_info = {k: v for k, v in saved_data.items() if k != 'model_state_dict'}
    else:
        # Legacy format - just the state dict
        model.load_state_dict(saved_data)
        additional_info = {}
    
    print(f"Model loaded from {model_path}")
    
    return model, additional_info


def get_latest_model_path(model_name, dataset_name, category=None):
    """
    Find the latest model file for a given model type and dataset
    
    Parameters:
    -----------
    model_name : str
        Name of the model
    dataset_name : str
        Name of the dataset
    category : int or None
        Category or fault type number (if applicable)
    
    Returns:
    --------
    latest_model_path : str or None
        Path to the latest model file, or None if no model found
    """
    # Construct directory path
    base_dir = os.path.join("results/models", dataset_name.lower(), model_name)
    
    if not os.path.exists(base_dir):
        return None
    
    # Pattern to match
    if category is not None:
        pattern = f"{model_name}_{dataset_name}_cat{category}_v"
    else:
        pattern = f"{model_name}_{dataset_name}_v"
    
    # Find all matching files
    matching_files = []
    for filename in os.listdir(base_dir):
        if filename.startswith(pattern) and filename.endswith('.pth'):
            file_path = os.path.join(base_dir, filename)
            file_time = os.path.getmtime(file_path)
            matching_files.append((file_path, file_time))
    
    # Return the most recent file
    if matching_files:
        latest_file = max(matching_files, key=lambda x: x[1])[0]
        return latest_file
    
    return None 