import os
import matplotlib.pyplot as plt

def save_plot(filename, prefix="results/plots"):
    """
    Save plot to specified directory with proper path handling
    
    Parameters:
    -----------
    filename : str
        Name of the file to save
    prefix : str
        Directory prefix to save to
    """
    # Create plots directory if it doesn't exist
    os.makedirs(prefix, exist_ok=True)
    
    # Ensure filename doesn't have the prefix already
    if filename.startswith(prefix):
        plt.savefig(filename)
    else:
        # Construct full path
        full_path = os.path.join(prefix, os.path.basename(filename))
        plt.savefig(full_path)
    
    print(f"Plot saved as {os.path.join(prefix, os.path.basename(filename))}")
