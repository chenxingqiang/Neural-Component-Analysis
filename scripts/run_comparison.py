#!/usr/bin/env python3
"""
Wrapper script to run comparison detection for SECOM and TE datasets.
This helps avoid import errors when running scripts directly.

Usage:
  python run_comparison.py --dataset secom [--skip_improved_transformer] [--include_transformer]
  python run_comparison.py --dataset te [--skip_improved_transformer]
"""

import os
import sys
import argparse

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run fault detection methods comparison')
    parser.add_argument('--dataset', type=str, choices=['secom', 'te'], default='secom',
                        help='Dataset to use: secom or te')
    parser.add_argument('--skip_improved_transformer', action='store_true', 
                        help='Skip Improved Transformer model (faster)')
    parser.add_argument('--include_transformer', action='store_true', 
                        help='Include Transformer-Enhanced Two-Stage detector (SECOM only)')
    
    args = parser.parse_args()
    
    # Ensure results directories exist
    os.makedirs("results/models", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("results/logs", exist_ok=True)
    
    # Set model and plot paths based on dataset
    dataset_prefix = args.dataset.lower()
    
    if args.dataset.lower() == 'secom':
        # Import and run the SECOM comparison script
        from scripts.secom_comparison_detection import main as secom_main
        
        # Set model paths for SECOM dataset
        enhanced_model_path = f"results/models/{dataset_prefix}_enhanced_transformer.pth"
        improved_model_path = f"results/models/{dataset_prefix}_improved_transformer.pth"
        
        # Pass model paths to SECOM main function
        secom_main(
            skip_improved_transformer=args.skip_improved_transformer,
            include_transformer=args.include_transformer,
            model_paths={
                'enhanced': enhanced_model_path,
                'improved': improved_model_path
            }
        )
    else:  # te dataset
        # Import and run the TE comparison script
        from scripts.te_comparison_detection import main as te_main
        
        # Set model paths for TE dataset
        enhanced_model_path = f"results/models/{dataset_prefix}_enhanced_transformer.pth"
        improved_model_path = f"results/models/{dataset_prefix}_improved_transformer.pth"
        
        # Pass model paths to TE main function
        te_main(
            skip_improved_transformer=args.skip_improved_transformer,
            model_paths={
                'enhanced': enhanced_model_path,
                'improved': improved_model_path
            }
        )

if __name__ == "__main__":
    main() 