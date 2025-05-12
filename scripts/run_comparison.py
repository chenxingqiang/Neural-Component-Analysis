#!/usr/bin/env python3
"""
Wrapper script to run comparison detection for SECOM and TE datasets.
This helps avoid import errors when running scripts directly.

Usage:
  python run_comparison.py --dataset secom [--skip_improved_transformer] [--include_transformer]
  python run_comparison.py --dataset te [--skip_improved_transformer]
  python run_comparison.py --dataset both [--skip_improved_transformer]
  python run_comparison.py --dataset transformer [--skip_basic]
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
    parser.add_argument('--dataset', type=str, choices=['secom', 'te', 'both', 'transformer'], default='secom',
                        help='Dataset to use: secom, te, both, or transformer')
    parser.add_argument('--skip_improved_transformer', action='store_true', 
                        help='Skip Improved Transformer model (faster)')
    parser.add_argument('--include_transformer', action='store_true', 
                        help='Include Transformer-Enhanced Two-Stage detector (SECOM only)')
    parser.add_argument('--skip_basic', action='store_true',
                        help='Skip Basic Transformer model (transformer comparison only)')
    
    args = parser.parse_args()
    
    # Ensure results directories exist
    os.makedirs("results/models", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("results/logs", exist_ok=True)
    
    # Set dataset prefix
    dataset_prefix = args.dataset.lower()
    
    if args.dataset.lower() == 'transformer':
        # Import and run the Transformer comparison script
        from scripts.transformer_comparison_detection import main as transformer_main
        
        # Run Transformer main function
        transformer_main()
        
    elif args.dataset.lower() in ['secom', 'both']:
        # Import and run the SECOM comparison script
        from scripts.secom_comparison_detection import main as secom_main
        
        # Set model paths for SECOM dataset
        enhanced_model_path = f"results/models/secom_enhanced_transformer.pth"
        improved_model_path = f"results/models/secom_improved_transformer.pth"
        
        # Pass model paths to SECOM main function
        secom_main(
            skip_improved_transformer=args.skip_improved_transformer,
            include_transformer=args.include_transformer,
            model_paths={
                'enhanced': enhanced_model_path,
                'improved': improved_model_path
            }
        )
        
        # If 'both' is selected, continue to TE dataset
        if args.dataset.lower() == 'both':
            # Import and run the TE comparison script
            from scripts.te_comparison_detection import main as te_main
            
            # Run TE main function (the TE script manages its own model paths)
            te_main(
                skip_improved_transformer=args.skip_improved_transformer
            )
            
    elif args.dataset.lower() == 'te':
        # Import and run the TE comparison script
        from scripts.te_comparison_detection import main as te_main
        
        # Run TE main function (the TE script manages its own model paths)
        te_main(
            skip_improved_transformer=args.skip_improved_transformer
        )

if __name__ == "__main__":
    main() 