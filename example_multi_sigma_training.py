#!/usr/bin/env python3
"""
Example script demonstrating multi-sigma blur simulation for CNNT training

This script shows how to:
1. Enable blur simulation with different difficulty levels
2. Use curriculum learning for progressive difficulty
3. Configure different sigma ranges and probabilities
4. Train the model with improved generalization

Usage:
    python example_multi_sigma_training.py --enable_blur_simulation --h5files your_data.h5
"""

import argparse
import sys
from main import main, arg_parser

def get_example_args():
    """
    Get example arguments for multi-sigma training
    """
    args = [
        # Data and model configuration
        "--h5files", "your_microscopy_data.h5",
        "--project", "CNNT_MultiSigma",
        "--run_name", "multi_sigma_curriculum_learning",
        "--num_epochs", "50",
        "--batch_size", "4",
        
        # Enable blur simulation
        "--enable_blur_simulation",
        
        # Curriculum learning configuration (automatic difficulty progression)
        # Don't set --blur_difficulty to enable curriculum learning
        
        # Multi-sigma configuration
        "--blur_sigma_ranges", "0.5,1.5", "1.5,3.0", "3.0,5.0", "5.0,8.0",
        "--blur_probabilities", "0.4", "0.3", "0.2", "0.1",
        "--variable_blur_prob", "0.3",  # 30% chance of spatially variable blur
        "--no_blur_prob", "0.1",  # 10% chance of clean images
        
        # Model architecture (optimized for multi-scale features)
        "--blocks", "32", "64", "96", "128",
        "--blocks_per_set", "4",
        "--n_head", "8",
        "--dropout_p", "0.1",
        
        # Training configuration
        "--height", "128", "160",
        "--width", "128", "160",
        "--time", "16",
        "--global_lr", "3e-4",
        "--scheduler", "OneCycleLR",
        
        # Loss function (balanced for detail preservation)
        "--loss", "mse", "ssim", "sobel",
        "--loss_weights", "0.3", "1.0", "0.2",
        
        # Optimization
        "--optim", "adamw",
        "--weight_decay", "0.01",
        "--clip_grad_norm", "1.0",
    ]
    
    return args

def demonstrate_fixed_difficulty():
    """
    Example of training with fixed difficulty levels
    """
    print("Training with EASY difficulty (good for initial testing):")
    easy_args = get_example_args() + ["--blur_difficulty", "easy"]
    
    print("Training with MEDIUM difficulty:")
    medium_args = get_example_args() + ["--blur_difficulty", "medium"] 
    
    print("Training with HARD difficulty:")
    hard_args = get_example_args() + ["--blur_difficulty", "hard"]
    
    print("Training with EXTREME difficulty:")
    extreme_args = get_example_args() + ["--blur_difficulty", "extreme"]
    
    return easy_args, medium_args, hard_args, extreme_args

def demonstrate_custom_sigma_ranges():
    """
    Example of custom sigma ranges for specific use cases
    """
    # For light microscopy (typically lower blur)
    light_microscopy_args = get_example_args() + [
        "--blur_sigma_ranges", "0.2,0.8", "0.8,1.5", "1.5,2.5", "2.5,4.0",
        "--blur_probabilities", "0.5", "0.3", "0.15", "0.05"
    ]
    
    # For electron microscopy (can handle higher blur)
    electron_microscopy_args = get_example_args() + [
        "--blur_sigma_ranges", "1.0,2.0", "2.0,4.0", "4.0,6.0", "6.0,10.0",
        "--blur_probabilities", "0.3", "0.4", "0.2", "0.1"
    ]
    
    return light_microscopy_args, electron_microscopy_args

if __name__ == "__main__":
    print("=== CNNT Multi-Sigma Blur Training Examples ===\n")
    
    # Parse command line arguments
    parser = arg_parser()
    args = parser.parse_args()
    
    if len(sys.argv) == 1:
        # Show examples if no arguments provided
        print("No arguments provided. Here are some examples:\n")
        
        print("1. CURRICULUM LEARNING (Recommended):")
        curriculum_args = get_example_args()
        print("python", sys.argv[0], " ".join(curriculum_args))
        print()
        
        print("2. FIXED DIFFICULTY LEVELS:")
        easy, medium, hard, extreme = demonstrate_fixed_difficulty()
        print("Easy:    python", sys.argv[0], " ".join(easy[-2:]))  # Show just the difficulty part
        print("Medium:  python", sys.argv[0], " ".join(medium[-2:]))
        print("Hard:    python", sys.argv[0], " ".join(hard[-2:]))
        print("Extreme: python", sys.argv[0], " ".join(extreme[-2:]))
        print()
        
        print("3. CUSTOM SIGMA RANGES:")
        light, electron = demonstrate_custom_sigma_ranges()
        print("Light microscopy: --blur_sigma_ranges 0.2,0.8 0.8,1.5 1.5,2.5 2.5,4.0")
        print("Electron microscopy: --blur_sigma_ranges 1.0,2.0 2.0,4.0 4.0,6.0 6.0,10.0")
        print()
        
        print("4. DISABLE BLUR SIMULATION:")
        print("python", sys.argv[0], "your_args_here  # (simply omit --enable_blur_simulation)")
        print()
        
        print("Key Benefits of Multi-Sigma Training:")
        print("- Better generalization to various blur levels")
        print("- Curriculum learning prevents overfitting to specific blur levels")
        print("- Spatially variable blur simulates real microscopy conditions")
        print("- Progressive difficulty helps model learn robust features")
        print("\nTo start training, add your data path and run with --enable_blur_simulation")
        
    else:
        # Run normal training with provided arguments
        print("Starting CNNT training with multi-sigma blur simulation...")
        print(f"Blur simulation enabled: {args.enable_blur_simulation}")
        if args.blur_difficulty:
            print(f"Fixed difficulty level: {args.blur_difficulty}")
        else:
            print("Using curriculum learning (automatic difficulty progression)")
        
        # Call the main training function
        main()