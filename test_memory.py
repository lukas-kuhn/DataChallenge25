#!/usr/bin/env python3
"""
Simple script to test memory usage of the optimized VAE model
"""

import torch
import torch.nn as nn
from vae_model import VAE

def test_memory_usage():
    """Test the memory usage of the VAE model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")
    
    if not torch.cuda.is_available():
        print("CUDA not available, cannot test GPU memory usage")
        return
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    
    # Create the memory-optimized model
    model = VAE(
        in_channels=3,
        latent_dim=256,
        model_channels=96,
        channel_mult=(1, 1, 2, 3, 4),
        num_res_blocks=2,
        attention_resolutions=(16,),
        dropout=0.0
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test different batch sizes
    batch_sizes = [1, 2, 4, 8, 16]
    
    for batch_size in batch_sizes:
        try:
            # Clear cache
            torch.cuda.empty_cache()
            
            # Create dummy batch
            x = torch.randn(batch_size, 3, 224, 224, device=device)
            
            # Forward pass
            with torch.no_grad():
                recon, mu, logvar = model(x)
            
            # Get memory usage
            memory_used = torch.cuda.max_memory_allocated() / 1024**3  # GB
            print(f"Batch size {batch_size:2d}: {memory_used:.2f} GB")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Batch size {batch_size:2d}: OUT OF MEMORY")
                break
            else:
                raise e
    
    # Test training memory usage (with gradients)
    print("\nTesting training memory usage (with gradients):")
    
    for batch_size in [1, 2, 4, 8]:
        try:
            # Clear cache
            torch.cuda.empty_cache()
            
            # Create model in training mode
            model.train()
            
            # Create dummy batch
            x = torch.randn(batch_size, 3, 224, 224, device=device, requires_grad=True)
            
            # Forward pass
            recon, mu, logvar = model(x)
            
            # Compute loss (simplified)
            loss = nn.MSELoss()(recon, x)
            
            # Backward pass
            loss.backward()
            
            # Get memory usage
            memory_used = torch.cuda.max_memory_allocated() / 1024**3  # GB
            print(f"Training batch size {batch_size:2d}: {memory_used:.2f} GB")
            
            # Clear gradients
            model.zero_grad()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Training batch size {batch_size:2d}: OUT OF MEMORY")
                break
            else:
                raise e

def main():
    print("=== VAE Memory Usage Test ===")
    test_memory_usage()
    print("\nTest completed!")

if __name__ == '__main__':
    main() 