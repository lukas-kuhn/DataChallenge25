import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from pathlib import Path

import analysis as a
from vae_model import VAE

def denormalize(tensor):
    """Denormalize tensor for visualization"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean

def load_model(checkpoint_path, device):
    """Load trained VAE model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    model = VAE(
        in_channels=3,
        latent_dim=config['latent_dim'],
        model_channels=config['model_channels'],
        channel_mult=(1, 1, 2, 3, 4),  # Match training configuration
        num_res_blocks=config['num_res_blocks'],
        attention_resolutions=(16,),  # Match training configuration
        dropout=0.0  # No dropout during inference
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, config

def preprocess_image(image_path, image_size=224):
    """Preprocess a single image for inference"""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def reconstruct_image(model, image_path, device, save_path=None):
    """Reconstruct a single image"""
    # Preprocess image
    image_tensor = preprocess_image(image_path).to(device)
    
    with torch.no_grad():
        # Encode and decode
        mu, logvar = model.encode(image_tensor)
        z = model.reparameterize(mu, logvar)
        reconstruction = model.decode(z)
        
        # Denormalize for visualization
        original = denormalize(image_tensor.cpu().squeeze(0))
        recon = denormalize(reconstruction.cpu().squeeze(0))
        
        # Create comparison plot
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        axes[0].imshow(original.permute(1, 2, 0).clamp(0, 1))
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        axes[1].imshow(recon.permute(1, 2, 0).clamp(0, 1))
        axes[1].set_title('Reconstructed')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Reconstruction saved to {save_path}")
        
        plt.show()
        
        return original, recon, mu, logvar

def generate_samples(model, device, num_samples=16, save_path=None):
    """Generate new samples from the latent space"""
    with torch.no_grad():
        samples = model.sample(num_samples, device)
        samples_denorm = denormalize(samples.cpu())
        
        # Create grid of generated images
        grid_size = int(np.sqrt(num_samples))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(2 * grid_size, 2 * grid_size))
        
        for i in range(grid_size):
            for j in range(grid_size):
                idx = i * grid_size + j
                if idx < num_samples:
                    axes[i, j].imshow(samples_denorm[idx].permute(1, 2, 0).clamp(0, 1))
                axes[i, j].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Generated samples saved to {save_path}")
        
        plt.show()
        
        return samples_denorm

def interpolate_between_images(model, image_path1, image_path2, device, num_steps=10, save_path=None):
    """Interpolate between two images in latent space"""
    # Preprocess images
    image1 = preprocess_image(image_path1).to(device)
    image2 = preprocess_image(image_path2).to(device)
    
    with torch.no_grad():
        # Encode both images
        mu1, _ = model.encode(image1)
        mu2, _ = model.encode(image2)
        
        # Create interpolation steps
        alphas = np.linspace(0, 1, num_steps)
        interpolations = []
        
        for alpha in alphas:
            # Linear interpolation in latent space
            z_interp = (1 - alpha) * mu1 + alpha * mu2
            recon = model.decode(z_interp)
            interpolations.append(denormalize(recon.cpu().squeeze(0)))
        
        # Create visualization
        fig, axes = plt.subplots(1, num_steps, figsize=(2 * num_steps, 2))
        
        for i, interp in enumerate(interpolations):
            axes[i].imshow(interp.permute(1, 2, 0).clamp(0, 1))
            axes[i].set_title(f'α={alphas[i]:.1f}')
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Interpolation saved to {save_path}")
        
        plt.show()
        
        return interpolations

def explore_latent_space(model, device, config, base_image_path=None, save_path=None):
    """Explore the latent space by modifying specific dimensions"""
    if base_image_path:
        # Start from a real image
        image_tensor = preprocess_image(base_image_path).to(device)
        with torch.no_grad():
            mu, logvar = model.encode(image_tensor)
            base_z = mu.clone()
    else:
        # Start from random noise
        base_z = torch.randn(1, config['latent_dim'], 7, 7, device=device)
    
    # Explore different latent dimensions
    fig, axes = plt.subplots(3, 7, figsize=(14, 6))
    
    with torch.no_grad():
        # Original/center
        center_recon = model.decode(base_z)
        center_denorm = denormalize(center_recon.cpu().squeeze(0))
        axes[1, 3].imshow(center_denorm.permute(1, 2, 0).clamp(0, 1))
        axes[1, 3].set_title('Original')
        axes[1, 3].axis('off')
        
        # Vary different dimensions
        variation_scale = 2.0
        variations = [-3, -2, -1, 0, 1, 2, 3]
        
        for i, var in enumerate(variations):
            if i == 3:  # Skip center (already done)
                continue
                
            # Modify random latent dimensions
            z_modified = base_z.clone()
            
            # Randomly select some dimensions to modify
            dims_to_modify = torch.randperm(config['latent_dim'])[:10]  # Modify 10 random dimensions
            for dim in dims_to_modify:
                z_modified[0, dim, :, :] += var * variation_scale * torch.randn_like(z_modified[0, dim, :, :]) * 0.1
            
            recon = model.decode(z_modified)
            recon_denorm = denormalize(recon.cpu().squeeze(0))
            
            axes[1, i].imshow(recon_denorm.permute(1, 2, 0).clamp(0, 1))
            axes[1, i].set_title(f'Var: {var}')
            axes[1, i].axis('off')
        
        # Fill remaining subplots with more random variations
        for row in [0, 2]:
            for col in range(7):
                z_random = base_z + torch.randn_like(base_z) * 0.5
                recon = model.decode(z_random)
                recon_denorm = denormalize(recon.cpu().squeeze(0))
                
                axes[row, col].imshow(recon_denorm.permute(1, 2, 0).clamp(0, 1))
                axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Latent space exploration saved to {save_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='VAE Inference and Analysis')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--mode', type=str, choices=['reconstruct', 'generate', 'interpolate', 'explore', 'calculate_latents', 'plot'], 
                       required=True, help='Inference mode')
    parser.add_argument('--plot_type', type=str, choices=['pca', 'umap', 'tsne', 'hdbscan'], default='umap', help='Type of plot for analysis')
    parser.add_argument('--image', type=str, help='Path to input image (for reconstruct/interpolate)')
    parser.add_argument('--image2', type=str, help='Path to second image (for interpolate)')
    parser.add_argument('--latents', type=str, help='Path to existing latents (for analysis)')
    parser.add_argument('--image_folder', type=str, help='Path to image folder (for analysis)')
    parser.add_argument('--output', type=str, help='Output path for saving results')
    parser.add_argument('--num_samples', type=int, default=16, help='Number of samples to generate')
    parser.add_argument('--num_steps', type=int, default=10, help='Number of interpolation steps')
    
    args = parser.parse_args()
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model, config = load_model(args.checkpoint, device)
    print("Model loaded successfully!")
    
    # Execute based on mode
    if args.mode == 'reconstruct':
        if not args.image:
            print("Error: --image required for reconstruct mode")
            return
        reconstruct_image(model, args.image, device, args.output)
        
    elif args.mode == 'generate':
        generate_samples(model, device, args.num_samples, args.output)
        
    elif args.mode == 'interpolate':
        if not args.image or not args.image2:
            print("Error: --image and --image2 required for interpolate mode")
            return
        interpolate_between_images(model, args.image, args.image2, device, args.num_steps, args.output)
        
    elif args.mode == 'explore':
        explore_latent_space(model, device, config, args.image, args.output)

    elif args.mode == 'calculate_latents':
        if not args.image_folder or not args.output:
            print("Error: --image_folder and --output required for calculating latents")
            return
        a.calculate_latents(model, args.image_folder, args.output)

    elif args.mode == 'plot':
        if not args.latents:
            print("Error: --latents required for plotting")
            return
        data = a.load_latents(args.latents)
        if args.plot_type == 'pca':
            print("Performing PCA analysis...")
            a.PCA_plot(data)
        elif args.plot_type == 'umap':
            print("Performing UMAP analysis...")
            a.UMAP_plot(data)
        elif args.plot_type == 'tsne':
            print("Performing t-SNE analysis...")
            a.tSNE_plot(data)
        elif args.plot_type == 'hdbscan':
            print("Performing HDBSCAN analysis...")
            a.HDBSCAN(data)


if __name__ == '__main__':
    main() 
