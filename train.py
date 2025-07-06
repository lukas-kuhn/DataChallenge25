import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
from PIL import Image
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import json
import time
from pathlib import Path
import wandb

from vae_model import VAE, vae_loss

class CoinDataset(Dataset):
    """Dataset for loading coin images from obverse folder"""
    def __init__(self, data_dir, transform=None, extensions=('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        
        # Collect all image paths from all class subdirectories
        for class_dir in os.listdir(data_dir):
            class_path = os.path.join(data_dir, class_dir)
            if os.path.isdir(class_path):
                for ext in extensions:
                    pattern = os.path.join(class_path, f'*{ext}')
                    self.image_paths.extend(glob.glob(pattern))
                    pattern = os.path.join(class_path, f'*{ext.upper()}')
                    self.image_paths.extend(glob.glob(pattern))
        
        print(f"Found {len(self.image_paths)} images in {data_dir}")
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        try:
            # Load and convert image
            image = Image.open(image_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
                
            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a random tensor as fallback
            return torch.randn(3, 224, 224)

def get_transforms(image_size=224):
    """Get data transforms for training and validation"""
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def denormalize(tensor):
    """Denormalize tensor for visualization"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean

def save_reconstruction_samples(model, dataloader, device, save_path, num_samples=8):
    """Save reconstruction samples for visualization"""
    model.eval()
    with torch.no_grad():
        batch = next(iter(dataloader))
        # Ensure we don't request more samples than available
        actual_samples = min(num_samples, batch.size(0))
        batch = batch[:actual_samples].to(device)
        
        recon, _, _ = model(batch)
        
        # Denormalize for visualization
        batch_denorm = denormalize(batch.cpu())
        recon_denorm = denormalize(recon.cpu())
        
        # Create comparison plot
        fig, axes = plt.subplots(2, actual_samples, figsize=(2 * actual_samples, 4))
        
        # Handle case where actual_samples is 1 (axes won't be 2D)
        if actual_samples == 1:
            axes = axes.reshape(2, 1)
        
        for i in range(actual_samples):
            # Original images
            axes[0, i].imshow(batch_denorm[i].permute(1, 2, 0).clamp(0, 1))
            axes[0, i].set_title('Original')
            axes[0, i].axis('off')
            
            # Reconstructed images
            axes[1, i].imshow(recon_denorm[i].permute(1, 2, 0).clamp(0, 1))
            axes[1, i].set_title('Reconstructed')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return fig

def save_generated_samples(model, device, save_path, num_samples=16):
    """Save generated samples from random latent vectors"""
    model.eval()
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
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return fig

def train_epoch(model, dataloader, optimizer, device, beta=1.0, scaler=None):
    """Train for one epoch with optional mixed precision"""
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        batch = batch.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            # Mixed precision training
            with autocast():
                recon, mu, logvar = model(batch)
                loss, recon_loss, kl_loss = vae_loss(recon, batch, mu, logvar, beta)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()
        else:
            # Regular training
            recon, mu, logvar = model(batch)
            loss, recon_loss, kl_loss = vae_loss(recon, batch, mu, logvar, beta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
        
        # Check for NaN values
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: NaN or Inf loss detected at step {len(pbar)}. Skipping batch.")
            continue
            
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()
        
        pbar.set_postfix({
            'Loss': f'{loss.item():.2f}',
            'Recon': f'{recon_loss.item():.2f}',
            'KL': f'{kl_loss.item():.2f}'
        })
    
    avg_loss = total_loss / len(dataloader)
    avg_recon_loss = total_recon_loss / len(dataloader)
    avg_kl_loss = total_kl_loss / len(dataloader)
    
    return avg_loss, avg_recon_loss, avg_kl_loss

def validate_epoch(model, dataloader, device, beta=1.0):
    """Validate for one epoch with optional mixed precision"""
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for batch in pbar:
            batch = batch.to(device, non_blocking=True)
            
            with autocast():
                recon, mu, logvar = model(batch)
                loss, recon_loss, kl_loss = vae_loss(recon, batch, mu, logvar, beta)
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.2f}',
                'Recon': f'{recon_loss.item():.2f}',
                'KL': f'{kl_loss.item():.2f}'
            })
    
    avg_loss = total_loss / len(dataloader)
    avg_recon_loss = total_recon_loss / len(dataloader)
    avg_kl_loss = total_kl_loss / len(dataloader)
    
    return avg_loss, avg_recon_loss, avg_kl_loss

def main():
    parser = argparse.ArgumentParser(description='Train large UNet VAE for coin images')
    parser.add_argument('--data_dir', type=str, default='obverse', help='Path to coin data directory')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory for models and logs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--beta', type=float, default=0.01, help='Beta for KL loss weighting')
    parser.add_argument('--beta_schedule', action='store_true', help='Use beta scheduling')
    parser.add_argument('--latent_dim', type=int, default=128, help='Latent dimension')
    parser.add_argument('--model_channels', type=int, default=128, help='Base model channels')
    parser.add_argument('--num_res_blocks', type=int, default=3, help='Number of residual blocks per down/up block')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--skip_dropout', type=float, default=0.1, help='Skip connection dropout rate')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--validation_split', type=float, default=0.1, help='Validation split ratio')
    parser.add_argument('--save_every', type=int, default=10, help='Save model every N epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--mixed_precision', action='store_true', help='Use automatic mixed precision training')
    parser.add_argument('--wandb_project', type=str, default='vae-coin-training', help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='Wandb run name')
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb logging')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'samples'), exist_ok=True)
    
    # Save config
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Device setup
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize wandb
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
            save_code=True
        )
    
    # Data setup
    train_transform, val_transform = get_transforms()
    
    # Load full dataset
    full_dataset = CoinDataset(args.data_dir, transform=train_transform)
    
    # Split into train and validation
    total_size = len(full_dataset)
    val_size = int(args.validation_split * total_size)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Update validation dataset transform
    val_dataset.dataset.transform = val_transform
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Model setup
    model = VAE(
        in_channels=3,
        latent_dim=args.latent_dim,
        model_channels=args.model_channels,
        channel_mult=(1, 2, 3, 4, 5, 6),  # Scaled up for fine-grained details
        num_res_blocks=args.num_res_blocks,
        attention_resolutions=(32, 16, 8),  # Multi-scale attention for better details
        dropout=args.dropout,
        skip_dropout=args.skip_dropout
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer and scheduler with more conservative settings
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-8, eps=1e-8)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])
    
    # Training loop
    print("Starting training...")
    for epoch in range(start_epoch, args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        
        # Beta scheduling for KL loss
        if args.beta_schedule:
            # Gradually increase beta from 0 to target value over more epochs
            beta = min(args.beta, args.beta * (epoch / 100))
        else:
            beta = args.beta
        
        # Train
        start_time = time.time()
        train_loss, train_recon, train_kl = train_epoch(model, train_loader, optimizer, device, beta)
        train_time = time.time() - start_time
        
        # Validate
        start_time = time.time()
        val_loss, val_recon, val_kl = validate_epoch(model, val_loader, device, beta)
        val_time = time.time() - start_time
        
        # Update scheduler
        scheduler.step()
        
        # Log results
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Train Loss: {train_loss:.4f} (Recon: {train_recon:.4f}, KL: {train_kl:.4f}) - {train_time:.1f}s")
        print(f"Val Loss: {val_loss:.4f} (Recon: {val_recon:.4f}, KL: {val_kl:.4f}) - {val_time:.1f}s")
        print(f"Beta: {beta:.4f}, LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # Log to wandb
        if not args.no_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train/loss': train_loss,
                'train/reconstruction_loss': train_recon,
                'train/kl_loss': train_kl,
                'val/loss': val_loss,
                'val/reconstruction_loss': val_recon,
                'val/kl_loss': val_kl,
                'beta': beta,
                'learning_rate': scheduler.get_last_lr()[0],
                'train_time': train_time,
                'val_time': val_time
            })
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'config': vars(args)
            }, os.path.join(args.output_dir, 'checkpoints', 'best_model.pth'))
            
            print(f"New best validation loss: {best_val_loss:.4f}")
        
        # Save periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'config': vars(args)
            }, os.path.join(args.output_dir, 'checkpoints', f'checkpoint_epoch_{epoch + 1}.pth'))
        
        # Save sample reconstructions and generations
        if (epoch + 1) % 5 == 0:
            recon_fig = save_reconstruction_samples(
                model, val_loader, device,
                os.path.join(args.output_dir, 'samples', f'recon_epoch_{epoch + 1}.png')
            )
            gen_fig = save_generated_samples(
                model, device,
                os.path.join(args.output_dir, 'samples', f'generated_epoch_{epoch + 1}.png')
            )
            
            # Log images to wandb
            if not args.no_wandb:
                wandb.log({
                    'reconstructions': wandb.Image(recon_fig),
                    'generated_samples': wandb.Image(gen_fig)
                })
    
    print("Training completed!")
    
    # Save final model
    torch.save({
        'epoch': args.num_epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'config': vars(args)
    }, os.path.join(args.output_dir, 'checkpoints', 'final_model.pth'))
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Log Scale)')
    plt.title('Training and Validation Loss (Log Scale)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    main()
