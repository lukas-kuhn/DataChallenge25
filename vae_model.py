import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ResidualBlock(nn.Module):
    """Residual block with group normalization and swish activation"""
    def __init__(self, in_channels, out_channels, time_emb_dim=None, dropout=0.0, skip_dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skip_dropout = skip_dropout
        
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        self.time_emb_proj = nn.Linear(time_emb_dim, out_channels) if time_emb_dim else None
        
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x, time_emb=None):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        if time_emb is not None and self.time_emb_proj is not None:
            h = h + self.time_emb_proj(F.silu(time_emb))[:, :, None, None]
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        # Apply skip connection dropout during training
        if self.training and self.skip_dropout > 0:
            if torch.rand(1).item() < self.skip_dropout:
                return h  # Skip the residual connection
        
        return h + self.shortcut(x)

class AttentionBlock(nn.Module):
    """Self-attention block for spatial attention"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(32, channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        
        q = self.q(h).view(B, C, H * W).transpose(1, 2)
        k = self.k(h).view(B, C, H * W)
        v = self.v(h).view(B, C, H * W).transpose(1, 2)
        
        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(C)
        attn = torch.softmax(torch.bmm(q, k) * scale, dim=-1)
        h = torch.bmm(attn, v).transpose(1, 2).view(B, C, H, W)
        
        h = self.proj_out(h)
        return x + h

class DownBlock(nn.Module):
    """Downsampling block with residual connections"""
    def __init__(self, in_channels, out_channels, num_layers=2, downsample=True, attention=False, dropout=0.0, skip_dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # First layer might change channels
        self.layers.append(ResidualBlock(in_channels, out_channels, dropout=dropout, skip_dropout=skip_dropout))
        
        # Additional layers
        for _ in range(num_layers - 1):
            self.layers.append(ResidualBlock(out_channels, out_channels, dropout=dropout, skip_dropout=skip_dropout))
            
        if attention:
            self.layers.append(AttentionBlock(out_channels))
            
        self.downsample = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1) if downsample else None
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        
        if self.downsample is not None:
            x = self.downsample(x)
            
        return x

class UpBlock(nn.Module):
    """Upsampling block with residual connections"""
    def __init__(self, in_channels, out_channels, num_layers=2, upsample=True, attention=False, dropout=0.0, skip_dropout=0.0):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, 4, stride=2, padding=1) if upsample else None
        
        self.layers = nn.ModuleList()
        # First layer might change channels
        self.layers.append(ResidualBlock(in_channels, out_channels, dropout=dropout, skip_dropout=skip_dropout))
        
        # Additional layers
        for _ in range(num_layers - 1):
            self.layers.append(ResidualBlock(out_channels, out_channels, dropout=dropout, skip_dropout=skip_dropout))
            
        if attention:
            self.layers.append(AttentionBlock(out_channels))
        
    def forward(self, x):
        if self.upsample is not None:
            x = self.upsample(x)
            
        for layer in self.layers:
            x = layer(x)
            
        return x

class UNetEncoder(nn.Module):
    """UNet encoder for VAE"""
    def __init__(self, in_channels=3, model_channels=128, channel_mult=(1, 2, 3, 4, 5), 
                 num_res_blocks=3, attention_resolutions=(16, 8), dropout=0.0, skip_dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        
        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        # Downsampling blocks
        self.down_blocks = nn.ModuleList()
        channels = [model_channels * mult for mult in channel_mult]
        
        in_ch = model_channels
        for i, out_ch in enumerate(channels):
            attention = (224 // (2 ** i)) in attention_resolutions
            downsample = i < len(channels) - 1
            
            self.down_blocks.append(
                DownBlock(in_ch, out_ch, num_res_blocks, downsample, attention, dropout, skip_dropout)
            )
            in_ch = out_ch
            
        # Middle block
        self.mid_block = nn.Sequential(
            ResidualBlock(channels[-1], channels[-1], dropout=dropout, skip_dropout=skip_dropout),
            AttentionBlock(channels[-1]),
            ResidualBlock(channels[-1], channels[-1], dropout=dropout, skip_dropout=skip_dropout)
        )
        
    def forward(self, x):
        h = self.conv_in(x)
        
        for block in self.down_blocks:
            h = block(h)
            
        h = self.mid_block(h)
        return h

class UNetDecoder(nn.Module):
    """UNet decoder for VAE"""
    def __init__(self, out_channels=3, model_channels=128, channel_mult=(1, 2, 3, 4, 5),
                 num_res_blocks=3, attention_resolutions=(16, 8), dropout=0.0, skip_dropout=0.0):
        super().__init__()
        self.out_channels = out_channels
        self.model_channels = model_channels
        
        channels = [model_channels * mult for mult in channel_mult]
        
        # Middle block
        self.mid_block = nn.Sequential(
            ResidualBlock(channels[-1], channels[-1], dropout=dropout, skip_dropout=skip_dropout),
            AttentionBlock(channels[-1]),
            ResidualBlock(channels[-1], channels[-1], dropout=dropout, skip_dropout=skip_dropout)
        )
        
        # Upsampling blocks
        self.up_blocks = nn.ModuleList()
        reversed_channels = list(reversed(channels))
        
        for i, out_ch in enumerate(reversed_channels[1:] + [model_channels]):
            in_ch = reversed_channels[i]
            attention = (7 * (2 ** i)) in attention_resolutions
            upsample = i < len(reversed_channels) - 1
            
            self.up_blocks.append(
                UpBlock(in_ch, out_ch, num_res_blocks, upsample, attention, dropout, skip_dropout)
            )
            
        # Final convolution
        self.norm_out = nn.GroupNorm(32, model_channels)
        self.conv_out = nn.Conv2d(model_channels, out_channels, 3, padding=1)
        
    def forward(self, h):
        h = self.mid_block(h)
        
        for block in self.up_blocks:
            h = block(h)
            
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        return h

class VAE(nn.Module):
    """Memory-optimized UNet-based VAE with 1D bottleneck for high-quality image reconstruction"""
    def __init__(self, in_channels=3, latent_dim=64, model_channels=96, 
                 channel_mult=(1, 1, 2, 3, 4), num_res_blocks=2, 
                 attention_resolutions=(16,), dropout=0.0, skip_dropout=0.0):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = UNetEncoder(
            in_channels=in_channels,
            model_channels=model_channels,
            channel_mult=channel_mult,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            skip_dropout=skip_dropout
        )
        
        # Calculate the spatial dimensions after encoding
        # For 224x224 input with 4 downsampling layers: 224 -> 112 -> 56 -> 28 -> 14 -> 7
        self.final_spatial_size = 7
        final_channels = model_channels * channel_mult[-1]
        self.final_channels = final_channels
        
        # Dummy forward pass to calculate actual encoder output size
        with torch.no_grad():
            dummy_input = torch.randn(1, in_channels, 224, 224)
            dummy_output = self.encoder(dummy_input)
            self.encoder_output_size = dummy_output.numel() // dummy_output.shape[0]
        
        # 1D Latent space projection (much smaller bottleneck)
        self.to_mu = nn.Linear(self.encoder_output_size, latent_dim)
        self.to_logvar = nn.Linear(self.encoder_output_size, latent_dim)
        self.from_latent = nn.Linear(latent_dim, self.encoder_output_size)
        
        # Decoder
        self.decoder = UNetDecoder(
            out_channels=in_channels,
            model_channels=model_channels,
            channel_mult=channel_mult,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            skip_dropout=skip_dropout
        )
        
    def encode(self, x):
        h = self.encoder(x)
        # Flatten spatial dimensions for 1D bottleneck
        batch_size = h.shape[0]
        h_flat = h.view(batch_size, -1)
        mu = self.to_mu(h_flat)
        logvar = self.to_logvar(h_flat)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = self.from_latent(z)
        # Reshape back to spatial dimensions for decoder
        batch_size = h.shape[0]
        # Calculate actual spatial dimensions from encoder output size
        total_elements = self.encoder_output_size
        spatial_size = int((total_elements // self.final_channels) ** 0.5)
        h = h.view(batch_size, self.final_channels, spatial_size, spatial_size)
        return self.decoder(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
    
    def sample(self, num_samples, device):
        # Sample from 1D latent space
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)

    def get_1d_latent(self, x):
        """Get a 1D latent representation suitable for PCA.
        Returns the mean of the latent distribution (mu) which is already 1D.
        """
        mu, _ = self.encode(x)
        return mu  # Shape: [batch_size, latent_dim]

def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """VAE loss with KL divergence and reconstruction loss"""
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + beta * kl_loss, recon_loss, kl_loss 