import torch
import torch.nn as nn

def vae_loss(recon_x, x, mu, logvar,alpha_value ,beta_value):
    """
    Computes the VAE loss, which consists of the reconstruction loss and KL divergence.

    Args:
        recon_x: Reconstructed images from the decoder.
        x: Original images from the dataset.
        mu: Mean vector from the encoder.
        logvar: Log variance vector from the encoder.

    Returns:
        Total VAE loss.
    """
    # Reconstruction loss (Binary Cross-Entropy)
    recon_loss = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL divergence loss
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return (alpha_value * recon_loss + beta_value * kl_div)
