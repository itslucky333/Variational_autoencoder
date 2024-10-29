import torch
import torch.nn as nn
import torch.optim as optim


from vae_loss import vae_loss

class Encoder(nn.Module):
    def __init__(self, latent_dim=2):
        super(Encoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(128 * 28 * 28, latent_dim)  # Mean vector
        self.fc_logvar = nn.Linear(128 * 28 * 28, latent_dim)  # Log variance vector

    def forward(self, x):
        x = self.conv(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim=2):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 128 * 28 * 28)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Output range [0, 1] for images
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 128, 28, 28)
        x = self.deconv(x)
        return x

class VAE(nn.Module):
    def __init__(self, latent_dim=2):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar

    def sample(self, num_samples):
        """Generates new images by sampling from the latent space.

        Args:
                num_samples (int): The number of samples to generate.

        Returns:
                torch.Tensor: Generated images of shape (num_samples, 3, H, W).
        """
        # Sample random points in the latent space
        z = torch.randn(num_samples, self.encoder.fc_mu.out_features).to(next(self.parameters()).device)
        # Decode the sampled points to generate images
        samples = self.decoder(z)
        return samples
