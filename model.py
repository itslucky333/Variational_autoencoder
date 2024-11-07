import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, latent_dim=16, input_shape=(224, 224)):
        super(Encoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate flattened dimension
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, *input_shape)
            self.flattened_dim = self.conv(dummy_input).view(1, -1).size(1)
        
        self.fc_mu = nn.Linear(self.flattened_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_dim, latent_dim)

    def forward(self, x):
        x = self.conv(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim=16, output_shape=(512, 512)):
        super(Decoder, self).__init__()
        
        self.initial_height = output_shape[0] // 16
        self.initial_width = output_shape[1] // 16
        
        self.fc = nn.Linear(latent_dim, 256 * self.initial_height * self.initial_width)
        
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Output range [0, 1]
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 256, self.initial_height, self.initial_width)
        x = self.deconv(x)
        return x


class VAE(nn.Module):
    def __init__(self, latent_dim=16, input_shape=(224, 224), output_shape=(512, 512)):
        super(VAE, self).__init__()
        print(f"Latent dimension: {latent_dim}")
        self.encoder = Encoder(latent_dim, input_shape=input_shape)
        self.decoder = Decoder(latent_dim, output_shape=output_shape)

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
        z = torch.randn(num_samples, self.encoder.fc_mu.out_features).to(next(self.parameters()).device)
        samples = self.decoder(z)
        return samples
