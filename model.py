import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, latent_dim=2, input_shape=(224, 224)):
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
        
        # Automatically calculate the flattened dimension from the input shape
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
    def __init__(self, latent_dim=2, output_shape=(512, 512)):
        super(Decoder, self).__init__()
        
        # Calculate the initial height and width after upsampling
        self.initial_height = output_shape[0] // 8
        self.initial_width = output_shape[1] // 8
        
        # Update this layer to match the decoder's input size calculation
        self.fc = nn.Linear(latent_dim, 128 * self.initial_height * self.initial_width)
        
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Output range [0, 1] for normalized images
        )

    def forward(self, z):
        # Ensuring z matches the expected shape
        x = self.fc(z).view(-1, 128, self.initial_height, self.initial_width)
        x = self.deconv(x)
        return x



class VAE(nn.Module):
    def __init__(self, latent_dim=2, input_shape=(224, 224), output_shape=(512, 512)):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim, input_shape=input_shape)
        self.decoder = Decoder(latent_dim, output_shape=output_shape)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        print(f"latent vector (z) shape : {z.shape}")
        recon_x = self.decoder(z)
        print(f"Reconstructed image shape: {recon_x.shape}") 
        return recon_x, mu, logvar

    def sample(self, num_samples):
        z = torch.randn(num_samples, self.encoder.fc_mu.out_features).to(next(self.parameters()).device)
        samples = self.decoder(z)
        return samples
