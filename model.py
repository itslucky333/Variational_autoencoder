import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os 
import numpy as np

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
    def __init__(self, latent_dim=16, output_shape=(224,224)):
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
    def __init__(self, latent_dim=16, input_shape=(224, 224), output_shape=(224, 224)):
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

    def reconstruction(self, dataloader, num_images=25, image_save_path =None, latent_space_save_path = None,name = "reconstruction.png"):
        # Set the model to evaluation mode
        self.eval()

        # Grab a batch of images from the dataloader
        images, _ = next(iter(dataloader))
        images = images[:num_images]  # Select only the first 'num_images' images

        # Pass the images through the encoder to get 'mu' and 'logvar'
        mu, logvar = self.encoder(images)

        # Reparameterize to sample the latent vectors
        z = self.reparameterize(mu, logvar)

        latent_save_path = os.path.join(latent_space_save_path, name)
        np.save(latent_save_path, z.cpu().detach().numpy())

        # Pass the latent vectors through the decoder to get the reconstructed images
        recon_images = self.decoder(z)

        # Convert images to numpy for plotting
        images = images.cpu().detach().numpy()
        recon_images = recon_images.cpu().detach().numpy()

        # Plot original images and reconstructed images side by side
        fig, axes = plt.subplots(5, 10, figsize=(15, 7))
        for i in range(5):
            for j in range(5):
                # Original image
                ax = axes[i, j]
                ax.imshow(images[i * 5 + j].transpose(1, 2, 0))  # (C, H, W) -> (H, W, C)
                ax.axis('off')

                # Reconstructed image
                ax = axes[i, j + 5]
                ax.imshow(recon_images[i * 5 + j].transpose(1, 2, 0))  # (C, H, W) -> (H, W, C)
                ax.axis('off')

        save_file_path = os.path.join(image_save_path, name)
        plt.savefig(save_file_path)