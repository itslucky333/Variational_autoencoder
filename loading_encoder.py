import torch
import matplotlib.pyplot as plt
import numpy as np
from model import VAE
from dataloader import create_data_loaders, print_batch_info


def load_vae(encoder_path, decoder_path, input_shape=(224, 224), latent_dim=30):
    """
    Load both encoder and decoder parts of a VAE model from saved state dictionaries.

    Parameters:
        input_shape (tuple): The shape of the input images, e.g., (224, 224).
        latent_dim (int): The dimension of the latent space.
        encoder_path (str): Path to the saved encoder state dictionary file.
        decoder_path (str): Path to the saved decoder state dictionary file.

    Returns:
        VAE: A VAE model with both the encoder and decoder loaded.
    """
    # Initialize the VAE model with the provided latent dimension and input shape
    vae = VAE(latent_dim=latent_dim, input_shape=input_shape)

    # Load the saved encoder and decoder state dictionaries
    encoder_state_dict = torch.load(encoder_path, weights_only=True)
    decoder_state_dict = torch.load(decoder_path, weights_only=True)

    # Load the state dictionaries into the model's encoder and decoder
    vae.encoder.load_state_dict(encoder_state_dict)
    vae.decoder.load_state_dict(decoder_state_dict)

    # Set both encoder and decoder to evaluation mode if using for inference
    vae.encoder.eval()
    vae.decoder.eval()

    print(f"Encoder and Decoder models loaded successfully from {encoder_path} and {decoder_path}")

    return vae


def get_mu_and_log_var(vae, dataloader, num_batches=1):
    """
    Extract the mean (mu) and log-variance (logvar) from the encoder for a batch of images.

    Parameters:
        vae (VAE): The VAE model with a loaded encoder.
        dataloader (DataLoader): A PyTorch DataLoader providing batches of images.
        num_batches (int): Number of batches to process (default is 1).

    Returns:
        tuple: (mu, logvar) for the last processed batch.
    """
    # Ensure the encoder is in evaluation mode
    vae.encoder.eval()

    mu_logvar_list = []
    with torch.no_grad():  # Disable gradient computation for inference
        for i, (images, _) in enumerate(dataloader):
            if i >= num_batches:
                break

            # Get mu and logvar from the encoder
            mu, logvar = vae.encoder(images)
            mu_logvar_list.append((mu, logvar))

            print(f"Batch {i+1} - Mu: {mu.shape}, LogVar: {logvar.shape}")
    
    return mu, logvar

def display_images(images, recon_images, num_images=25):
    """
    Display a 5x5 grid of original and reconstructed images.

    Parameters:
        images (Tensor): A batch of original images.
        recon_images (Tensor): A batch of reconstructed images.
        num_images (int): Number of images to display (default is 25).
    """
    # Ensure both original and reconstructed images have the same number
    images = images[:num_images]
    recon_images = recon_images[:num_images]

    # Create a 5x10 grid for displaying images (5 rows and 10 columns)
    fig, axes = plt.subplots(5, 10, figsize=(15, 7))
    
    for i in range(5):
        for j in range(5):
            # Original image on the left side
            ax = axes[i, j]
            ax.imshow(images[i * 5 + j].permute(1, 2, 0).cpu().numpy())  # (C, H, W) -> (H, W, C)
            ax.axis('off')

            # Reconstructed image on the right side
            ax = axes[i, j + 5]
            ax.imshow(recon_images[i * 5 + j].permute(1, 2, 0).detach().cpu().numpy())  # (C, H, W) -> (H, W, C)
            ax.axis('off')

    plt.suptitle("Original Images (Left) and Reconstructed Images (Right)")
    plt.show()



import torch
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

def latent_space_traversal(latent_vector, decoder, latent_dim=30, num_steps=10, spacing=1.0, device='cpu'):
    """
    Function to traverse the latent space by varying each dimension of the latent vector.
    
    Parameters:
        latent_vector (Tensor): The latent vector for a single image of shape (1, latent_dim).
        decoder (nn.Module): The trained decoder model to generate images from latent space.
        latent_dim (int): The dimensionality of the latent space (default is 30).
        num_steps (int): Number of steps to vary each dimension (default is 10).
        spacing (float): The spacing of the changes in each dimension (default is 1.0).
        device (str): The device on which to run the model ('cpu' or 'cuda').
    """
    # Ensure the latent vector is on the correct device
    latent_vector = latent_vector.to(device)

    # Prepare the figure for plotting with large image sizes
    fig, axes = plt.subplots(latent_dim, num_steps, figsize=(20, 40))  # Increase figure size

    for dim in range(latent_dim):
        # Start with the original latent vector
        original_latent_vector = latent_vector.clone()

        for i in range(num_steps):
            # Create a new latent vector where only the `dim`-th dimension is modified
            new_latent_vector = original_latent_vector.clone()
            new_latent_vector[0, dim] = latent_vector[0, dim].item() + i * spacing  # Modify the `dim`-th dimension

            # Decode the new latent vector and generate the corresponding image
            with torch.no_grad():
                generated_image = decoder(new_latent_vector)

            # Plot the generated image
            ax = axes[dim, i]
            ax.imshow(generated_image.squeeze(0).permute(1, 2, 0).cpu().numpy())  # Remove batch dimension and permute to (H, W, C)
            ax.axis('off')
            if i == 0:
                ax.set_title(f"Latent Dim {dim}")

    plt.suptitle(f"Latent Space Traversal (Varying each dimension)", fontsize=16)
    
    # Adjust layout and spacing to make images larger
    plt.subplots_adjust(hspace=0.5, wspace=0.5)

    # Show the plot
    plt.savefig("output/latentspacetraversel.png")
    plt.show()




def main():
    train_loader, val_loader = create_data_loaders(
        train_path='../../masked_autoencoder/1/Training/',
        val_path='../../masked_autoencoder/1/Validation/'
    )
    vae = load_vae(
        encoder_path='output/result+2024-11-09_13-48-12/vae_model_encoder.pth',
        decoder_path='output/result+2024-11-09_13-48-12/vae_model_decoder.pth'
    )
    mu, log_var = get_mu_and_log_var(vae, train_loader, 1)
    print("Mu shape:", mu.shape)
    print("LogVar shape:", log_var.shape)
    reconstruted_image = vae.decoder(mu)
    print(reconstruted_image.shape)

    images, _ = next(iter(train_loader))
    
    # Pass the images through the VAE to get reconstructed images
    mu, log_var = get_mu_and_log_var(vae, train_loader, 1)
    reconstructed_image = vae.decoder(mu)
    
    # Display original and reconstructed images
    display_images(images, reconstructed_image, num_images=25)

    # Get the 6th image's latent vector (mu)
    latent_vector_nth_image = mu[0].unsqueeze(0)  
    
    # Now perform latent space traversal using the 6th image's latent vector
    latent_space_traversal(latent_vector_nth_image, vae.decoder, latent_dim=30, num_steps=10, spacing=0.5, device='cpu')
if __name__ == "__main__":
    main()
