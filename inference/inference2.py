from Variational_autoencoder.inference.inference import get_latent_space
from model import VAE
import torch
import matplotlib.pyplot as plt

# Function to load the decoder part of the VAE model
def load_decoder(input_shape, latent_dim, path):
    """
    Load the decoder part of the VAE model from a saved state dictionary.

    Parameters:
        input_shape (tuple): The shape of the input images, e.g., (224, 224).
        latent_dim (int): The dimension of the latent space.
        path (str): Path to the saved decoder state dictionary file.

    Returns:
        VAE: A VAE model with the decoder loaded.
    """
    # Initialize the VAE model with the provided latent dimension, input shape
    vae = VAE(latent_dim=latent_dim, input_shape=input_shape)

    # Load the saved decoder state dictionary
    decoder_state_dict = torch.load(path)

    # Load the state dictionary into the model's decoder with strict=False to allow mismatches
    try:
        vae.decoder.load_state_dict(decoder_state_dict, strict=False)
        print(f"Decoder model loaded successfully from {path}")
    except RuntimeError as e:
        print(f"Error loading state_dict: {e}")

    # Set the model to evaluation mode for inference
    vae.decoder.eval()
    
    return vae

# Function to perform inference (process multiple batches to get the required number of images)
def perform_inference(vae, latent_vectors, device='cpu', batch_size=8, num_images_to_display=25):
    """
    Perform inference to generate reconstructed images from latent vectors.

    Parameters:
        vae (VAE): The VAE model with the decoder loaded.
        latent_vectors (tensor): Latent vectors from the VAE's encoder.
        device (str): Device to run the inference on ('cpu' or 'cuda').
        batch_size (int): The batch size for inference.
        num_images_to_display (int): The number of images to display.

    Returns:
        tensor: Reconstructed images.
    """
    vae = vae.to(device)  # Move the model to the appropriate device
    latent_vectors = latent_vectors.to(device)  # Move latent vectors to the appropriate device

    reconstructed_images = []
    num_images = latent_vectors.size(0)

    # Process batches of latent vectors
    for i in range(0, num_images, batch_size):
        batch_latent_vectors = latent_vectors[i:i + batch_size]
        reconstructed_batch = vae.decoder(batch_latent_vectors)  # Generate reconstructed images
        reconstructed_images.append(reconstructed_batch)
        
        # Stop if we have enough images
        if len(reconstructed_images) * batch_size >= num_images_to_display:
            break

    # Concatenate all reconstructed images
    reconstructed_images = torch.cat(reconstructed_images, dim=0)
    return reconstructed_images[:num_images_to_display]  # Return only the required number of images

# Load the VAE decoder model
decoder_path = '../Variational_autoencoder/output/result+2024-11-06_23-46-23/vae_model_decoder.pth'
vae = load_decoder(input_shape=(24, 24), latent_dim=2, path=decoder_path)

# Get latent space predictions (latent vectors) and their corresponding labels
predictions, labels = get_latent_space()

# Assuming predictions is a list of latent vectors, convert them into a tensor
latent_vectors = torch.cat(predictions, dim=0)  # Concatenate along batch dimension

# Perform inference (generate reconstructed images from latent vectors)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
reconstructed_images = perform_inference(vae, latent_vectors, device=device, batch_size=8, num_images_to_display=25)

# Visualize the reconstructed images in a 5x5 grid
fig, axes = plt.subplots(5, 5, figsize=(15, 15))

# Loop through the grid and display the images
for i, ax in enumerate(axes.flat):
    if i < len(reconstructed_images):
        ax.imshow(reconstructed_images[i].permute(1, 2, 0).cpu().detach().numpy())  # Permute to [H, W, C] for plotting
        ax.axis('off')  # Hide the axes
    else:
        ax.axis('off')  # Hide empty subplots

plt.show()
