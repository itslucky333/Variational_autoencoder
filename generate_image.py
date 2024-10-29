# main.py
import torch
from model import VAE  # Import the VAE class from vae_model.py
import matplotlib.pyplot as plt  # Import Matplotlib for plotting

# Initialize the VAE model
latent_dim = 2  # Set this to whatever you need
vae = VAE(latent_dim)

# Sample images after training the model
num_samples = 64  # Number of samples to generate
sampled_images = vae.sample(num_samples)

# Print the shape of the sampled images
print(sampled_images.shape)

# Display the sampled images in a grid of 8x8

def plot_and_save_generated_images(sampled_images):
    fig, axes = plt.subplots(8, 8, figsize=(10, 10))  # Create a grid of 8x8
    axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration

    # Iterate through the axes and images
    for ax, img in zip(axes, sampled_images):
        img = img.permute(1, 2, 0)  # Change the shape from (C, H, W) to (H, W, C)
        ax.imshow(img.detach().cpu().numpy())  # Convert to numpy for plotting
        ax.axis('off')  # Turn off axis labels

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

plot_and_save_generated_images(sampled_images=sampled_images)