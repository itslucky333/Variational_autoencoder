import numpy as np
import matplotlib.pyplot as plt

# Mock decoder to generate random grayscale images from latent vectors
class RandomDecoder:
    def decode(self, latent_vector):
        # For demonstration, return a grayscale image based on the latent vector
        # Here we generate a 28x28 image with values based on the latent vector
        base_value = (latent_vector[0] + latent_vector[1]) / 10
        return np.full((28, 28), base_value)  # Simple grayscale image based on latent position

def plot_latent_space_grid(decoder, grid_size=10, latent_range=(-5, 5)):
    """
    Generates and plots a grid of images by traversing the latent space.

    Parameters:
    - decoder: Decoder model that takes a latent vector and returns an image.
    - grid_size: Size of the grid (grid_size x grid_size images).
    - latent_range: Tuple indicating the min and max values for latent space traversal.

    Returns:
    - None. Displays a grid plot of decoded images.
    """
    # Create a linear space for each axis within the latent range
    latent_values = np.linspace(latent_range[0], latent_range[1], grid_size)
    
    # Set up the plot grid
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    fig.suptitle("Latent Space Grid of Generated Images", fontsize=16)
    
    # Traverse the latent space and decode images
    for i, x in enumerate(latent_values):
        for j, y in enumerate(latent_values):
            # Create a latent vector for each grid position
            latent_vector = np.array([x, y])
            # Decode the latent vector to generate an image
            img = decoder.decode(latent_vector)
            # Plot the generated image in the grid
            axes[i, j].imshow(img, cmap='gray')
            axes[i, j].axis('off')  # Hide axes for a cleaner look
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()

# Usage example
decoder = RandomDecoder()
plot_latent_space_grid(decoder, grid_size=10, latent_range=(-5, 5))
