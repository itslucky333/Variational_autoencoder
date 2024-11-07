import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_latent_space(encoder, data, labels, latent_dim=2, title="Latent Space Representation"):
    """
    Plots the latent space of the data using the encoder model.

    Parameters:
    - encoder: Trained encoder model that takes data and outputs latent vectors.
    - data: Data to be encoded, typically as a numpy array or tensor.
    - labels: Labels for each data point, for color-coding in the plot.
    - latent_dim: Dimensionality of the latent space (2 for 2D or 3 for 3D).
    - title: Title of the plot.

    Returns:
    - None. Displays a plot of the latent space.
    """

    # Encode the data to get the latent representation
    z_latent = encoder.predict(data)
    
    # Check if latent_dim is 2 or 3 for plotting
    if latent_dim == 2:
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(z_latent[:, 0], z_latent[:, 1], c=labels, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label="Class Label")
        plt.xlabel("Latent Dimension 1")
        plt.ylabel("Latent Dimension 2")
        plt.title(title)
        plt.show()
    
    elif latent_dim == 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(z_latent[:, 0], z_latent[:, 1], z_latent[:, 2], c=labels, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label="Class Label")
        ax.set_xlabel("Latent Dimension 1")
        ax.set_ylabel("Latent Dimension 2")
        ax.set_zlabel("Latent Dimension 3")
        plt.title(title)
        plt.show()
    else:
        print("Latent dimension must be 2 or 3 for visualization.")


import numpy as np
import matplotlib.pyplot as plt

# Simulated encoder function to generate random latent points
class RandomEncoder:
    def predict(self, data):
        # Randomly generate points in a 2D latent space
        return np.random.randn(len(data), 3)  # 2D latent space

# Generate random data and labels
num_samples = 200
data = np.random.randn(num_samples, 10)  # 10-dimensional input data (not used in this example)
labels = np.random.randint(0, 5, num_samples)  # Random labels from 0 to 4

# Initialize the RandomEncoder and plot the latent space
encoder = RandomEncoder()
plot_latent_space(encoder, data, labels, latent_dim=3, title="Random Latent Space")
