import torch
import matplotlib.pyplot as plt
import numpy as np
from model import VAE  # Ensure this imports your VAE class definition
from dataloader import create_data_loaders

def load_encoder(input_shape, latent_dim, path):
    """
    Load the encoder part of a VAE model from a saved state dictionary.

    Parameters:
        input_shape (tuple): The shape of the input images, e.g., (224, 224).
        latent_dim (int): The dimension of the latent space.
        path (str): Path to the saved encoder state dictionary file.

    Returns:
        VAE: A VAE model with the encoder loaded.
    """
    # Initialize the VAE model with the provided latent dimension, input shape
    vae = VAE(latent_dim=latent_dim, input_shape=input_shape)

    # Load the saved encoder state dictionary
    encoder_state_dict = torch.load(path)

    # Load the state dictionary into the model's encoder
    vae.encoder.load_state_dict(encoder_state_dict)

    # Set the model to evaluation mode if you're using it for inference
    vae.encoder.eval()

    print(f"Encoder model loaded successfully from {path}")
    
    return vae

def get_predictions_from_loader(data_loader, model, device):
    """
    Get the predictions (latent vectors) for a given data loader by passing data through the model.

    Parameters:
        data_loader (DataLoader): The PyTorch DataLoader containing the dataset.
        model (VAE): The VAE model with the encoder loaded.
        device (str): The device ('cpu' or 'cuda') to use for inference.

    Returns:
        tuple: List of predictions (latent vectors) and the corresponding labels.
    """
    model.eval()  # Set the model to evaluation mode
    predictions = []
    labels = []

    with torch.no_grad():  # No need to track gradients for inference
        for images, targets in data_loader:
            # Move images and targets to the appropriate device (CPU or GPU)
            images = images.to(device)
            targets = targets.to(device)

            # Pass the images through the encoder to get latent representations (mu)
            mu, _ = model.encoder(images)  # mu is the latent representation
            predictions.append(mu.cpu())  # Store the latent vectors in predictions
            labels.append(targets.cpu())  # Store the labels

    return predictions, labels

def plot_predictions(predictions, labels, classes, title='Latent Space Predictions'):
    """
    Plot the predicted latent vectors in the latent space with colors corresponding to class labels.

    Parameters:
        predictions (list): Latent vectors predicted by the model.
        labels (list): True class labels corresponding to each prediction.
        classes (list): List of class names for the labels.
        title (str): Title for the plot.
    """
    # Convert predictions and labels to numpy arrays
    predictions = np.concatenate(predictions, axis=0)
    labels = np.concatenate(labels, axis=0)

    # Plotting the latent space
    plt.figure(figsize=(10, 8))

    # Create a scatter plot, using labels for colors
    scatter = plt.scatter(predictions[:, 0], predictions[:, 1], c=labels, cmap='viridis', alpha=0.7)

    # Add color bar for class labels
    cbar = plt.colorbar(scatter, label='Class')
    
    # Add class labels
    cbar.set_ticks(np.arange(len(classes)))
    cbar.set_ticklabels(classes)

    # Labeling the axes
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')

    # Title of the plot
    plt.title(title)

    # Show the plot
    plt.show()




def get_latent_space():
    vae = load_encoder(input_shape=(24, 24), latent_dim=2, path='../Variational_autoencoder/output/result+2024-11-06_23-46-23/vae_model_encoder.pth')

    # Example DataLoader (replace with your actual data loader)
    train_loader, val_loader = create_data_loaders(
        train_path='../../masked_autoencoder/1/Training/',
        val_path='../../masked_autoencoder/1/Validation',
        image_size=(24, 24)
    )

    # Get predictions (latent vectors) and their corresponding labels from the train_loader
    predictions, labels = get_predictions_from_loader(data_loader=train_loader, model=vae, device=device)
    return predictions , labels

device = 'cpu'
get_latent_space()
# # Define the class names (for coloring)
# classes = ['Class 0', 'Class 1']  # Modify as per your classes
# predictions, labels = get_latent_space()
# plot_predictions(predictions, labels, classes)


