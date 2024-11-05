import torch
from torchvision import transforms
from PIL import Image
# from fid_loss import calculate_frechet_distance



def calculate_mean_and_covariance(images):
    # Flatten the images to shape (batch_size, channels * height * width)
    batch_size = images.size(0)
    images_flat = images.view(batch_size, -1)

    # Calculate mean and covariance
    mean = torch.mean(images_flat, dim=0)
    covariance = torch.cov(images_flat.T)

    return mean, covariance

