import numpy as np
from fid_loss import calculate_frechet_distance

# Set a random seed for reproducibility
np.random.seed(42)

# Generate random means (mu1 and mu2) for two Gaussian distributions
mu1 = np.random.rand(2048)  # Assuming the feature vector has 2048 dimensions
mu2 = np.random.rand(2048)

# Generate random covariance matrices (sigma1 and sigma2)
sigma1 = np.random.rand(2048, 2048)
sigma1 = np.dot(sigma1, sigma1.transpose())  # Ensure it's positive semi-definite
sigma2 = np.random.rand(2048, 2048)
sigma2 = np.dot(sigma2, sigma2.transpose())  # Ensure it's positive semi-definite

# Calculate the Frechet distance (FID)
fid_value = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

print("Frechet Inception Distance (FID):", fid_value)
