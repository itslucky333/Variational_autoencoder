import os
import argparse
import ast  # Import ast to safely evaluate the tuple string
import torch
import torch.optim as optim
from model import VAE # Assuming model.py contains your VAE class and loss function
from dataloader import create_data_loaders, print_batch_info
from create_output_directory import create_output_directory, create_parameter_file , write_metrics_json , plot_and_save_generated_images, create_directory, save_model
from vae_loss import vae_loss
from fid.calculate_mean_and_covariance_for_fid import calculate_mean_and_covariance
from fid.fid_loss import calculate_frechet_distance
from torchsummary import summary

def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_path=args.train_data,
        val_path=args.val_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size  # Assuming image_size is now a tuple
    )

    # Initialize model

    # Print batch info
    # print_batch_info(data_loader=train_loader, batch_idx=0)

    # Training loop (simplified example)
    directory = create_output_directory()
    create_parameter_file(directory= directory,args= args,device=device)
    metrics_file_path = os.path.join(directory, "metrics.json")
    num_of_images_in_generated_grid = 64
    generated_image_directory_path = create_directory(directory, "generated_images")
    reconstruction_directory = create_directory( directory,"reconstruction_images")
    latent_space_directory = create_directory(directory, "latent_spaces")
    beta_value = args.beta_value
    alpha_value = args.alpha_value
    

    vae = VAE(latent_dim=args.latent_dim, input_shape=args.image_size, output_shape=args.image_size).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=args.learning_rate)
    summary(vae, input_size = (3,224,224))
    for epoch in range(args.epochs):
        vae.train()
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            optimizer.zero_grad()

            # Forward pass through the VAE
            recon_images, mu, logvar = vae(images)

            # Calculate mean and covariance for real and generated images
            # mean_real, cov_real = calculate_mean_and_covariance(images)
            # mean_generated, cov_generated = calculate_mean_and_covariance(recon_images)

            # # Detach the mean and covariance tensors to prevent gradient tracking
            # mean_real, cov_real = mean_real.detach(), cov_real.detach()
            # mean_generated, cov_generated = mean_generated.detach(), cov_generated.detach()

            # # Calculate Frechet Distance
            # fid_value = calculate_frechet_distance(
            #     mu1=mean_real.cpu().numpy(),
            #     sigma1=cov_real.cpu().numpy(),
            #     mu2=mean_generated.cpu().numpy(),
            #     sigma2=cov_generated.cpu().numpy()
            # )
            # print(f"The value of FID is {fid_value}")

            # Convert FID to a tensor and move to device for backpropagation
            # fid_tensor = torch.tensor(fid_value, requires_grad=True, device=device)

            # Compute VAE loss
            vae_loss_value = vae_loss(recon_images, images, mu, logvar, alpha_value, beta_value)
            # print(f"the fid value is {fid_tensor}")
            # Define gamma to scale the contribution of FID
            # gamma = 10  # Adjust gamma as needed
            # combined_loss = vae_loss_value  + gamma * fid_tensor
            combined_loss = vae_loss_value 

            # Backpropagate combined loss
            combined_loss.backward()
            optimizer.step()

            # Write metrics after each batch
            write_metrics_json(epoch, batch_idx + 1, combined_loss.item(), metrics_file_path)

            # Display epoch and batch in human-readable format (1-based index)
            print(f"Epoch [{epoch + 1}/{args.epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {combined_loss.item()}")

        vae.reconstruction(dataloader = train_loader, num_images = 25, image_save_path = reconstruction_directory, latent_space_save_path = latent_space_directory , name = f"{epoch + 1}")

        sampled_images = vae.sample(num_of_images_in_generated_grid)
        plot_and_save_generated_images(sampled_images=sampled_images, path = generated_image_directory_path, name = epoch+1)
                    
        save_model(vae, base_path=directory,model_name=f"{epoch+1}", epoch= epoch+1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Variational Autoencoder.")
    parser.add_argument("--train_data", type=str, required=True, help="Path to training data.")
    parser.add_argument("--val_data", type=str, required=True, help="Path to validation data.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of worker processes for data loading.")
    parser.add_argument("--image_size", type=str, default="(224,224)", help="Image size for training as a tuple (width, height).")
    parser.add_argument("--latent_dim", type=int, default= 30, help="Dimensionality of the latent space.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for optimizer.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--beta_value", type = int , default = 1, help=" this is beta value : total loss = alpha * reconstruction loss + beta * KL divergence loss")
    parser.add_argument("--alpha_value", type = int, default=1, help="this is alpha value : total loss = alpha * reconstruction loss + beta * kl divergence loss")
    args = parser.parse_args()

    # Convert the image_size string to a tuple
    args.image_size = ast.literal_eval(args.image_size)

    main(args)
