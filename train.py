import os
import argparse
import ast  # Import ast to safely evaluate the tuple string
import torch
import torch.optim as optim
from model import VAE # Assuming model.py contains your VAE class and loss function
from dataloader import create_data_loaders, print_batch_info
from create_output_directory import create_output_directory, create_parameter_file , write_metrics_json , plot_and_save_generated_images, create_directory
from vae_loss import vae_loss

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
    vae = VAE(latent_dim=args.latent_dim).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=args.learning_rate)

    # Print batch info
    # print_batch_info(data_loader=train_loader, batch_idx=0)

    # Training loop (simplified example)
    directory = create_output_directory()
    create_parameter_file(directory= directory,args= args,device=device)
    metrics_file_path = os.path.join(directory, "metrics.json")
    num_of_images_in_generated_grid = 64
    generated_image_directory_path = create_directory(directory, "generated_images")

    beta_value = args.beta_value
    alpha_value = args.alpha_value
    for epoch in range(args.epochs):
        vae.train()
        for batch_idx, (images, labels) in enumerate(train_loader):
            # Use the loop variable directly for the epoch and batch number
            images = images.to(device)
            optimizer.zero_grad()

            recon_images, mu, logvar = vae(images)
            loss = vae_loss(recon_images, images, mu, logvar,alpha_value, beta_value)
            loss.backward()
            optimizer.step()

            # Write metrics after each batch
            write_metrics_json(epoch, batch_idx + 1, loss.item(), metrics_file_path)
            
            # Display epoch and batch in human-readable format (1-based index)
            print(f"Epoch [{epoch + 1}/{args.epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item()}")

        sampled_images = vae.sample(num_of_images_in_generated_grid)
        plot_and_save_generated_images(sampled_images=sampled_images, path = generated_image_directory_path, name = epoch+1)
            




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Variational Autoencoder.")
    parser.add_argument("--train_data", type=str, required=True, help="Path to training data.")
    parser.add_argument("--val_data", type=str, required=True, help="Path to validation data.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of worker processes for data loading.")
    parser.add_argument("--image_size", type=str, default="(224,224)", help="Image size for training as a tuple (width, height).")
    parser.add_argument("--latent_dim", type=int, default=2, help="Dimensionality of the latent space.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for optimizer.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--beta_value", type = int , default = 1, help=" this is beta value : total loss = alpha * reconstruction loss + beta * KL divergence loss")
    parser.add_argument("--alpha_value", type = int, default=1, help="this is alpha value : total loss = alpha * reconstruction loss + beta * kl divergence loss")
    args = parser.parse_args()

    # Convert the image_size string to a tuple
    args.image_size = ast.literal_eval(args.image_size)

    main(args)
