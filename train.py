import torch
from dataloader import create_data_loaders, print_batch_info
from model import VAE
import torch.optim as optim
from vae_loss import vae_loss

def main():
    # Specify paths to the training and validation datasets
    train_path = '../../masked_autoencoder/1/Training/'
    val_path = '../../masked_autoencoder/1/Validation/'
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_path=train_path,
        val_path=val_path,
        batch_size=32,
        num_workers=4,
        image_size=(224, 224)
    )
    
    print_batch_info(data_loader=train_loader, dataset=train_loader.dataset, batch_idx=0)

    latent_dim = 2
    learning_rate = 0.001
    epochs = 10
    vae = VAE(latent_dim=latent_dim)
    optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

    vae.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, _ in train_loader:
            optimizer.zero_grad()
            recon_images, mu, logvar = vae(images)
            loss = vae_loss(recon_images, images, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader.dataset)}")

    print("Training complete!")

if __name__ == "__main__":
    main()
