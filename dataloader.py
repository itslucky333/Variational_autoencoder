import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader



def create_data_loaders(train_path, val_path, batch_size=32, num_workers=4, image_size=(224, 224)):
    # Define the transformation with the given image size
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        # Uncomment the line below if normalization is needed
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Create datasets
    train_dataset = datasets.ImageFolder(root=train_path, transform=transform)
    val_dataset = datasets.ImageFolder(root=val_path, transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    return train_loader, val_loader

def print_batch_info(data_loader, dataset, batch_idx=1):
    # Loop through the DataLoader for one batch
    for images, labels in data_loader:
        print("Image batch shape:", images.shape)   # Shape of image batch
        print("Label batch shape:", labels.shape)   # Shape of label batch

        # Print numeric label and its corresponding class name for the specified item in the batch
        label_idx = labels[batch_idx].item()                # Convert label tensor to a number
        class_name = dataset.classes[label_idx]             # Map label to class name

        print("Numeric label:", label_idx)
        print("Class name:", class_name)
        break  # Just to print one batch for demonstration
