import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import torch

def save_model(model, base_path, epoch, model_name="vae_model"):
    # Create the model directory
    model_dir = os.path.join(base_path, "model")
    os.makedirs(model_dir, exist_ok=True)
    
    # Create the epoch directory inside the model directory
    epoch_dir = os.path.join(model_dir, f"epoch_{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)
    
    # Save the model components to the epoch directory
    torch.save(model.encoder.state_dict(), os.path.join(epoch_dir, f"{model_name}_encoder.pth"))
    torch.save(model.decoder.state_dict(), os.path.join(epoch_dir, f"{model_name}_decoder.pth"))
    torch.save(model.state_dict(), os.path.join(epoch_dir, f"{model_name}.pth"))
    
    print(f"Models saved in directory: {epoch_dir}")


def create_output_directory():
    if not os.path.exists("output"):
        print(" Output directory does not exist. Creating a directory named output")
        os.makedirs("output", exist_ok=True)
    current_time = datetime.now()
    time_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    directory_name = f"result+{time_str}"
    directory_name = os.path.join("output",directory_name)
    os.makedirs(directory_name,exist_ok = True)
    print(f"Directory created: {directory_name}")
    return directory_name

def create_directory(path, name):
    directory_path = os.path.join(path,name)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    return directory_path

def create_parameter_file(directory,args,device):
    file_name = "parameter"
    file_path = os.path.join(directory, file_name)
    with open(file_path, 'w') as file:
     file.write("Argument Values:\n")
     file.write(f"Trained on {device}\n")
     file.write(f"Train Data Path: {args.train_data}\n")
     file.write(f"Validation Data Path: {args.val_data}\n")
     file.write(f"Batch Size: {args.batch_size}\n")
     file.write(f"Number of Workers: {args.num_workers}\n")
     file.write(f"Image Size: {args.image_size}\n")
     file.write(f"Latent Dimension: {args.latent_dim}\n")
     file.write(f"Learning Rate: {args.learning_rate}\n")
     file.write(f"Epochs: {args.epochs}\n")
     file.write(f"beta_value : {args.beta_value} \n")
     file.write(f"Alpha value : {args.alpha_value} \n")

def write_metrics_json(epoch, batch, loss, file_path):
    metrics = {
        "epoch": epoch + 1,       # increasing 1 because it starts from zero    
        "batch": batch,            # not increasing one because value is already added in tran.py
        "loss": loss              
    }
    
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
    else:
        data = []
        
    data.append(metrics)
    
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


   
def plot_and_save_generated_images(sampled_images, path, name):
    fig, axes = plt.subplots(8, 8, figsize=(10, 10))  # Create a grid of 8x8
    axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration

    # Iterate through the axes and images
    for ax, img in zip(axes, sampled_images):
        img = img.permute(1, 2, 0)  # Change the shape from (C, H, W) to (H, W, C)
        ax.imshow(img.detach().cpu().numpy())  # Convert to numpy for plotting
        ax.axis('off')  # Turn off axis labels

    # Adjust layout and show the plot
    plt.tight_layout()
    save_path = os.path.join(path, f"{name}.jpg")
    plt.savefig(save_path)
    