from noise_scheduler import add_noise
from model import DiffusionModel
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import csv


def save_losses_to_csv(losses, file_name="training_losses.csv"):
    """
        Save the list of losses to a CSV file.

        Args:
            losses (list of dict): Each dict contains 'epoch', 'batch', and 'loss'.
            file_name (str): Name of the CSV file.
    """

    with open(file_name, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "batch", "loss"])
        writer.writeheader()
        writer.writerows(losses)
    print(f"Training losses saved to {file_name}")



def train_sample(model, data, total_timesteps, device="cuda", verbose=False):
    """
        Perform training for a single batch of data over multiple timesteps.

        This function iteratively adds noise to the input data and trains the model
        to predict the noise at each timestep. The loss for each timestep is 
        computed, and the average loss for the batch is returned.

        Parameters:
        ----------
        model : torch.nn.Module
            The u-net model to be trained.
            
        data : torch.Tensor
            A batch of input data (e.g., images or features) to be used for training.
            
        total_timesteps : int
            The total number of timesteps for training. The model will iterate 
            through all timesteps in reverse order (from `total_timesteps - 1` to `0`).
            
        device : str, optional
            The device to use for computation. Either "cuda" (GPU) or "cpu". Default is "cuda".
            
        verbose : bool, optional
            If True, prints debugging information for each timestep and the average 
            loss after processing the batch. Default is False.

        Returns:
        -------
        model : torch.nn.Module
            The updated u-nrt model after processing the batch.
            
        avg_loss : float
            The average loss across all timesteps for the batch.
    """

    losses = []
    for t in range(total_timesteps - 1, -1, -1):
        
        # Single scalar tensor for timestep
        t = torch.tensor([t], device=device)  
        total_timesteps_tensor = torch.tensor([total_timesteps], device=device)

        # Add noise to the entire batch(noisy tensor)
        noisy_tensor, noise = add_noise(data, t, total_timesteps_tensor, device)

        # Debug: Check noisy tensor shape
        # print(f"Step {t.item()} - Noisy tensor shape: {noisy_tensor.shape}, Noise shape: {noise.shape}")

        # Training step
        loss = model.training_step(noisy_tensor, noise, t)
        losses.append(loss)
    
    avg_loss = sum(losses) / len(losses)
    if verbose:
        print(f"Average loss: {avg_loss:.4f}")
    return model, avg_loss



def train_model(model, dataloader, total_timesteps, epochs=1000, device="cuda", verbose=True):
    """
        Train a u-net model over multiple epochs using a provided dataset.

        This function iterates over the dataset in batches, trains the model, logs losses,
        and periodically saves intermediate models. After training, it exports the
        training losses to a CSV file.

        Parameters:
        ----------
        model : torch.nn.Module
            The u-net model to be trained.
            
        dataloader : torch.utils.data.DataLoader
            A DataLoader that provides batches of data for training.
            
        total_timesteps : int
            The total number of timeste
            
        epochs : int, optional
            The number of epochs to train the model. Default is 1000.
            
        device : str, optional
            The device to use for computation. Either "cuda" (GPU) or "cpu". Default is "cuda".
            
        verbose : bool, optional
            If True, prints batch-wise loss during training. Default is True.

        Returns:
        -------
        None
            The function does not return anything but saves intermediate models and training
            losses to a CSV file.
    """
        
    model.to(device)

    for epoch in range(epochs):
        training_losses = []
        for i, (data, _) in enumerate(dataloader):
            data = data.to(device)

            # Train the model and get the loss
            model, avg_loss = train_sample(model, data, total_timesteps, device, verbose=verbose)
            
            # Log the loss
            training_losses.append({"epoch": epoch, "batch": i, "loss": avg_loss})
            
            if verbose:
                print(f"Epoch {epoch} - Batch {i} - Loss: {avg_loss:.4f}")
            
        # Save intermediate models
        if epoch % 50 == 0:
            model.save(f"models/model_epoch{epoch}.pth")

        # Save all losses to a CSV file after training
        save_losses_to_csv(training_losses, file_name="training_losses.csv")



if __name__ == '__main__':
    dataset_path = "/root/font_recognition/Fardin/stable_diffusion_from_scratch/Catfusion/datset/final"

    # Define transformations to apply to the dataset
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  
        transforms.ToTensor(),             
        transforms.Normalize((0.5,), (0.5,))  
    ])

    # Load the dataset and create a dataloader to iterate over the images in batches
    dataset = datasets.ImageFolder(root=dataset_path,transform=transform)
    dataloader = DataLoader(dataset, batch_size=1000, shuffle=True)

    # Create model
    model = DiffusionModel(lr=0.01)
    
    total_timesteps = 100

    # Train the model
    train_model(model, dataloader, total_timesteps, epochs=1000, device="cuda")