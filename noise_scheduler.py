import torch

def add_noise(tensor, timestep, total_timesteps, device="cuda"):
    """
    Add progressive noise to a tensor based on the current timestep.

    This function simulates the process of adding noise to data, where the 
    amount of noise increases as the timestep decreases. The noisy tensor 
    is clamped to ensure its values remain within a valid range.

    Parameters:
    ----------
    tensor : torch.Tensor
        The input tensor to which noise will be added.
        
    timestep : int
        The current timestep, determining the amount of noise to add.
        
    total_timesteps : int
        The total number of timesteps, used to scale the noise proportionally.
        
    device : str, optional
        The device to use for computation. Either "cuda" (GPU) or "cpu". Default is "cuda".

    Returns:
    -------
    noisy_tensor : torch.Tensor
        The tensor with added noise, clamped to the range [-1, 1].
        
    noise : torch.Tensor
        The noise that was added to the tensor.
    """

    how_much_noise = torch.sqrt(1 - ((total_timesteps - timestep) / total_timesteps)).to(device)

    noise = torch.randn_like(tensor) * how_much_noise 

    noisy_tensor = tensor + noise

    # Normalize the tensor values to (-1,1) range
    noisy_tensor = torch.clamp(noisy_tensor, -1, 1)

    return noisy_tensor, noise

