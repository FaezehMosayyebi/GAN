import torch
import torch.nn as nn

def gradient_penalty(critic, real, fake, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape

    """
        epsilon in for making interpolated images.
    """
    epsilon = torch.ramd((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_image = real * epsilon + fake * (1 - epsilon)


    # Calculate Critic score
    mixed_score = critic(interpolated_image)

    # Calculating Lipschitz score (gradient penalty)

    gradient = torch.autograd.grad(
        input=interpolated_image,
        outputs=mixed_score,
        grad_outputs=torch.ones_like(mixed_score),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)

    return gradient_penalty