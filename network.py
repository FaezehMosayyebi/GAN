import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super().__init__()

        self.disc = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid() # As the output will be 0 or 1 showing fake and real respectively.
        )

    def forward(self, x):
        return self.disc(x)
    
class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()

        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, img_dim),
            nn.Tanh() # As the input is a normalize image between -1 and 1
        )

    def forward(self, x):
        return self.gen(x)