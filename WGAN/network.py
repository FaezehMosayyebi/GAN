# Implementation of WGAN paper

import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, img_channels, features_d):
        super(Discriminator, self).__init__()

        self.disc = nn.Sequential(

            # Input: N x img_channels x 64 x 64
            nn.Conv2d(
                img_channels, features_d, kernel_size=4, stride=2, padding=1
            ), # 32 x 32
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d*2, 4, 2, 1), # 16 x 16
            self._block(features_d*2, features_d*4, 4, 2, 1), # 8 x 8
            self._block(features_d*4, features_d*8, 4, 2, 1), # 4 x 4
            nn.Conv2d(features_d*8, 1, kernel_size=4, stride=2, padding=0), # 1 x 1
            nn.Sigmoid()
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )
    
    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, z_dim, img_channels, features_g):
        super(Generator, self).__init__()
        
        self.gen = nn.Sequential(
            # Input: N x z_dim x 1 x 1
            self._block(z_dim, features_g*16, 4, 1, 0), # N x f_g*16 x 4 x 4
            self._block(features_g*16, features_g*8, 4, 2, 1), # 8 x 8
            self._block(features_g*8, features_g*4, 4, 2, 1), # 16 x 16
            self._block(features_g*4, features_g*2, 4, 2, 1), # 32 x 32
            nn.ConvTranspose2d(features_g*2, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh() # [-1, 1]
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.gen(x)
    
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

if __name__ == '__main__':
    
    # Test
    N, in_channel, H, W = 8, 3, 64, 64
    z_dim = 100
    img = torch.randn((N, in_channel, H, W))
    disc = Discriminator(in_channel, 8)
    initialize_weights(disc)
    assert disc(img).shape == (N, 1, 1, 1)

    gen = Generator(z_dim, in_channel, 8)
    initialize_weights(gen)
    z = torch.rand((N, z_dim, 1, 1))
    assert gen(z).shape == (N, in_channel, H, W)

    print('Success')