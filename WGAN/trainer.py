# Implementation of WGAN paper

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transform
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from network import Discriminator, Generator, initialize_weights

device = torch.device("cuda" if torch.cuda.is_available else "cpu")
LEARNING_RATE = 5e-5
BATCH_SIZE = 64
IMAGE_SIZE = 64
IMG_CHANNELS = 1
Z_DIM = 100
FEATURE_GEN = 64
FEATURE_CRITIC = 64
NUM_EPOCHS = 50
CRITIC_ITERATIONS = 5
WEIGHT_CLIP = 0.01

transforms = transform.Compose(
    [
        transform.Resize(IMAGE_SIZE),
        transform.ToTensor(),
        transform.Normalize(
            [0.5 for _ in range(IMG_CHANNELS)], [0.5 for _ in range(IMG_CHANNELS)]
        ),
    ]
)

dataset = datasets.MNIST("dataset/", transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
gen = Generator(Z_DIM, IMG_CHANNELS, FEATURE_GEN).to(device)
critic = Discriminator(IMG_CHANNELS, FEATURE_CRITIC).to(device)

initialize_weights(gen)
initialize_weights(critic)

opt_gen = optim.RMSprop(gen.parameters(), lr=LEARNING_RATE)
opt_critic = optim.RMSprop(critic.parameters(), lr=LEARNING_RATE)

fixed_noise = torch.randn((32, Z_DIM, 1, 1)).to(device)

writer_fake = SummaryWriter(f"logs/fake")
writer_real = SummaryWriter(f"logs/real")
step = 0

gen.train()
critic.train()

for epoch in range(NUM_EPOCHS):
    for batch_idx, (real, _) in enumerate(loader):  # (image, label)
        real = real.to(device)

        ### Train Critic
        for _ in range(CRITIC_ITERATIONS):
            noise = torch.randn((BATCH_SIZE, Z_DIM, 1, 1)).to(device)
            fake = gen(noise)

            critic_fake = critic(fake.detach()).reshape(-1)
            critic_real = critic(real).reshape(-1)  # N x 1 x 1

            # We want to maximise E(critic(real) - E(critic(generated))
            critic_loss = -(torch.mean(critic_real) - torch.mean(critic_fake))

            critic.zero_grad()
            critic_loss.backward()
            opt_critic.step

            for p in critic.parameters():
                p.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)

        ### Train Generator: minimize E(critic(real) - E(critic(generated))
        output = critic(fake).reshape(-1)

        gen_loss = -torch.mean(output)

        gen.zero_grad()
        gen_loss.backward()
        opt_gen.step()

        if batch_idx % 100 == 0:
            print(
                f"Epoch [ {epoch}/{NUM_EPOCHS}]\ "
                f"Loss D: {critic_loss:.4f}, Loss G: {gen_loss:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)

            writer_fake.add_image("MNIST Fake Images", img_grid_fake, global_step=step)
            writer_real.add_image("MNIST Real Images", img_grid_real, global_step=step)

            step += 1
