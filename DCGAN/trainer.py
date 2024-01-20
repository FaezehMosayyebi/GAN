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
LEARNING_RATE = 2e-4
BATCH_SIZE = 128
IMAGE_SIZE = 64
IMG_CHANNELS = 1
Z_DIM = 100
FEATURE_GEN = 64
FEATURE_DISC = 64
NUM_EPOCHS = 50

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
disc = Discriminator(IMG_CHANNELS, FEATURE_DISC).to(device)

initialize_weights(gen)
initialize_weights(disc)

opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
criterion = nn.BCELoss()

fixed_noise = torch.randn((32, Z_DIM, 1, 1)).to(device)

writer_fake = SummaryWriter(f"logs/fake")
writer_real = SummaryWriter(f"logs/real")
step = 0

gen.train()
disc.train()

for epoch in range(NUM_EPOCHS):
    for batch_idx, (real, _) in enumerate(loader):  # (image, label)
        real = real.to(device)
        noise = torch.randn((BATCH_SIZE, Z_DIM, 1, 1)).to(device)
        fake = gen(noise)

        ### Train Discriminator: max log(D(real))+log(1-D(G(noise)))
        disc_fake = disc(fake.detach()).reshape(-1)
        disc_real = disc(real).reshape(-1)  # N x 1 x 1

        loss_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_real = criterion(disc_real, torch.ones_like(disc_real))

        disc_loss = (loss_fake + loss_real) / 2

        disc.zero_grad()
        disc_loss.backward()
        opt_disc.step

        ### Train Generator: min log(1-D(G(noise))) <--> max log(D(G(noise)))
        output = disc(fake).reshape(-1)

        gen_loss = criterion(output, torch.ones_like(output))

        gen.zero_grad()
        gen_loss.backward()
        opt_gen.step()

        if batch_idx % 100 == 0:
            print(
                f"Epoch [ {epoch}/{NUM_EPOCHS}]\ "
                f"Loss D: {disc_loss:.4f}, Loss G: {gen_loss:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)

            writer_fake.add_image("MNIST Fake Images", img_grid_fake, global_step=step)
            writer_real.add_image("MNIST Real Images", img_grid_real, global_step=step)

            step += 1
