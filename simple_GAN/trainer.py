import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transform
from torch.utils.tensorboard import SummaryWriter
from network import Generator, Discriminator 

def GAN_trainer(batch_size, num_epochs, lr):
    # Hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    "cuda" if torch.cuda.is_available() else "cpu"
    img_dim = 28 * 28 * 1  # mnist 784
    z_dim = 64

    # Network Initialization
    disc = Discriminator(img_dim=img_dim).to(device)
    gen = Generator(z_dim, img_dim).to(device)

    fixed_noise = torch.randn((batch_size, z_dim)).to(device)  # For testing network

    trans = transform.Compose(
        [transform.ToTensor(), transform.Normalize((0.5,), (0.5,))]
    )
    dataset = datasets.MNIST(root="dataset/", transform=trans, download=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    creterion = nn.BCELoss()
    disc_optim = optim.Adam(disc.parameters(), lr=lr)
    gen_optim = optim.Adam(gen.parameters(), lr=lr)

    writer_fake = SummaryWriter(f"run/GAN_MNIST/Fake")
    writer_real = SummaryWriter(f"run/GAN_MNIST/Real")
    step = 0

    for epoch in range(num_epochs):
        for batch_idx, (image, label) in enumerate(
            loader
        ):  # image = real image , label=no label is ab=available
            image = image.view(-1, 784).to(device)

            noise = torch.randn(batch_size, z_dim).to(device)
            fake = gen(noise)

            ### Train Discriminator: we want to max log(D(image))+log(1-D(G(z)))
            disc_real = disc(image).view(-1)  # we need to flat every thing
            disc_fake = disc(fake.detach()).view(
                -1
            )  # we want to use fake later so we detach it

            lossD_real = creterion(
                disc_real, torch.ones_like(disc_real)
            )  # see the BCELoss documentation
            lossD_fake = creterion(
                disc_fake, torch.zeros_like(disc_fake)
            )  # we dont want to delete fake data in backpropagation to prevent extra processes so we detach it
            lossD = (lossD_real + lossD_fake) / 2

            disc.zero_grad()
            lossD.backward()
            disc_optim.step()

            # Train Generator: we want to min log(1-D(G(z))) <--> max log(D(G(z)))

            output = disc(fake).view(-1)

            lossG = creterion(output, torch.ones_like(output))

            gen.zero_grad()
            lossG.backward()
            gen_optim.step()

            if batch_idx % 100 == 0:
                print(
                    f"Epoch [ {epoch}/{num_epochs}]\ "
                    f"Loss D: {lossD:.4f}, Loss G: {lossG:.4f}"
                )

                with torch.no_grad():
                    fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                    data = image.reshape(-1, 1, 28, 28)
                    img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                    img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                writer_fake.add_image(
                    "MNIST Fake Images", img_grid_fake, global_step=step
                )
                writer_real.add_image(
                    "MNIST Real Images", img_grid_real, global_step=step
                )

                step += 1
