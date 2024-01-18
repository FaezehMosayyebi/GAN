import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.datasets as dataset
import torchvision.transforms as transform
from network import Generator, Discriminator


def GAN_trainer(input_noie, dataset, batch_size, num_epochs, lr):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_dim = 28 * 28 * 1 # mnist 784
    z_dim = 64

    disc = Discriminator(img_dim=img_dim).to(device)
    gen = Generator(z_dim, img_dim).to(device)
    trans = transform.Compose(
        [transform.ToTensor, transform.Normalize((0.5,), (0.5,))]
    )


    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    creterion = nn.BCELoss()
    disc_optim = optim.Adam(disc.parameters(), lr=lr)
    gen_optim = optim.Adam(gen.parameters(), lr=lr)

    for epoch in range(num_epochs):

        for batch_idx, (image, label) in enumerate(DataLoader): # image = real image , label=no label is ab=available

            noise = torch.random((batch_size, z_dim)).to(device)
            fake = gen(noise)
            
            real = real.view(-1, 784).to(device)

            disc_real = disc(image).veiw(-1) # we need to flat every thing
            disc_fake = disc(fake.detach()).view(-1) # we want to use fake later so we detach it

            ### Train Discriminator: we want to max log(D(real))+log(1-D(G(z)))

            lossD_real = creterion(disc_real, torch.ones_like(real)) # see the BCELoss documentation
            lossD_fake = creterion(disc_fake, torch.zeros_like(fake)) # we dont want to delete fake data in backpropagation to prevent extra processes so we detach it
            lossD = (lossD_real + lossD_fake) / 2

            disc.zero_grad()
            lossD.bakward()
            disc_optim.step()

            # Train Generator: we want to min log(1-D(G(z))) <--> max log(D(G(z)))

            output = disc(fake).view(-1)

            lossG = creterion(output, torch.zeros_like(output))

            gen.zero_grad()
            lossG.backward()
            gen_optim.step()



            



    





