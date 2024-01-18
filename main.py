from trainer import *

if __name__ == "__main__":
    batch_size = 32
    num_epochs = 50
    lr = 3e-4

    GAN_trainer(batch_size, num_epochs, lr)
