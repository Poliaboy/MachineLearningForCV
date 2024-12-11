import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import os

# Set random seed for reproducibility
torch.manual_seed(42)

# Hyperparameters
latent_dim = 100
image_size = 64
channels = 3
batch_size = 128
num_epochs = 50
lr = 0.0002
beta1 = 0.5

# Device configuration
device = torch.device('mps' if torch.mps.is_available() else 'cpu')

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.main = nn.Sequential(
            # Input is latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # State size: 512 x 4 x 4
            
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # State size: 256 x 8 x 8
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # State size: 128 x 16 x 16
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # State size: 64 x 32 x 32
            
            nn.ConvTranspose2d(64, channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # Final state size: 3 x 64 x 64
        )

    def forward(self, x):
        x = x.view(x.size(0), latent_dim, 1, 1)
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.main = nn.Sequential(
            # Input is 3 x 64 x 64
            nn.Conv2d(channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: 64 x 32 x 32
            
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: 128 x 16 x 16
            
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: 256 x 8 x 8
            
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: 512 x 4 x 4
            
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # Final state size: 1 x 1 x 1
        )

    def forward(self, x):
        return self.main(x).view(-1, 1).squeeze(1)

def weights_init(m):
    """Custom weights initialization"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def prepare_celeba_dataset(data_root='./data/celeba'):
    """Prepare CelebA dataset"""
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    dataset = ImageFolder(root=data_root, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    return dataloader

def train_dcgan(dataloader):
    """Train the DCGAN"""
    # Initialize networks
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    
    # Initialize weights
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    
    # Loss function and optimizers
    criterion = nn.BCELoss()
    g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
    
    # Lists to store losses for plotting
    g_losses = []
    d_losses = []
    
    print("Starting Training...")
    
    for epoch in range(num_epochs):
        for i, (real_images, _) in enumerate(dataloader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            
            # Labels
            real_labels = torch.ones(batch_size).to(device)
            fake_labels = torch.zeros(batch_size).to(device)
            
            # Train Discriminator
            d_optimizer.zero_grad()
            output_real = discriminator(real_images)
            d_loss_real = criterion(output_real, real_labels)
            
            # Generate fake images
            noise = torch.randn(batch_size, latent_dim).to(device)
            fake_images = generator(noise)
            output_fake = discriminator(fake_images.detach())
            d_loss_fake = criterion(output_fake, fake_labels)
            
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()
            
            # Train Generator
            g_optimizer.zero_grad()
            output_fake = discriminator(fake_images)
            g_loss = criterion(output_fake, real_labels)
            g_loss.backward()
            g_optimizer.step()
            
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], '
                      f'd_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}')
        
        # Save losses for plotting
        g_losses.append(g_loss.item())
        d_losses.append(d_loss.item())
        
        # Generate and save sample images
        if (epoch + 1) % 5 == 0:
            save_sample_images(generator, epoch + 1)
    
    return generator, discriminator, g_losses, d_losses

def save_sample_images(generator, epoch, num_images=25):
    """Generate and save sample images"""
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(num_images, latent_dim).to(device)
        fake_images = generator(noise)
        fake_images = (fake_images + 1) / 2  # Denormalize
        fake_images = fake_images.cpu()

    # Plot the fake images
    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        img = fake_images[i].permute(1, 2, 0)
        ax.imshow(img)
        ax.axis('off')
    
    plt.suptitle(f'Generated Face Images at Epoch {epoch}')
    plt.savefig(f'dcgan_faces_epoch_{epoch}.png')
    plt.close()
    generator.train()

def plot_losses(g_losses, d_losses):
    """Plot the generator and discriminator losses"""
    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label='Generator Loss')
    plt.plot(d_losses, label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('DCGAN Training Losses')
    plt.legend()
    plt.savefig('dcgan_losses.png')
    plt.close()

if __name__ == "__main__":
    # Check if dataset exists
    if not os.path.exists('./data/celeba'):
        print("Please download the CelebA dataset and place it in ./data/celeba")
        print("Dataset should contain folders with images in the following structure:")
        print("./data/celeba/img_align_celeba/")
        exit()
    
    # Prepare dataset
    dataloader = prepare_celeba_dataset()
    
    # Train DCGAN
    generator, discriminator, g_losses, d_losses = train_dcgan(dataloader)
    
    # Plot losses
    plot_losses(g_losses, d_losses)
    
    # Generate final sample images
    save_sample_images(generator, num_epochs) 