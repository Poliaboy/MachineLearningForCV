import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)



# Device configuration
device = torch.device('mps' if torch.mps.is_available() else 'cpu')
image_size = 28 * 28
hidden_dim = 256
latent_dim = 100


# Generator Network
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            # First hidden layer
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            
            # Second hidden layer
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            
            # Third hidden layer
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.LeakyReLU(0.2),
            
            # Output layer
            nn.Linear(hidden_dim * 4, image_size),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 1, 28, 28)
        return img

# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            # First hidden layer
            nn.Linear(image_size, hidden_dim * 4),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            # Second hidden layer
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            # Third hidden layer
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            # Output layer
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

def train_gan():
    # Hyperparameters
    batch_size = 64
    num_epochs = 20
    lr = 0.0002
    beta1 = 0.5  # Beta1 parameter for Adam optimizer
    
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    mnist = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    dataloader = torch.utils.data.DataLoader(mnist, batch_size=batch_size, shuffle=True)

    # Initialize networks and optimizers
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    
    g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
    
    criterion = nn.BCELoss()

    # Lists to store losses for plotting
    g_losses = []
    d_losses = []
    
    # Training loop
    for epoch in range(num_epochs):
        for i, (real_images, _) in enumerate(dataloader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)

            # Create labels
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # Train Discriminator
            d_optimizer.zero_grad()
            outputs = discriminator(real_images)
            d_loss_real = criterion(outputs, real_labels)
            
            # Generate fake images
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_images = generator(z)
            outputs = discriminator(fake_images.detach())
            d_loss_fake = criterion(outputs, fake_labels)
            
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()

            # Train Generator
            g_optimizer.zero_grad()
            outputs = discriminator(fake_images)
            g_loss = criterion(outputs, real_labels)
            g_loss.backward()
            g_optimizer.step()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], '
                      f'd_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}')
                
        # Save losses for plotting
        g_losses.append(g_loss.item())
        d_losses.append(d_loss.item())
        
        # Generate and save images
        if (epoch + 1) % 2 == 0:
            save_generated_images(generator, epoch + 1)
    
    return generator, discriminator, g_losses, d_losses

def save_generated_images(generator, epoch, num_images=25):
    """Generate and save images at different epochs"""
    generator.eval()
    with torch.no_grad():
        z = torch.randn(num_images, latent_dim).to(device)
        generated_images = generator(z)
        generated_images = generated_images.cpu().numpy()

    # Plot the generated images
    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        ax.imshow(generated_images[i, 0, :, :], cmap='gray')
        ax.axis('off')
    
    plt.suptitle(f'Generated Images at Epoch {epoch}')
    plt.savefig(f'gan_images_epoch_{epoch}.png')
    plt.close()
    generator.train()

def plot_losses(g_losses, d_losses):
    """Plot the generator and discriminator losses"""
    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label='Generator Loss')
    plt.plot(d_losses, label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Generator and Discriminator Losses')
    plt.legend()
    plt.savefig('gan_losses.png')
    plt.close()

if __name__ == "__main__":
    # Train the GAN

    generator, discriminator, g_losses, d_losses = train_gan()
    
    # Plot the losses
    plot_losses(g_losses, d_losses) 