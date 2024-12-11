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
    
def train_gan_with_noise(noise_type='gaussian'):
    """Train GAN with specified noise distribution"""
    # Hyperparameters
    latent_dim = 100
    batch_size = 64
    num_epochs = 20
    lr = 0.0002
    beta1 = 0.5

    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    mnist = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    dataloader = torch.utils.data.DataLoader(mnist, batch_size=batch_size, shuffle=True)

    # Initialize networks
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    
    # Optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
    
    criterion = nn.BCELoss()

    def generate_noise(batch_size, latent_dim):
        if noise_type == 'uniform':
            # Uniform distribution [-1, 1]
            return 2 * torch.rand(batch_size, latent_dim).to(device) - 1
        else:  # gaussian
            # Standard normal distribution
            return torch.randn(batch_size, latent_dim).to(device)

    # Training loop
    for epoch in range(num_epochs):
        for i, (real_images, _) in enumerate(dataloader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)

            # Labels
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # Train Discriminator
            d_optimizer.zero_grad()
            outputs = discriminator(real_images)
            d_loss_real = criterion(outputs, real_labels)
            
            # Generate fake images with specified noise distribution
            z = generate_noise(batch_size, latent_dim)
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
                print(f'[{noise_type.capitalize()}] Epoch [{epoch+1}/{num_epochs}], '
                      f'Step [{i+1}/{len(dataloader)}], '
                      f'd_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}')

        # Generate and save sample images every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_sample_images(generator, epoch + 1, noise_type, generate_noise)

    return generator

def save_sample_images(generator, epoch, noise_type, generate_noise, num_images=25):
    """Generate and save sample images"""
    generator.eval()
    with torch.no_grad():
        z = generate_noise(num_images, 100)  # Generate noise using the specified distribution
        generated_images = generator(z)
        generated_images = generated_images.cpu().numpy()

    # Plot the generated images
    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        ax.imshow(generated_images[i, 0, :, :], cmap='gray')
        ax.axis('off')
    
    plt.suptitle(f'Generated Images at Epoch {epoch}\nNoise Distribution: {noise_type.capitalize()}')
    plt.savefig(f'gan_images_{noise_type}_epoch_{epoch}.png')
    plt.close()
    generator.train()
    
def generate_comparison(gaussian_generator, uniform_generator, num_samples=10):
    """Generate side-by-side comparison of images from both distributions"""
    gaussian_generator.eval()
    uniform_generator.eval()
    
    with torch.no_grad():
        # Generate images with Gaussian noise
        z_gaussian = torch.randn(num_samples, 100).to(device)
        gaussian_images = gaussian_generator(z_gaussian).cpu().numpy()
        
        # Generate images with Uniform noise
        z_uniform = (2 * torch.rand(num_samples, 100) - 1).to(device)
        uniform_images = uniform_generator(z_uniform).cpu().numpy()

    # Create side-by-side comparison
    fig, axes = plt.subplots(num_samples, 2, figsize=(8, 20))
    plt.suptitle('Comparison of Noise Distributions\nLeft: Gaussian, Right: Uniform')
    
    for i in range(num_samples):
        # Plot Gaussian-generated image
        axes[i, 0].imshow(gaussian_images[i, 0], cmap='gray')
        axes[i, 0].axis('off')
        
        # Plot Uniform-generated image
        axes[i, 1].imshow(uniform_images[i, 0], cmap='gray')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('noise_distribution_comparison.png')
    plt.close()
    
def compare_noise_distributions():
    """Train GANs with different noise distributions and compare results"""
    # Train with Gaussian noise
    print("Training GAN with Gaussian noise...")
    gaussian_generator = train_gan_with_noise('gaussian')
    
    # Train with Uniform noise
    print("\nTraining GAN with Uniform noise...")
    uniform_generator = train_gan_with_noise('uniform')
    
    # Generate final comparison
    generate_comparison(gaussian_generator, uniform_generator)



if __name__ == "__main__":
    compare_noise_distributions() 