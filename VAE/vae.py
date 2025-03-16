import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Define the Encoder
class Encoder(nn.Module):
    def __init__(self, input_channels, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(64 * 2 * 2, 128)  # Flattening the output from conv layers
        self.fc2_mu = nn.Linear(128, latent_dim)
        self.fc2_logvar = nn.Linear(128, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.reshape(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        mu = self.fc2_mu(x)
        logvar = self.fc2_logvar(x)
        return mu, logvar

# Define the Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim, output_channels):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, 64 * 2 * 2)
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, output_channels, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, z):
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        z = z.reshape(z.size(0), 64, 2, 2)  # Reshape to match conv layers
        z = F.relu(self.deconv1(z))
        z = torch.sigmoid(self.deconv2(z))  # Output layer
        return z

# Reparameterization Trick to sample from the latent space
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

# Define the VAE Model
class VAE(nn.Module):
    def __init__(self, input_channels=5, latent_dim=32):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_channels, latent_dim)
        self.decoder = Decoder(latent_dim, input_channels)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = reparameterize(mu, logvar)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mu, logvar

# Define the loss function (VAE loss = Reconstruction Loss + KL Divergence)
def vae_loss(x_reconstructed, x, mu, logvar, beta=1.0):
    # Reconstruction loss (binary cross-entropy for binary board states)
    BCE = F.binary_cross_entropy(x_reconstructed.view(-1, 5*8*8), x.view(-1, 5*8*8), reduction='sum')
    
    # KL divergence loss
    # KL Divergence = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = 0.5 * torch.sum(mu.pow(2) + torch.exp(logvar) - logvar - 1)
    
    # Normalize by batch size
    batch_size = x.size(0)
    return (BCE + beta * KLD) / batch_size
