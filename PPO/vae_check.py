import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "VAE")))
from vae import VAE
import torch

vae_path = 'board_game_vae.pt'
# Load the VAE (provide any latent_dim, it will be overwritten by the loaded weights)
vae = VAE(input_channels=5, latent_dim=32)
vae.load_state_dict(torch.load(vae_path, map_location='cpu'))

# Check the actual latent dimension
latent_dim = vae.decoder.fc1.in_features
print(f"Encoder output dimension: {vae.encoder.fc2_mu.out_features}")
print(f"Decoder input dimension: {vae.decoder.fc1.in_features}")


