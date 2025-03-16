import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import os
import sys
sys.path.append('..')
from cathedral_rl import cathedral_v0  
from vae import VAE, vae_loss
import torch.nn.functional as F

def collect_vae_training_data(num_games=1000, board_size=8, save_path="vae_training_data.pt"):

    env = cathedral_v0.env(
        board_size=board_size, 
        render_mode=None,
        per_move_rewards=True, 
        final_reward_score_difference=False
    )
    
    all_states = []
    
    for game in range(num_games):
        env.reset()
        
        # Print progress
        if game % 100 == 0:
            print(f"Collecting data from game {game}/{num_games}")
        
        # Play until game is over
        while env.agents:
            current_agent = env.agent_selection
            observation = env.observe(current_agent)
            
            # Get the board state
            state = observation["observation"]
            #print(f'State {state.shape}')
            

            all_states.append(state.copy())
            
            # Select random valid action
            action_mask = observation["action_mask"]
            valid_actions = np.where(action_mask == 1)[0]
            action = np.random.choice(valid_actions)
            
            # Take action
            env.step(action)
    
    env.close()
    
    print(f'All states length {len(all_states)}')
    states_array = np.array(all_states)
    states_array = np.transpose(states_array, (0, 3, 1, 2))
    all_states_tensor = torch.tensor(states_array, dtype=torch.float32)
    print(f"Collected {len(all_states)} board states")
    print(f"Original shape: {all_states[0].shape}, Transposed for CNN: {all_states_tensor[0].shape}")
    
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    torch.save(all_states_tensor, save_path)
    print(f"Data saved to {save_path}")
    
    return all_states_tensor

def train_vae_on_collected_data(data_path="vae_training_data.pt", batch_size=64, num_epochs=200, 
                              latent_dim=32, save_model_path="board_game_vae.pt",
                              patience=20, beta=1.0):
    
    # Load the collected data
    print(f"Loading data from {data_path}")
    states = torch.load(data_path)
    
    # Create dataset and dataloader
    train_size = int(0.9 * len(states))
    val_size = len(states) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        TensorDataset(states), [train_size, val_size]
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize the VAE
    input_channels = states.shape[1]
    print(f'Input channels {input_channels}')
    vae = VAE(input_channels=input_channels, latent_dim=latent_dim)
    
    # Training parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae.to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    
    # Early stopping setup
    best_val_loss = float('inf')
    best_epoch = 0
    epochs_no_improve = 0
    
    # Training loop
    print(f"Training VAE on {train_size} training states, {val_size} validation states for {num_epochs} epochs")
    for epoch in range(num_epochs):
        # Training phase
        vae.train()
        total_train_loss = 0
        for batch_idx, (data,) in enumerate(train_loader):
            data = data.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            reconstructed, mu, logvar = vae(data)
            
            # Calculate loss
            loss = vae_loss(reconstructed, data, mu, logvar, beta=beta)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Evaluation phase
        vae.eval()
        total_val_loss = 0
        with torch.no_grad():
            for val_data, in val_loader:
                val_data = val_data.to(device)
                val_recon, val_mu, val_logvar = vae(val_data)
                val_loss = vae_loss(val_recon, val_data, val_mu, val_logvar, beta=beta)
                total_val_loss += val_loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Check if this is the best model so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            # Save the best model
            torch.save(vae.state_dict(), save_model_path)
            print(f"New best model saved! Val Loss: {avg_val_loss:.4f}")
        else:
            epochs_no_improve += 1
        
        # Early stopping
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs. Best epoch was {best_epoch+1} with Val Loss: {best_val_loss:.4f}")
            break
    
    print(f"Training complete. Best model saved to {save_model_path} (epoch {best_epoch+1})")
    
    # Load the best model for return
    vae.load_state_dict(torch.load(save_model_path))
    return vae

def evaluate_vae(vae_path="board_game_vae.pt", test_data_path="vae_training_data.pt"):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load the VAE and some test data
    vae = VAE(input_channels=5, latent_dim=32)
    vae.load_state_dict(torch.load(vae_path, map_location=device))
    vae.to(device)
    vae.eval()
    
    test_data = torch.load(test_data_path, map_location=device)
    # Use a small batch for testing
    test_samples = test_data[:100].to(device)
    
    with torch.no_grad():
        # Reconstruct the test samples
        reconstructed, mu, logvar = vae(test_samples)
        
        # Calculate binary accuracy (how many pixels were correctly reconstructed)
        threshold = 0.5
        binary_accuracy = ((reconstructed > threshold) == (test_samples > threshold)).float().mean().item()
        
        # Calculate MSE
        mse = F.mse_loss(reconstructed, test_samples).item()
        
        print(f"Reconstruction Binary Accuracy: {binary_accuracy:.4f}")
        print(f"Reconstruction MSE: {mse:.4f}")

if __name__ == "__main__":

    #states = collect_vae_training_data(num_games=1000, save_path="vae_training_data.pt")
    
    #train_vae_on_collected_data(data_path="vae_training_data.pt", num_epochs=300)
    
    evaluate_vae()


def example_rl_usage(board_state, vae_path="board_game_vae.pt"):
    
    # Load the VAE
    vae = VAE(input_channels=5, latent_dim=32)
    vae.load_state_dict(torch.load(vae_path))
    vae.eval()
    
    # Convert a single board state to tensor
    if isinstance(board_state, np.ndarray):
        board_state = torch.tensor(board_state, dtype=torch.float32)
    
    # Add batch dimension if needed
    if len(board_state.shape) == 3:
        board_state = board_state.unsqueeze(0)
    
    # Encode the board state
    with torch.no_grad():
        mu, _ = vae.encoder(board_state)
    
    # The encoded state (mu) can now be used as input to your RL algorithm
    encoded_state = mu.squeeze(0).numpy()  # Shape: (latent_dim,)
    
    return encoded_state