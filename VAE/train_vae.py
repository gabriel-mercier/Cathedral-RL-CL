import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import os
import sys
import matplotlib.pyplot as plt
sys.path.append('..')
from cathedral_rl import cathedral_v0  
from vae import VAE, vae_loss
import torch.nn.functional as F
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "PPO")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "PPO", "models")))
from ppo_cnn import PPOCNN

def collect_vae_training_data(num_games=1000, board_size=8, save_path="vae_training_data.pt"):

    env = cathedral_v0.env(
        board_size=board_size, 
        render_mode=None,
        per_move_rewards=True, 
        final_reward_score_difference=False
    )
    env.reset()
    K_epochs = 20        
    eps_clip = 0.1     
    gamma = 0.97        

    lr_actor = 0.0005 
    lr_critic = 0.0002
    
    player_0 = "player_0"

    n_actions = env.action_space(player_0).n
    
    obs_shape = env.observe(player_0)["observation"].shape 
    
    ppo_agent = PPOCNN(
        obs_shape=obs_shape,
        action_dim=n_actions,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        gamma=gamma,
        K_epochs=K_epochs,
        eps_clip=eps_clip
    )
    
    model_path = 'cathedral_ppo_self_play_adversarial_final.pth'
    checkpoint = torch.load(model_path, weights_only=False)
    model_state_dict = checkpoint['model_state_dict']
    ppo_agent.policy.load_state_dict(model_state_dict)
    ppo_agent.policy_old.load_state_dict(model_state_dict)
    ppo_agent.policy.eval()
    
    all_states = []
    
    for game in range(num_games):
        env.reset()
        
        # Print progress
        if game % 100 == 0:
            print(f"Collecting data from game {game}/{num_games}")
            
        if game <(num_games // 2) : 
        
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
                
        else :
            
            while env.agents:
                current_agent = env.agent_selection
                observation = env.observe(current_agent)
                
                if current_agent == player_0:  # PPO agent's turn
                    state = observation["observation"]
                    action_mask = observation["action_mask"]
                    
                    # Use PPO to select action (deterministic for evaluation)
                    action = ppo_agent.select_action_evaluation(state, action_mask)
                    
                    all_states.append(state.copy())
                    
                else :
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
                              latent_dim=32, save_model_path="board_game_vae_v2_ppo.pt",
                              patience=30, beta=1.0):
    
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
    
    # Lists to store loss values for plotting
    train_losses = []
    val_losses = []
    
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
        train_losses.append(avg_train_loss)  # Store training loss
        
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
        val_losses.append(avg_val_loss)  # Store validation loss
        
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
    
    # Plot training and validation loss
    plot_training_loss(train_losses, val_losses, best_epoch, save_path=f"vae_training_loss_{latent_dim}dim.png")
    
    # Load the best model for return
    vae.load_state_dict(torch.load(save_model_path))
    return vae, train_losses, val_losses, best_epoch

def plot_training_loss(train_losses, val_losses, best_epoch=None, save_path="vae_training_loss.png"):
    """
    Plot training and validation losses over epochs
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        best_epoch: Index of the best epoch (for marking on the plot)
        save_path: Path to save the plot image
    """
    plt.figure(figsize=(12, 8))
    epochs = range(1, len(train_losses) + 1)
    
    # Plot losses
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2, alpha=0.8)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2, alpha=0.8)
    
    # Mark the best epoch if provided
    if best_epoch is not None:
        plt.axvline(x=best_epoch+1, color='green', linestyle='--', alpha=0.5, 
                   label=f'Best Model (Epoch {best_epoch+1})')
        plt.plot(best_epoch+1, val_losses[best_epoch], 'go', markersize=10)
    
    # Add titles and labels
    plt.title('VAE Training and Validation Loss', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # Format the y-axis
    plt.gca().yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:.2f}'))
    
    # Show the plot
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training loss plot saved to {save_path}")
    
    return save_path

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

    states = collect_vae_training_data(num_games=4000, save_path="vae_training_data_with_ppo_cnn.pt")
    
    train_vae_on_collected_data(data_path="vae_training_data_with_ppo_cnn.pt", num_epochs=300)
    
    evaluate_vae(vae_path="board_game_vae_v2_ppo.pt", test_data_path="vae_training_data_with_ppo_cnn.pt")