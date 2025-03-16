import torch
import torch.nn as nn
from torch.distributions import Categorical
import sys
sys.path.append('..')
from ppo_utils import RolloutBuffer
import numpy as np

device = torch.device('cpu')

if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()

class ActorCriticCNN(nn.Module):
    def __init__(self, obs_shape, action_dim):
        super(ActorCriticCNN, self).__init__()
        
        # Extract observation dimensions
        self.channels = obs_shape[2]
        self.height = obs_shape[1]   
        self.width = self.height 
        self.action_dim = action_dim
        print(f'Loading PPO CNN with {self.channels}, height {self.height}, width {self.width}')
        # Shared convolutional layers
        self.shared_cnn = nn.Sequential(
            nn.Conv2d(self.channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Calculate the size after convolutions (unchanged if proper padding is used)
        conv_output_size = 64 * self.height * self.width
        
        # Actor head (policy network)
        self.actor = nn.Sequential(
            nn.Linear(conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, self.action_dim)
            # No softmax - will apply with action mask later
        )
        
        # Critic head (value network)
        self.critic = nn.Sequential(
            nn.Linear(conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
    
    def _get_conv_features(self, state):
        # Process the state through the shared CNN layers
        x = self.shared_cnn(state)
        return x.view(x.size(0), -1) if x.dim() > 2 else x.view(1, -1)
    
    def forward(self):
        raise NotImplementedError
    
    def act(self, state, action_mask):
        # Ensure state has batch dimension and channel-first format
        if state.dim() == 3:  # If no batch dimension
            state = state.unsqueeze(0)  # Add batch dimension
        
        # Get shared features
        features = self._get_conv_features(state)
        features = features.squeeze(0)  # Remove batch dimension for single sample
        
        # Get action logits
        action_logits = self.actor(features)
        
        # Apply action mask
        action_mask_tensor = torch.tensor(action_mask, dtype=torch.bool)
        action_logits[~action_mask_tensor] = -1e8
        
        # Apply softmax
        action_probs = torch.softmax(action_logits, dim=0)
        
        # Create distribution
        dist = Categorical(action_probs)
        
        # Sample action
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        # Get state value
        state_val = self.critic(features)
        
        return action.detach(), action_logprob.detach(), state_val.detach()
    
    def evaluate(self, states, actions, action_masks):
        # Ensure proper dimensions for CNN input
        batch_size = states.shape[0]
        
        # Get shared features
        features = self._get_conv_features(states)
        
        # Get action logits for all states
        action_logits = self.actor(features)
        
        # Apply action masks
        for i in range(batch_size):
            action_mask = action_masks[i]
            action_logits[i][~action_mask] = -1e8
        
        # Apply softmax
        action_probs = torch.softmax(action_logits, dim=1)
        
        # Create distributions
        dist = Categorical(action_probs)
        
        # Get log probabilities, entropies, and state values
        action_logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        state_values = self.critic(features)
        
        return action_logprobs, state_values, dist_entropy


class PPOCNN:
    def __init__(self, obs_shape, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()
        
        self.policy = ActorCriticCNN(obs_shape, action_dim).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic},
            {'params': self.policy.shared_cnn.parameters(), 'lr': lr_actor}
        ])
        
        self.policy_old = ActorCriticCNN(obs_shape, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
    
    def select_action(self, state, action_mask):
        with torch.no_grad():
            # Convert state to proper format (C, H, W)
            state_tensor = torch.FloatTensor(state).to(device)
            # Make sure we have channels first format
            if state_tensor.dim() == 3 and state_tensor.shape[0] != 5:
                # Assuming the first dimension should be channels, but it's not
                state_tensor = state_tensor.permute(2, 0, 1)
                
            action_mask_tensor = torch.BoolTensor(action_mask).to(device)
            action, action_logprob, state_val = self.policy_old.act(state_tensor, action_mask_tensor)
        
        self.buffer.states.append(state_tensor)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)
        self.buffer.action_masks.append(action_mask_tensor)
        
        return action.item()
    
    def select_action_episodic(self, state, action_mask, episode_buffer):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(device)
            # Make sure we have channels first format
            if state_tensor.dim() == 3 and state_tensor.shape[0] != 5:
                state_tensor = state_tensor.permute(2, 0, 1)
                
            action_mask_tensor = torch.BoolTensor(action_mask).to(device)
            action, action_logprob, state_val = self.policy_old.act(state_tensor, action_mask_tensor)
        
        # Store experiences in the episode buffer instead of the main buffer
        episode_buffer.states.append(state_tensor)
        episode_buffer.actions.append(action)
        episode_buffer.logprobs.append(action_logprob)
        episode_buffer.state_values.append(state_val)
        episode_buffer.action_masks.append(action_mask_tensor)
        
        return action.item()
    
    def select_action_evaluation(self, state, action_mask):
        """Version of select_action that doesn't modify the buffer - use for evaluation only"""
        with torch.no_grad():
            # Convert state to proper format (C, H, W)
            state_tensor = torch.FloatTensor(state).to(device)
            # Make sure we have channels first format
            if state_tensor.dim() == 3 and state_tensor.shape[0] != 5:
                # Assuming the first dimension should be channels, but it's not
                state_tensor = state_tensor.permute(2, 0, 1)
                
            action_mask_tensor = torch.BoolTensor(action_mask).to(device)
            action, _, _ = self.policy_old.act(state_tensor, action_mask_tensor)
        
        # Don't store anything in the buffer
        return action.item()
    
    def add_episode_buffer(self, episode_buffer):
        # Transfer experiences from episode buffer to main buffer
        self.buffer.states.extend(episode_buffer.states)
        self.buffer.actions.extend(episode_buffer.actions)
        self.buffer.logprobs.extend(episode_buffer.logprobs)
        self.buffer.state_values.extend(episode_buffer.state_values)
        self.buffer.action_masks.extend(episode_buffer.action_masks)
        self.buffer.rewards.extend(episode_buffer.rewards)
        self.buffer.is_terminals.extend(episode_buffer.is_terminals)
        
    def compute_gae(self, rewards, values, masks, gamma=0.99, lambda_=0.95):
        """Compute Generalized Advantage Estimation"""
        gae = 0
        advantages = []
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0  # Terminal state has value 0
            else:
                next_value = values[i + 1]
            
            delta = rewards[i] + gamma * next_value * masks[i] - values[i]
            gae = delta + gamma * lambda_ * masks[i] * gae
            advantages.insert(0, gae)
            
        return torch.tensor(advantages, dtype=torch.float32).to(device)
    
    def update(self):
        # Convert buffer to tensors
        states = torch.stack(self.buffer.states).detach().to(device)
        actions = torch.stack(self.buffer.actions).detach().to(device)
        old_logprobs = torch.stack(self.buffer.logprobs).detach().to(device)
        old_values = torch.stack(self.buffer.state_values).squeeze().detach().to(device)
        rewards = torch.tensor(self.buffer.rewards, dtype=torch.float32).to(device)
        masks = 1 - torch.tensor(self.buffer.is_terminals, dtype=torch.float32).to(device)
        
        # Compute returns and advantages using GAE
        advantages = self.compute_gae(
            rewards.cpu().numpy(), 
            old_values.cpu().numpy(), 
            masks.cpu().numpy(),
            self.gamma,
            0.95  # Lambda parameter for GAE
        )
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        surr1_array = []
        surr2_array = []
        values_array = []
        mse_loss_array = []
        total_loss_array = []
        ratios_array = []
        ratio_std_array = []
        
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(states, actions, torch.stack(self.buffer.action_masks).detach().to(device))
            
            # Match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            mean_ratio = torch.mean(ratios).item()
            std_ratio = torch.std(ratios).item()
            ratios_array.append(mean_ratio)
            ratio_std_array.append(std_ratio)
            
            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            mse_loss = self.MseLoss(state_values, rewards)
            
            # Final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * mse_loss - 0.01 * dist_entropy
            
            # Take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
            surr1_array.append(surr1.mean())
            surr2_array.append(surr2.mean())
            values_array.append(state_values.mean())
            mse_loss_array.append(mse_loss.mean())
            total_loss_array.append(loss.mean())
        
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # Clear buffer
        self.buffer.clear()
        
        return (
            np.mean([x.detach().cpu().numpy() for x in surr1_array]),  
            np.mean([x.detach().cpu().numpy() for x in surr2_array]),
            np.mean([x.detach().cpu().numpy() for x in values_array]),
            np.mean([x.detach().cpu().numpy() for x in mse_loss_array]),
            np.mean([x.detach().cpu().numpy() for x in total_loss_array]),
            np.mean([x for x in ratios_array]),
            np.mean([x for x in ratio_std_array])
        )
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
    
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))