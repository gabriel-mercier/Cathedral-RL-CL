import torch
import torch.nn as nn
from torch.distributions import Categorical
import sys
sys.path.append('..')
from ppo_utils import RolloutBuffer

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


# Modified PPO class to use the CNN model
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
    
    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards (if there are any)
        if len(rewards) > 0:
            rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
            if rewards.std() > 0:
                rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        else:
            return  # Nothing to update if buffer is empty
        
        # Convert lists to tensors
        old_states = torch.stack(self.buffer.states).detach().to(device)
        old_actions = torch.stack(self.buffer.actions).detach().to(device)
        old_logprobs = torch.stack(self.buffer.logprobs).detach().to(device)
        old_state_values = torch.stack(self.buffer.state_values).detach().squeeze().to(device)
        old_action_masks = torch.stack(self.buffer.action_masks).detach().to(device)
        
        # Calculate advantages
        advantages = rewards - old_state_values
        
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions, old_action_masks)
            
            # Match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            
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
        
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # Clear buffer
        self.buffer.clear()
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
    
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))