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

class ActorCritic(nn.Module):
    def __init__(self, obs_shape, action_dim):
        super(ActorCritic, self).__init__()
        
        # Calculate input size from observation shape
        self.state_dim = obs_shape[0] * obs_shape[1] * obs_shape[2]  # Flattened input
        self.action_dim = action_dim

        # Actor network - outputs action probabilities
        self.actor = nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
            # No softmax here - will apply it with action mask later
        )
        
        # Critic network - outputs state value
        self.critic = nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
    def forward(self):
        raise NotImplementedError
    
    def act(self, state, action_mask):
        # Flatten state for linear layers
        state_flat = state.reshape(-1)
        
        # Get action logits from actor
        action_logits = self.actor(state_flat)
        
        # Apply action mask (set invalid actions to negative infinity)
        action_mask_tensor = torch.tensor(action_mask, dtype=torch.bool)
        action_logits[~action_mask_tensor] = -1e8
        
        # Apply softmax to get action probabilities
        action_probs = torch.softmax(action_logits, dim=0)
        
        # Create categorical distribution
        dist = Categorical(action_probs)
        
        # Sample action
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        # Get state value from critic
        state_val = self.critic(state_flat)
        
        return action.detach(), action_logprob.detach(), state_val.detach()
    
    def evaluate(self, states, actions, action_masks):
        # Flatten states for linear layers
        batch_size = states.shape[0]
        states_flat = states.reshape(batch_size, -1)
        
        # Get action logits from actor for all states
        action_logits = self.actor(states_flat)
        
        # Apply action masks (set invalid actions to negative infinity)
        for i in range(batch_size):
            action_mask = action_masks[i]
            action_logits[i][~action_mask] = -1e8
        
        # Apply softmax to get action probabilities
        action_probs = torch.softmax(action_logits, dim=1)
        
        # Create categorical distributions
        dist = Categorical(action_probs)
        
        # Get log probabilities, entropies, and state values
        action_logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        state_values = self.critic(states_flat)
        
        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, obs_shape, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()
        
        self.policy = ActorCritic(obs_shape, action_dim).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])
        
        self.policy_old = ActorCritic(obs_shape, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
    
    def select_action(self, state, action_mask):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(device)
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
            #print(f'reward {reward}, is terminal {is_terminal}')
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards (if there are any)
        if len(rewards) > 0:
            rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
            #print(f' reward normalisation {rewards}')
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
        #print(f'Advantages {advantages.mean()}, rewards {rewards.mean()} old state values {old_state_values.mean()}')
        
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
            #print(f' surr1 {surr1.mean()}, surr2{surr2.mean()} mse loss * 0.5 {mse_loss.mean() * 0.5} dist entropy * 0.01 {0.01 * dist_entropy.mean()}')
            
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