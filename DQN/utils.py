import numpy as np
from .model import DQN, ResNet, DCNNet
import torch
import random
import torch.nn.functional as F

def epsilon_by_episode(episode, epsilon_start, epsilon_final, epsilon_decay):
    return epsilon_final + (epsilon_start - epsilon_final) * np.exp(-episode / epsilon_decay)

def temperature_by_episode(episode, temperature_start, temperature_final, temperature_decay):
    return temperature_final + (temperature_start - temperature_final) * np.exp(-episode / temperature_decay)

def create_networks(obs_shape, n_actions, model, board_size, device):
    if model == 'DQN':
        policy_net = DQN(obs_shape, n_actions).to(device)
        target_net = DQN(obs_shape, n_actions).to(device)
        
    elif model == 'ResNet':
        policy_net = ResNet(obs_shape, n_actions, board_size).to(device)
        target_net = ResNet(obs_shape, n_actions, board_size).to(device)
    
    elif model == 'DCNNet':
        policy_net = DCNNet(obs_shape, n_actions, board_size).to(device)
        target_net = DCNNet(obs_shape, n_actions, board_size).to(device)
        
    target_net.load_state_dict(policy_net.state_dict())
    
    return policy_net, target_net

def select_action_dqn(model, obs, action_mask, legal_moves, device, method, epsilon, temperature, training=True):
    model.eval()
    with torch.no_grad():
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 10, 10, 5)
        q_values = model(obs_tensor).squeeze(0)  # (n_actions,)
        
        if method == 'eps_greedy':
            if not training:
                epsilon=0
            if random.random() < epsilon:
                first_action = random.choice(legal_moves)
                action = first_action
            else:
                first_action = torch.argmax(q_values).item()
                mask = torch.tensor(action_mask, dtype=torch.bool, device=device)
                q_values[~mask] = -1e8
                action = torch.argmax(q_values).item()
        
        elif method == 'boltzmann':
            first_action = torch.argmax(q_values).item()
            mask = torch.tensor(action_mask, dtype=torch.bool, device=device)
            q_values[~mask] = -1e8
            action = torch.argmax(q_values).item()
            if training:
                probabilities = F.softmax(q_values / temperature, dim=-1)
                action = torch.multinomial(probabilities, num_samples=1).item()
            if mask[first_action]:
                first_action = action
        
    model.train()
    return first_action, action



def compute_q_values(policy_net, target_net, next_states_tensor, next_action_masks_tensor, rewards_tensor, dones_tensor, gamma): # Double DQN
    with torch.no_grad():
        # print(f'next_states_tensor : {next_states_tensor.shape}')
        next_q_policy = policy_net(next_states_tensor)  # [batch_size, n_actions]
        next_q_policy[~next_action_masks_tensor] = -1e8
        next_actions = next_q_policy.argmax(dim=1, keepdim=True)  # [batch_size, 1]

        next_q_target = target_net(next_states_tensor)
        next_q_target[~next_action_masks_tensor] = -1e8
        
        next_q_values = next_q_target.gather(1, next_actions).squeeze(1)

        mask_sum = next_action_masks_tensor.sum(dim=1)
        next_q_values[mask_sum == 0] = 0.0

        # target_q_values = rewards_tensor + gamma * next_q_values * (1 - dones_tensor)
        target_q_values = rewards_tensor - gamma * next_q_values * (1 - dones_tensor)
        return target_q_values