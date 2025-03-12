from cathedral_rl import cathedral_v0 
from cathedral_rl.game.board import Board 
 
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import deque
from tqdm import tqdm 

from ReplayBuffer import ReplayBuffer, PrioritizedReplayBuffer
from model import DCNNet, DQN, ResNet


num_episodes = 10000
buffer_capacity = 40000
treshold_penalize_illegal = 0
penalize_value = 10
batch_size = 64
gamma = 0.95
learning_rate = 1e-3
board_size = 8

factor_illegal_action = 1
prioritized_replay_buffer = True

parameters_updates = 10   
target_update_freq = 30    
opponent_update_freq = 1

epsilon_start = 1
epsilon_final = 0.1
epsilon_decay = 1200  

agents = ["player_0", "player_1"]
method = "eps_greedy" # "eps_greedy" or "boltzmann"
model = "DQN" # "DQN" or "ResNet" or "DCNNet"

per_move_rewards=True
final_reward_score_difference=False

name = f'{per_move_rewards}_{final_reward_score_difference}_model_{model}_episodes{num_episodes}_buffer{buffer_capacity}_prioritized{prioritized_replay_buffer}_batch_size{batch_size}_gamma{gamma}_target_update{target_update_freq}_opponent_freq{opponent_update_freq}_treshold_penalize{treshold_penalize_illegal}_penalize_value{penalize_value}_eps{epsilon_start}to{epsilon_final}with{epsilon_decay}_{method}'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device : {device}')

def epsilon_by_episode(episode):
    return epsilon_final + (epsilon_start - epsilon_final) * np.exp(-episode / epsilon_decay)

def temperature_by_episode(episode): # not implemented yet
    return 1


def create_networks(obs_shape, n_actions, model):
    if model == 'DQN':
        policy_net = DQN(obs_shape, n_actions).to(device)
        target_net = DQN(obs_shape, n_actions).to(device)
        opponent_net = DQN(obs_shape, n_actions).to(device)
        
    elif model == 'ResNet':
        policy_net = ResNet(obs_shape, n_actions, board_size).to(device)
        target_net = ResNet(obs_shape, n_actions, board_size).to(device)
        opponent_net = ResNet(obs_shape, n_actions, board_size).to(device)
    
    elif model == 'DCNNet':
        policy_net = DCNNet(obs_shape, n_actions, board_size).to(device)
        target_net = DCNNet(obs_shape, n_actions, board_size).to(device)
        opponent_net = DCNNet(obs_shape, n_actions, board_size).to(device)
        
    target_net.load_state_dict(policy_net.state_dict())
    opponent_net.load_state_dict(policy_net.state_dict())
    
    return policy_net, target_net, opponent_net

def compute_q_values(policy_net, target_net, next_states_tensor, next_action_masks_tensor, rewards_tensor, dones_tensor): # Double DQN
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


def select_action_dqn(model, obs, action_mask, legal_moves, episode, device, method, verbose=False):
    model.eval()
    with torch.no_grad():
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 10, 10, 5)
        q_values = model(obs_tensor).squeeze(0)  # (n_actions,)
        
        if method == 'eps_greedy':
            epsilon = epsilon_by_episode(episode)
            if random.random() < epsilon:
                first_action = random.choice(legal_moves)
                action = first_action
            else:
                first_action = torch.argmax(q_values).item()
                mask = torch.tensor(action_mask, dtype=torch.bool, device=device)
                q_values[~mask] = -1e8
                action = torch.argmax(q_values).item()
        
        elif method == 'boltzmann':
            temperature = temperature_by_episode(episode)
            first_action = torch.argmax(q_values).item()
            mask = torch.tensor(action_mask, dtype=torch.bool, device=device)
            q_values[~mask] = -1e8
            action = torch.argmax(q_values).item()
            probabilities = F.softmax(q_values / temperature, dim=-1)
            action = torch.multinomial(probabilities, num_samples=1).item()
            if mask[first_action]:
                first_action = action
        
    model.train()
    return first_action, action



def train_dqn(name):
    print(f'Name : {name}')
    env = cathedral_v0.env(board_size=board_size, render_mode="text", per_move_rewards=per_move_rewards, final_reward_score_difference=final_reward_score_difference)
    env.reset()
    
    enter_train = False
    n_actions = env.action_space('player_0').n
    print(f'n_actions : {n_actions}')
    obs_shape = env.observe('player_0')["observation"].shape

    policy_net, target_net, _ = create_networks(obs_shape, n_actions, model)
    
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    
    if prioritized_replay_buffer:
        replay_buffer = PrioritizedReplayBuffer(buffer_capacity, alpha=0.6)
    else:
       replay_buffer = ReplayBuffer(buffer_capacity) 
       
    list_reward = []
    list_epsilon = []
    list_loss = []      # To store average loss per episode
    list_steps = []     # To store number of steps per episode
    list_winner = []    # To store the winner result (1 win, 0.5 draw, 0 loss)
    list_legal_actions = []
    list_episode_ended = []
    list_legal_moves_possible = []
    win_count = 0

    for episode in range(num_episodes):
        if episode == treshold_penalize_illegal:
            print("STOP penalizing illegal actions")
        env.reset()
        list_agents = env.agents
        total_reward = 0
        legal_actions = 0
        losses = []
        episode_ended = True
        steps = 0
        while env.agents:
            steps += 1
            current_agent = env.agent_selection
            opponent_agent = 'player_0' if current_agent == 'player_1' else 'player_1'
            
            observation = env.observe(current_agent)
            legal_moves = [i for i, valid in enumerate(observation["action_mask"]) if valid]
            
            state = observation["observation"]
            action_mask = observation["action_mask"]
            
            first_action, action = select_action_dqn(policy_net, state, action_mask, legal_moves, episode, device, method=method)
            
            legal_action = first_action == action
            env.step(action)
            reward = env.rewards[current_agent]
            # reward -= factor_illegal_action * not_legal_action # Penalize illegal actions
            
            if episode > treshold_penalize_illegal:
                legal_action = True
            
            if legal_action:
                legal_actions += 1
                total_reward += reward
            else:
                total_reward -= 10
                
            if current_agent in env.agents and legal_action:
                next_obs = env.observe(opponent_agent)
                next_state = next_obs["observation"]
                next_action_mask = next_obs["action_mask"]
                done_flag = 0
            else:
                next_state = np.zeros_like(state)
                next_action_mask = np.zeros_like(action_mask)
                done_flag = 1
            
            if legal_action:
                replay_buffer.push(state, action, reward, next_state, done_flag, action_mask, next_action_mask)

            else:
                replay_buffer.push(state, first_action, -10, next_state, done_flag, action_mask, next_action_mask)
                
            
            if len(replay_buffer) >= batch_size:
                enter_train = True
                
                for _ in range(parameters_updates):  
                    if prioritized_replay_buffer:
                        states, actions, rewards, next_states, dones, action_masks, next_action_masks, _, _ = replay_buffer.sample(batch_size)
                    else:
                        states, actions, rewards, next_states, dones, action_masks, next_action_masks = replay_buffer.sample(batch_size)
                    states_tensor = torch.tensor(states, dtype=torch.float32).to(device)
                    actions_tensor = torch.tensor(actions, dtype=torch.long).to(device)
                    rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(device)
                    next_states_tensor = torch.tensor(next_states, dtype=torch.float32).to(device)
                    dones_tensor = torch.tensor(dones, dtype=torch.float32).to(device)
                    next_action_masks_tensor = torch.tensor(next_action_masks, dtype=torch.bool).to(device)
                    
                    q_values = policy_net(states_tensor)
                    q_values = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
                    
                    target_q_values = compute_q_values(policy_net, target_net, next_states_tensor, next_action_masks_tensor, rewards_tensor, dones_tensor)
                
                    loss = nn.MSELoss()(q_values, target_q_values)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    losses.append(loss.item())
                    
            if not legal_action:
                episode_ended = False
                break
                    
        list_legal_actions.append(legal_actions)
        list_reward.append(total_reward)
        list_steps.append(steps)
        list_epsilon.append(epsilon_by_episode(episode))
        
        if losses:
            avg_loss = sum(losses) / len(losses)
        else:
            avg_loss = 0
        list_loss.append(avg_loss)
        

        winner = env.winner
        if winner == -1:
            win_count += 0.5
            list_winner.append(0.5)
            wins = 'draw'
        else:
            if list_agents[winner] == 'player_0':  # "player_0" wins
                wins = 'player_0'
                win_count += 1
                list_winner.append(1)
            else:
                wins = 'player_1'
                list_winner.append(0)

        if episode_ended:
            list_episode_ended.append(1)
        else:
            list_episode_ended.append(0)
        
        if enter_train:
           print(
    f"Episode {episode+1}/{num_episodes} - Reward total: {total_reward:.2f} - Loss: {sum(losses)/(len(losses)+1e-8):.4f} - Winner: {wins}"
    + (f"- Legal Actions : {legal_actions} - Ep Ended: {episode_ended}" if treshold_penalize_illegal > 0 else "")
    + (f"- Epsilon: {epsilon_by_episode(episode):.2f}" if method=='eps_greedy' else f"- Temperature: {temperature_by_episode(episode):.2f}")
)

        if (episode+1) % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())
            print("Update target_net")
            
        # if (episode+1) % opponent_update_freq == 0:
        #     policy_net_checkpoints.append(policy_net.state_dict())
        #     opponent_net.load_state_dict(policy_net.state_dict())
        #     # print("Update opponent_net")    
        
    env.close()

    torch.save({
        'model_state_dict': policy_net.state_dict(),
        'list_reward': list_reward,
        'list_epsilon': list_epsilon,
        'list_loss': list_loss,
        'list_steps': list_steps,
        'list_winner': list_winner,
        'list_legal_actions': list_legal_actions,
        'list_episode_ended': list_episode_ended,
        'win_rate': win_count/num_episodes
    }, f"model_weights_DQN/{name}.pth")

    
    print(f'Winrate : {win_count/num_episodes}')
    
    return list_reward, list_epsilon, list_loss, list_steps, list_winner


list_reward, list_epsilon, list_loss, list_steps, list_winner = train_dqn('10000ep')