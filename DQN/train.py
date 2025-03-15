
from ReplayBuffer import ReplayBuffer, PrioritizedReplayBuffer
from utils import epsilon_by_episode, temperature_by_episode, create_networks, select_action_dqn, compute_q_values
from cathedral_rl import cathedral_v0 
import torch.optim as optim
import torch
import numpy as np
import torch.nn as nn

def train_dqn(name, board_size, num_episodes, buffer_capacity, treshold_penalize_illegal, batch_size, gamma, learning_rate, prioritized_replay_buffer, parameters_updates, target_update_freq, epsilon_start, epsilon_final, epsilon_decay, temperature_start, temperature_final, temperature_decay, method, model, per_move_rewards, final_reward_score_difference, device):
    print(f'Model name : {name}')

    env = cathedral_v0.env(board_size=board_size, render_mode="text", per_move_rewards=per_move_rewards, final_reward_score_difference=final_reward_score_difference)
    env.reset()
    
    n_actions = env.action_space('player_0').n
    print(f'n_actions : {n_actions}')
    obs_shape = env.observe('player_0')["observation"].shape

    policy_net, target_net = create_networks(obs_shape, n_actions, model, board_size, device)
    
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    
    if prioritized_replay_buffer:
        replay_buffer = PrioritizedReplayBuffer(buffer_capacity, alpha=0.6)
    else:
       replay_buffer = ReplayBuffer(buffer_capacity) 
    
    enter_train = False
    list_reward = []
    list_epsilon = []
    list_temperature = []
    list_loss = []      
    list_steps = []     
    list_winner = []    
    list_legal_actions = []
    list_episode_ended = []
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
        epsilon = epsilon_by_episode(episode, epsilon_start, epsilon_final, epsilon_decay)
        temperature = temperature_by_episode(episode, temperature_start, temperature_final, temperature_decay)

        while env.agents:
            steps += 1
            current_agent = env.agent_selection
            opponent_agent = 'player_0' if current_agent == 'player_1' else 'player_1'
            
            observation = env.observe(current_agent)
            legal_moves = [i for i, valid in enumerate(observation["action_mask"]) if valid]
            
            state = observation["observation"]
            action_mask = observation["action_mask"]
            
            first_action, action = select_action_dqn(policy_net, state, action_mask, legal_moves, device, method, epsilon, temperature)
                                 
            legal_action = first_action == action
            env.step(action)
            reward = env.rewards[current_agent]
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
                    
                    target_q_values = compute_q_values(policy_net, target_net, next_states_tensor, next_action_masks_tensor, rewards_tensor, dones_tensor, gamma)
                
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
        list_epsilon.append(epsilon_by_episode(episode, epsilon_start, epsilon_final, epsilon_decay))
        list_temperature.append(temperature_by_episode(episode, temperature_start, temperature_final, temperature_decay))
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
    + (f"- Epsilon: {epsilon_by_episode(episode, epsilon_start, epsilon_final, epsilon_decay):.2f}" if method=='eps_greedy' else f"- Temperature: {temperature_by_episode(episode, temperature_start, temperature_final, temperature_decay):.2f}")
)

        if (episode+1) % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())
            print("Update target_net") 
        
    env.close()

    torch.save({
        'model_state_dict': policy_net.state_dict(),
        'list_reward': list_reward,
        'list_epsilon': list_epsilon,
        'list_temperature': list_temperature,
        'list_loss': list_loss,
        'list_steps': list_steps,
        'list_winner': list_winner,
        'list_legal_actions': list_legal_actions,
        'list_episode_ended': list_episode_ended,
        'win_rate': win_count/num_episodes
    }, f"model_weights_DQN/{name}.pth")

    
    print(f'Winrate : {win_count/num_episodes}')
    