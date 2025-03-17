from model import DQN
from utils import select_action_dqn, epsilon_by_episode, temperature_by_episode
import sys
sys.path.append('..')
from cathedral_rl import cathedral_v0 
import torch
from tqdm import tqdm
import random

def evaluate_dqn(opponents, board_size, device, name=None, state_dict=None, num_episodes_eval=1000, epsilon=0.15):
    env = cathedral_v0.env(board_size=board_size, render_mode="text", per_move_rewards=True, final_reward_score_difference=False)
    env.reset()
    
    controlled_agent='player_0' # player_0 or player_1 (does not matter)

    n_actions = env.action_space(controlled_agent).n
    obs_shape = env.observe(controlled_agent)["observation"].shape 
    
    policy_net = DQN(obs_shape, n_actions).to(device)

    if name is not None:
        checkpoint = torch.load(f"model_weights_DQN/{name}.pth", weights_only=False)
        policy_net.load_state_dict(checkpoint['model_state_dict'])    
    else:
        policy_net.load_state_dict(state_dict)
    
    for opponent in opponents:
        list_reward = []
        win_count = 0
        print(f"Opponent {opponent}")
            
        for episode in tqdm(range(num_episodes_eval)):
            env.reset()
            list_agents = env.agents
            total_reward = 0

            while env.agents:
                current_agent = env.agent_selection
                observation = env.observe(current_agent)
                legal_moves = [i for i, valid in enumerate(observation["action_mask"]) if valid]
                state = observation["observation"]
                action_mask = observation["action_mask"]
                    
                if current_agent == controlled_agent: # tested model plays
                    _, action = select_action_dqn(policy_net, state, action_mask, legal_moves, device, 'eps_greedy', 0, 0, training=False)
               
                    reward = env.rewards[current_agent]
                    total_reward += reward
                    list_reward.append(total_reward)
                    
                else:  # opponent plays
                    if opponent == -1: # random opponent
                        action = random.choice(legal_moves)
                    else:
                        if random.random() < epsilon: # random action for opponent to avoid deterministic behavior
                            action = random.choice(legal_moves)
                        else:
                            _, action = select_action_dqn(policy_net, state, action_mask, legal_moves, device, 'eps_greedy', 0, 0, training=False)
                    
                
                env.step(action)
                        
            winner = env.winner
            if winner == -1:
                win_count += 0.5
            else:
                if list_agents[winner] == controlled_agent:  # the tested model wins
                    win_count += 1

        
        avg_reward = sum(list_reward)/len(list_reward)
        print(f"Opponent {opponent} : {num_episodes_eval} episodes => Avg Reward : {avg_reward:.4f} // Winrate : {win_count/num_episodes_eval:.4f}")
        
        return avg_reward, win_count/num_episodes_eval
    


def compare_dqn(model_1, model_2, board_size, device, num_episodes_eval=1000, epsilon=0.15):
    env = cathedral_v0.env(board_size=board_size, render_mode="text", per_move_rewards=True, final_reward_score_difference=False)
    env.reset()
    
    controlled_agent_1='player_0' # player_0 or player_1 (does not matter)

    n_actions = env.action_space(controlled_agent_1).n
    obs_shape = env.observe(controlled_agent_1)["observation"].shape 
    
    model_1_net = DQN(obs_shape, n_actions).to(device)
    model_2_net = DQN(obs_shape, n_actions).to(device)

    checkpoint_1 = torch.load(f"model_weights_DQN/{model_1}.pth", weights_only=False)
    model_1_net.load_state_dict(checkpoint_1['model_state_dict'])

    checkpoint_2 = torch.load(f"model_weights_DQN/{model_2}.pth", weights_only=False)
    model_2_net.load_state_dict(checkpoint_2['model_state_dict'])

    list_reward_1 = []
    win_count_1 = 0
    list_reward_2 = []
    win_count_2 = 0
        
    for episode in tqdm(range(num_episodes_eval)):
        env.reset()
        list_agents = env.agents

        while env.agents:
            current_agent = env.agent_selection
            observation = env.observe(current_agent)
            legal_moves = [i for i, valid in enumerate(observation["action_mask"]) if valid]
            state = observation["observation"]
            action_mask = observation["action_mask"]

            if current_agent == controlled_agent_1: # model 1 plays
                _, action = select_action_dqn(model_1_net, state, action_mask, legal_moves, device, 'boltzmann', 0, 0.3)
                reward = env.rewards[current_agent]
                list_reward_1.append(reward)
                
            else:  
                _, action = select_action_dqn(model_2_net, state, action_mask, legal_moves, device, 'boltzmann', 0, 0.3)
                reward = env.rewards[current_agent]
                list_reward_2.append(reward)
            
            env.step(action)
                        
        winner = env.winner
        if winner == -1:
            win_count_1 += 0.5
            win_count_2 += 0.5
        else:
            if list_agents[winner] == controlled_agent_1:  # the tested model wins
                win_count_1 += 1
            else:
                win_count_2 += 1

        
    avg_reward_1 = sum(list_reward_1)/len(list_reward_1)
    avg_reward_2 = sum(list_reward_2)/len(list_reward_2)
    winrate_1 = win_count_1/num_episodes_eval
    winrate_2 = win_count_2/num_episodes_eval
    print(f" {num_episodes_eval} episodes:\nAvg Reward 1: {avg_reward_1:.4f} // Winrate 1: {winrate_1:.4f}\nAvg Reward 2: {avg_reward_2:.4f} // Winrate 2: {winrate_2:.4f}")
    
    return avg_reward_1, winrate_1, avg_reward_2, winrate_2