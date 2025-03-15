from model import DQN
from utils import select_action_dqn, epsilon_by_episode, temperature_by_episode
from cathedral_rl import cathedral_v0 
import torch
from tqdm import tqdm
import random

def evaluate_dqn(name, opponents, board_size, device, num_episodes_eval=1000, epsilon=0.15):
    env = cathedral_v0.env(board_size=board_size, render_mode="text", per_move_rewards=True, final_reward_score_difference=False)
    env.reset()
    
    controlled_agent='player_0' # player_0 or player_1 (does not matter)

    n_actions = env.action_space(controlled_agent).n
    obs_shape = env.observe(controlled_agent)["observation"].shape 
    
    checkpoint = torch.load(f"model_weights_DQN/{name}.pth", weights_only=False)

    policy_net = DQN(obs_shape, n_actions).to(device)
    policy_net.load_state_dict(checkpoint['model_state_dict'])        
    
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
        