import torch
from train import train_dqn

board_size = 8

num_episodes = 4000
buffer_capacity = 10000

treshold_penalize_illegal = 1500
treshold_play_vs_random = 0

batch_size = 64
gamma = 0.95
learning_rate = 1e-3

prioritized_replay_buffer = True

parameters_updates = 10   
target_update_freq = 30  

epsilon_start = 1
epsilon_final = 0.1
epsilon_decay = 500
temperature_start = 1
temperature_final = 0.1
temperature_decay = 500

method = "boltzmann" # "eps_greedy" or "boltzmann"
model = "DQN" # "DQN" or "ResNet" or "DCNNet"

per_move_rewards=True
final_reward_score_difference=False

name = f'{per_move_rewards}_{final_reward_score_difference}_model_{model}_episodes{num_episodes}_buffer{buffer_capacity}_prioritized{prioritized_replay_buffer}_batch_size{batch_size}_gamma{gamma}_target_update{target_update_freq}_treshold_penalize{treshold_penalize_illegal}_{method}'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device : {device}')

train_dqn(name=name, 
        board_size=board_size, 
        num_episodes=num_episodes, 
        buffer_capacity=buffer_capacity, 
        treshold_penalize_illegal=treshold_penalize_illegal,  
        treshold_play_vs_random= treshold_play_vs_random,
        batch_size=batch_size,
        gamma=gamma, 
        learning_rate=learning_rate, 
        prioritized_replay_buffer=prioritized_replay_buffer, 
        parameters_updates=parameters_updates, 
        target_update_freq=target_update_freq, 
        epsilon_start=epsilon_start, 
        epsilon_final=epsilon_final, 
        epsilon_decay=epsilon_decay, 
        temperature_start=temperature_start, 
        temperature_final=temperature_final, 
        temperature_decay=temperature_decay, 
        method=method, 
        model=model, 
        per_move_rewards=per_move_rewards, 
        final_reward_score_difference=final_reward_score_difference, 
        device=device)
    

