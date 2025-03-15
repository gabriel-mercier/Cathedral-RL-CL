import os
import sys
sys.path.append('..')
from cathedral_rl import cathedral_v0  
import numpy as np
import pygame 
import torch
from PIL import Image

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
        self.action_masks = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]
        del self.action_masks[:]

def print_buffer_sequence(ppo_agent):
    """
    Utility function to print the entire buffer sequence with (state, action, reward)
    for each step in an episode.
    
    Args:
        ppo_agent: An instance of the PPOCNN class
    """
    buffer = ppo_agent.buffer
    
    if len(buffer.rewards) == 0:
        print("Buffer is empty.")
        return
    
    print("\n===== BUFFER SEQUENCE =====")
    step_counter = 0
    
    for i in range(len(buffer.rewards)):
        state = buffer.states[i]
        action = buffer.actions[i].item()
        reward = buffer.rewards[i]
        is_terminal = buffer.is_terminals[i]
        
        # Print step information
        print(f"  Step {step_counter}:")
        print(f"    State shape: {state.shape}")
        print(f"    Action: {action}")
        print(f"    Reward: {reward}")
        print(f"    Terminal: {is_terminal}")
        
        step_counter += 1
    
    print("===== END OF BUFFER =====\n")


def create_game_gif(episode_num, ppo_agent, gif_dir, board_size):
        """Create and save a GIF animation of a full game using Pygame rendering"""
        print(f"Recording game animation for episode {episode_num}")
        
        # Initialize rendering environment for this game
        gif_env = cathedral_v0.env(board_size=board_size, render_mode="human", 
                                  per_move_rewards=True, final_reward_score_difference=False)
        gif_env.reset()
        
        # Initialize pygame if not already done
        if pygame.get_init() == False:
            pygame.init()
        
        # Store frames for the GIF
        frames = []
        
        # Create a clock for controlling rendering speed
        clock = pygame.time.Clock()
        
        # Capture initial board state
        gif_env.render()
        pygame.display.flip()
        frame = pygame.surfarray.array3d(pygame.display.get_surface())
        frame = np.transpose(frame, (1, 0, 2))  # Transpose to correct orientation
        frames.append(Image.fromarray(frame))
        
        # Game loop
        while gif_env.agents:
            current_agent = gif_env.agent_selection
            observation = gif_env.observe(current_agent)
            state = observation["observation"]
            action_mask = observation["action_mask"]
            
            # Get action using PPO
            action = ppo_agent.select_action(state, action_mask)
            
            # Take step
            gif_env.step(action)
            
            # Render and capture frame
            gif_env.render()
            pygame.display.flip()
            
            # Capture the displayed frame
            frame = pygame.surfarray.array3d(pygame.display.get_surface())
            frame = np.transpose(frame, (1, 0, 2))  # Transpose to correct orientation
            frames.append(Image.fromarray(frame))
            
            # Control rendering speed
            clock.tick(2)  # Limit to 2 FPS for the gif
        
        # Save frames as a GIF
        gif_path = os.path.join(gif_dir, f"game_episode_{episode_num}.gif")
        frames[0].save(
            gif_path,
            format='GIF',
            append_images=frames[1:],
            save_all=True,
            duration=500,  # Duration between frames in milliseconds
            loop=0  # Loop indefinitely
        )
        
        # Close environment
        gif_env.close()
        print(f"Game animation saved to {gif_path}")
        
        return
    
def evaluate_ppo_against_random(ppo_agent, num_episodes=100, board_size=8, render=False):
    """
    Evaluate a trained PPO agent against a random policy after training is complete.
    
    Args:
        model_path (str): Path to the saved PPO model weights
        num_episodes (int): Number of evaluation episodes
        board_size (int): Size of the board
        render (bool): Whether to render the game
        
    Returns:
        dict: Evaluation metrics including win rate
    """
    env = cathedral_v0.env(
        board_size=board_size, 
        render_mode="ansi" if render else None, 
        per_move_rewards=True, 
        final_reward_score_difference=False
    )
    env.reset()
    
    # Get observation shape and action space
    player_0 = "player_0"  # PPO agent
    player_1 = "player_1"  # Random agent
    n_actions = env.action_space(player_0).n
    obs_shape = env.observe(player_0)["observation"].shape
    
    # Set to evaluation mode
    ppo_agent.policy.eval()
    
    # Statistics
    stats = {
        'ppo_wins': 0,
        'random_wins': 0,
        'draws': 0,
        'ppo_rewards': [],
        'random_rewards': [],
    }
    
    for episode in range(num_episodes):
        env.reset()
        list_agents = env.agents
        episode_reward_ppo = 0
        episode_reward_random = 0
        
        # Print progress
        if episode % 10 == 0:
            print(f"Playing episode {episode}/{num_episodes}")
        
        # Clear buffer for evaluation (since we don't need to preserve it post-training)
        ppo_agent.buffer.clear()
        
        while env.agents:
            current_agent = env.agent_selection
            observation = env.observe(current_agent)
            
            if current_agent == player_0:  # PPO agent's turn
                state = observation["observation"]
                action_mask = observation["action_mask"]
                
                # Use PPO to select action (deterministic for evaluation)
                action = ppo_agent.select_action_evaluation(state, action_mask)
                
                if render:
                    print(f'PPO action {action}')
                
            else:  # Random agent's turn
                action_mask = observation["action_mask"]
                
                # Random valid action
                valid_actions = np.where(action_mask == 1)[0]
                action = np.random.choice(valid_actions)
                
                if render:
                    print(f'Random action {action}')
            
            # Take action
            env.step(action)
            
            # Track rewards
            if current_agent == player_0:
                episode_reward_ppo += env.rewards[current_agent]
            else:
                episode_reward_random += env.rewards[current_agent]
        
        # Record game outcome
        winner = env.winner
        if winner != -1 and list_agents[winner] == player_0:
            stats['ppo_wins'] += 1
            if render:
                print(f'Episode {episode} - PPO wins!')
        elif winner != -1 and list_agents[winner] == player_1:
            stats['random_wins'] += 1
            if render:
                print(f'Episode {episode} - Random wins!')
        else:  # Draw
            stats['draws'] += 1
            if render:
                print(f'Episode {episode} - Draw!')
        
        # Record episode rewards
        stats['ppo_rewards'].append(episode_reward_ppo)
        stats['random_rewards'].append(episode_reward_random)
    
    # Calculate win rates
    stats['ppo_win_rate'] = stats['ppo_wins'] / num_episodes
    stats['random_win_rate'] = stats['random_wins'] / num_episodes
    stats['draw_rate'] = stats['draws'] / num_episodes
    
    # Calculate average rewards
    stats['avg_ppo_reward'] = sum(stats['ppo_rewards']) / num_episodes
    stats['avg_random_reward'] = sum(stats['random_rewards']) / num_episodes
    
    # Print summary
    print("\n===== Evaluation Results =====")
    print(f"Episodes played: {num_episodes}")
    print(f"PPO Wins: {stats['ppo_wins']} ({stats['ppo_win_rate']:.2%})")
    print(f"Random Wins: {stats['random_wins']} ({stats['random_win_rate']:.2%})")
    print(f"Draws: {stats['draws']} ({stats['draw_rate']:.2%})")
    print(f"Average PPO Reward: {stats['avg_ppo_reward']:.2f}")
    print(f"Average Random Reward: {stats['avg_random_reward']:.2f}")
    
    env.close()
    return stats