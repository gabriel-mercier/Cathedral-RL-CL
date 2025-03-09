import os
import sys
sys.path.append('..')
from cathedral_rl import cathedral_v0  
import numpy as np
import pygame 
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