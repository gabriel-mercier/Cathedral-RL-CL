# noqa
"""
# Connect Four

```{figure} classic_connect_four.gif
:width: 140px
:name: connect_four
```

This environment is part of the <a href='..'>classic environments</a>. Please read that page first for general information.

| Import             | `from pettingzoo.classic import connect_four_v3` |
|--------------------|--------------------------------------------------|
| Actions            | Discrete                                         |
| Parallel API       | Yes                                              |
| Manual Control     | Yes                                              |
| Agents             | `agents= ['player_0', 'player_1']`               |
| Agents             | 2                                                |
| Action Shape       | (1,)                                             |
| Action Values      | Discrete(7)                                      |
| Observation Shape  | (6, 7, 2)                                        |
| Observation Values | [0,1]                                            |


Connect Four is a 2-player turn based game, where players must connect four of their tokens vertically, horizontally or diagonally. The players drop their respective token in a column of a standing grid, where each token will fall until it reaches the bottom of the column or reaches an existing
token. Players cannot place a token in a full column, and the game ends when either a player has made a sequence of 4 tokens, or when all 7 columns have been filled.

### Observation Space

The observation is a dictionary which contains an `'observation'` element which is the usual RL observation described below, and an  `'action_mask'` which holds the legal moves, described in the Legal Actions Mask section.


The main observation space is 5 channels each representing planes of a 10x10 grid:
 * Channel 1: White pieces: a 1 indicates a white piece occupies a given space, 0 means either the space is empty, or the opponent has a piece occupying that space.
 * Channel 2: Black pieces: a 1 indicates a brown piece occupies a given space, 0 means either the space is empty, or the opponent has a piece occupying that space.
 * Channel 3: Cathedral: a 1 indicates the cathedral occupies a given space, 0 means that space is empty or is occupied by a player's piece.
 * Channel 4: White territory: a 1 indicates a given space is part of white's territory, a 0 means the space is occupied, and -1 indicates it is not part of white's territory
 * Channel 5: Black territory: a 1 indicates a given space is part of white's territory, a 0 means the space is occupied, and -1 indicates it is not part of brown's territory

The first and second panes represent a specific agent's tokens, and each location in the grid represents the placement of the corresponding agent's token. 1 indicates that the agent has a token placed in that cell, and 0 indicates they do not have a token in
that cell. A 0 means that either the cell is empty, or the other agent has a token in that cell.



#### Legal Actions Mask

The legal moves available to the current agent are found in the `action_mask` element of the dictionary observation.
The `action_mask` is a binary vector where each index of the vector represents whether the action is legal or not. The `action_mask` will be all zeros for any agent except the one
whose turn it is. Taking an illegal move ends the game with a reward of -1 for the illegally moving agent and a reward of 0 for all other agents.


### Action Space

The action space is the set of integers from 0 to 6 (inclusive), where the action represents which column a token should be dropped in.

### Rewards

If an agent successfully connects four of their tokens, they will be rewarded 1 point. At the same time, the opponent agent will be awarded -1 points. If the game ends in a draw, both players are rewarded 0.


### Version History

* v0: Initial versions release (1.0.0)

"""
import functools
from typing import Optional

import gymnasium
import numpy as np
import pygame
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils.agent_selector import agent_selector

from .board import Board


def env(board_size=10, render_mode=None, per_move_rewards=False, final_reward_score_difference=False):
    env = raw_env(
        board_size=board_size,
        render_mode=render_mode,
        per_move_rewards=per_move_rewards,
        final_reward_score_difference=final_reward_score_difference,
    )
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class raw_env(AECEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "cathedral_v0",
        "is_parallelizable": False,
        "render_fps": 0,
        "has_manual_policy": True,
    }

    def __init__(
        self,
        board_size,
        render_mode=None,
        per_move_rewards: Optional[bool] = False,
        final_reward_score_difference: Optional[bool] = False,
    ):

        super().__init__()
        self.screen = None
        self.render_mode = render_mode
        self.board_size = board_size
        # Enable to set per-move rewards as heuristic score of how good a move is (used by simple greedy agent)
        self.per_move_rewards = per_move_rewards

        # Enable to change final score returns from {-1, 1} to return the score difference for each agent
        # Useful for testing score-based vs winrate-based optimization (
        self.final_reward_score_difference = final_reward_score_difference

        # Pygame setup
        if render_mode == "human":
            # pygame.init()
            # self.clock = pygame.time.Clock
            # self.WINDOW_WIDTH = 1000
            # self.WINDOW_HEIGHT = 1000
            # self.window = pygame.display.set_mode(
            #     (self.WINDOW_HEIGHT/2, self.WINDOW_HEIGHT/2)
            # )
            # self.clock = pygame.time.Clock()
            # self.WINDOW_WIDTH, self.WINDOW_HEIGHT = self.window.get_size()
            pygame.init()
            self.clock = pygame.time.Clock()
            # Option 1: Automatically set to current display resolution
            info = pygame.display.Info()
            self.WINDOW_WIDTH, self.WINDOW_HEIGHT = info.current_w, info.current_h
            # Option 2: Alternatively, set your own default resolution
            # self.WINDOW_WIDTH, self.WINDOW_HEIGHT = 1000, 1000
            self.window = pygame.display.set_mode(
                (self.WINDOW_WIDTH, self.WINDOW_HEIGHT), pygame.RESIZABLE
            )
            # Update dimensions after window is created
            self.WINDOW_WIDTH, self.WINDOW_HEIGHT = self.window.get_size()


        self.agents = ["player_0", "player_1"]
        self.possible_agents = self.agents[:]
        
        self.board = Board(board_size=self.board_size, players= self.agents)

        self.action_spaces = {
            i: spaces.Discrete(self.board.num_actions) for i in self.agents
        }
        self.observation_spaces = {
            i: spaces.Dict(
                {
                    "observation": spaces.Box(
                        low=0, high=1, shape=(self.board_size, self.board_size, 5), dtype=np.int8
                    ),
                    "action_mask": spaces.Box(
                        low=0, high=1, shape=(self.board.num_actions,), dtype=np.int8
                    ),
                }
            )
            for i in self.agents
        }

    # Key
    # ----
    # blank space = 0
    # agent 0 = 1
    # agent 1 = 2
    # An observation is list of lists, where each list represents a row
    #
    # array([[0, 1, 1, 2, 0, 1, 0],
    #        [1, 0, 1, 2, 2, 2, 1],
    #        [0, 1, 0, 0, 1, 2, 1],
    #        [1, 0, 2, 0, 1, 1, 0],
    #        [2, 0, 0, 0, 1, 1, 0],
    #        [1, 1, 2, 1, 0, 1, 0]], dtype=int8)
    def observe(self, agent):
        board_vals = self.board.squares.reshape(self.board_size, self.board_size)
        board_territory = self.board.territory.reshape(self.board_size, self.board_size)
        cur_player = self.possible_agents.index(agent)
        opp_player = (cur_player + 1) % 2

        cur_p_board = np.equal(board_vals, cur_player + 1)
        opp_p_board = np.equal(board_vals, opp_player + 1)
        cathedral_board = np.equal(board_vals, 3)
        cur_p_territory = np.equal(board_territory, cur_player + 1)
        opp_p_territory = np.equal(board_territory, opp_player + 1)

        layers = [
            cur_p_board,
            opp_p_board,
            cathedral_board,
            cur_p_territory,
            opp_p_territory,
        ]

        observation = np.stack(layers, axis=2).astype(np.int8)

        # If this is the first observation, calculate legal moves, otherwise this is done every step
        if len(self.legal_moves[agent]) == 0:
            self._calculate_legal_moves(agent)
            
        legal_moves = self.legal_moves[agent] if agent == self.agent_selection else []

        action_mask = np.zeros(self.action_space(agent).n, dtype=np.int8)
        for i in range(self.action_space(agent).n):
            if i in legal_moves:
                action_mask[i] = 1

        return {"observation": observation, "action_mask": action_mask}

    # this cache ensures that same space object is returned for the same agent
    # allows action space seeding to work as expected
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]

    # Calculate the number of legal moves per agent, legal moves per piece, and legal pieces to be played
    def _calculate_legal_moves(self, agent):
        #print(f'Calculating legal moves of {agent}')
        legal_moves = []
        self.legal_moves_per_piece[agent] = np.zeros(self.board.num_pieces)

        for act in range(self.board.num_actions):
            if self.board.is_legal(agent, act):
                legal_moves.append(act)
                self.legal_moves_per_piece[agent][
                    self.board.action_to_piece_map(act)[0]
                ] += 1
        self.legal_pieces[agent] = self.legal_moves_per_piece[agent].nonzero()[0]
        self.legal_moves[agent] = legal_moves

    # Calculate rewards for a given step: score of piece placed + amount of territory claimed + size of piece removed
    # Score of a piece placed: size of piece - size of largest legally playable piece remaining
    # This penalizes playing small pieces when there are larger pieces available to place, rewards claiming territory
    def _calculate_rewards(
        self, piece_size: int, territory_claimed: int, piece_removed_size: int
    ):
        # Piece score: number of potential squares lost by move (zero if a piece of the largest size possible is played)
        legal_piece_sizes = [
            self.board.pieces[self.agent_selection][piece].size
            for piece in self.legal_pieces[self.agent_selection]
        ]
        piece_score = piece_size - np.amax(legal_piece_sizes)

        self.rewards[self.agent_selection] = piece_score + territory_claimed
        if piece_removed_size != 0:
            self.rewards[self.agent_selection] += piece_removed_size
            
            self.rewards[self.agents[1 - self.agents.index(self.agent_selection)]] = (
                -1 * piece_removed_size
            )

    # Calculate score heuristic: squares/turn + difference between total squares remaining + difference in territory
    # Score will be positive if agent has placed more large pieces, or claimed more territory
    def _calculate_score(self):
        _, score = self.board.get_score()
        territory = self.board.territory
        
        # Calculate components
        for i in range(2):
            # Squares per turn (efficiency metric)
            if self.turns[self.agents[i]] > 0:
                avg_squares = (self.board.total_piece_squares - score[self.agents[i]]) / self.turns[self.agents[i]]
                # Normalize to typical range
                self.score[self.agents[i]]["squares_per_turn"] = avg_squares / 5.0  # Assuming avg piece size ~5
            else:
                self.score[self.agents[i]]["squares_per_turn"] = 0
            
            # Remaining pieces (placement advantage)
            opponent_idx = 1 - i
            self.score[self.agents[i]]["remaining_pieces"] = (
                score[self.agents[opponent_idx]] - score[self.agents[i]]
            ) / 20.0  # Normalize by approximate total remaining piece value
            
            # Territory control
            agent_territory = len(territory[territory == i+1])
            opponent_territory = len(territory[territory == opponent_idx+1])
            self.score[self.agents[i]]["territory"] = (agent_territory - opponent_territory) / (self.board_size * self.board_size / 2)
            
            # Calculate weighted total
            self.score[self.agents[i]]["total"] = (
                0.5 * self.score[self.agents[i]]["squares_per_turn"] +  # Efficiency matters but less
                1.0 * self.score[self.agents[i]]["remaining_pieces"] +  # Piece advantage is important
                2.0 * self.score[self.agents[i]]["territory"]           # Territory is most important
            )
        
        # Add win bonus - ensure winning is always better than losing
        winner = self.board.check_for_winner()[0]
        if winner >= 0:  # Not a draw
            win_bonus = 5.0  # Significant bonus that dominates other components
            self.score[self.agents[winner]]["total"] += win_bonus
            self.score[self.agents[1-winner]]["total"] -= win_bonus

    # Calculate winner at the end of game
    def _calculate_winner(self):
        winner, pieces_remaining, piece_score = self.board.check_for_winner()
        if self.final_reward_score_difference:
            self._calculate_score()
            self.rewards[self.agents[0]] = self.score[self.agents[0]]['total']
            self.rewards[self.agents[1]] = self.score[self.agents[1]]['total']
        else:
            if winner == 0:
                self.rewards[self.agents[0]] = 10
                self.rewards[self.agents[1]] = -10
            elif winner == 1:
                self.rewards[self.agents[0]] = -10
                self.rewards[self.agents[1]] = 10
            else:
                self.rewards[self.agents[0]] = 0
                self.rewards[self.agents[1]] = 0

        self.winner = winner
        self.final_pieces = pieces_remaining
        self.piece_score = piece_score
        self.terminations = {i: True for i in self.agents}
        self.agents = []

    def next_step(self, state, action, agent):
        if (
            self.truncations[agent]
            or self.terminations[agent]
        ):

            return self._was_dead_step(action)

        # Check that it is a valid move
        if not self.board.is_legal(agent, action):
            raise Exception("played illegal move.")

        # Play the turn
        piece_size = self.board.play_turn(agent, action)
        territory_claimed, piece_removed_size = self.board.check_territory(
            agent
        )

        # Don't count placing the cathedral as a turn (only count placing regular pieces)
        if piece_size != 6:
            self.turns[self.agent_selection] += 1

        if self.per_move_rewards:
            self._calculate_rewards(
                piece_size, territory_claimed, piece_removed_size
            )  # Heuristic reward for current move
        self._accumulate_rewards()

        # Calculate score heuristics every other turn (when both agents have placed the same number of pieces)
        if self.turns[self.agents[0]] == self.turns[self.agents[1]]:
            self._calculate_score()

        next_agent = self._agent_selector.next()

        # If the next agent has legal moves to play, switch agents
        self._calculate_legal_moves(next_agent)
        if len(self.legal_moves[next_agent]) != 0:
            self.agent_selection = next_agent

        # If both agents have zero moves left (game over), calculate winners
        else:
            self._calculate_legal_moves(self.agent_selection)
            if len(self.legal_moves[self.agent_selection]) == 0:
                self._calculate_score()  # Calculate score heuristics (even if one agent has played more turns)
                self._calculate_winner()
                # print("GAME OVER")
                # return self._was_dead_step(action)

            # If the next agent has no legal moves left, current agent continues placing pieces
            else:
                # print(f"{next_agent} out of moves, {self.agent_selection} continues placing")
                if len(self.legal_moves[self.agent_selection]) == 0:
                    self.terminations[self.agent_selection] = True

        if self.render_mode == "human":
            self.render()
        # else:
            # print(f"Cumulative rewards: {self._cumulative_rewards}, rewards: {self.rewards}")

    def step(self, action):
        if (
            self.truncations[self.agent_selection]
            or self.terminations[self.agent_selection]
        ):

            return self._was_dead_step(action)

        # Check that it is a valid move
        if not self.board.is_legal(self.agent_selection, action):
            raise Exception("played illegal move.")

        # Play the turn
        piece_size = self.board.play_turn(self.agent_selection, action)
        territory_claimed, piece_removed_size = self.board.check_territory(
            self.agent_selection
        )
        #print(f'Step by agent {self.agent_selection} piece size {piece_size}')

        # Don't count placing the cathedral as a turn (only count placing regular pieces)
        if piece_size != 6:
            self.turns[self.agent_selection] += 1

        if self.per_move_rewards:
            self._calculate_rewards(
                piece_size, territory_claimed, piece_removed_size
            )  # Heuristic reward for current move
        self._accumulate_rewards()
        
        # Calculate score heuristics every other turn (when both agents have placed the same number of pieces)
        if self.turns[self.agents[0]] == self.turns[self.agents[1]]:
            self._calculate_score()

        next_agent = self._agent_selector.next()

        # If the next agent has legal moves to play, switch agents
        self._calculate_legal_moves(next_agent)
        if len(self.legal_moves[next_agent]) != 0:
            self.agent_selection = next_agent

        # If both agents have zero moves left (game over), calculate winners
        else:
            self._calculate_legal_moves(self.agent_selection)
            if len(self.legal_moves[self.agent_selection]) == 0:
                self._calculate_score()  # Calculate score heuristics (even if one agent has played more turns)
                self._calculate_winner()
                # print("GAME OVER")
                # return self._was_dead_step(action)

            # If the next agent has no legal moves left, current agent continues placing pieces
            else:
                # print(f"{next_agent} out of moves, {self.agent_selection} continues placing")
                if len(self.legal_moves[self.agent_selection]) == 0:
                    self.terminations[self.agent_selection] = True

        if self.render_mode == "human":
            self.render()
        # else:
            # print(f"Cumulative rewards: {self._cumulative_rewards}, rewards: {self.rewards}")

    def reset(self, seed=None, return_info=False, options=None):

        random = np.random.random()
        if random < 0.5 :
            self.agents = self.possible_agents[:]
        else :
            self.agents = [self.possible_agents[1], self.possible_agents[0]]
            
            # reset environment
        self.board = Board(board_size=self.board_size, players=self.agents)
        
        #print(f'reset, agents are {self.agents}')
        self.rewards = {i: 0 for i in self.agents}
        self._cumulative_rewards = {name: 0 for name in self.agents}
        self.terminations = {i: False for i in self.agents}
        self.truncations = {i: False for i in self.agents}
        self.infos = {i: {} for i in self.agents}

        self._agent_selector = agent_selector(self.agents)

        self.agent_selection = self._agent_selector.reset()

        # Track the number of turns each agent has been able to play so far
        self.turns = {agent: 0 for agent in self.agents}

        # Track the total number of legal moves per agent, legal moves per piece, and legal pieces to play
        self.legal_moves = {agent: [] for agent in self.agents}
        self.legal_moves_per_piece = {
            agent: np.zeros(self.board.num_pieces) for agent in self.agents
        }
        self.legal_pieces = {agent: [] for agent in self.agents}

        # Additional info about game outcome
        self.winner = -1
        self.score = {
            name: {
                "squares_per_turn": 0,
                "remaining_pieces": 0,
                "territory": 0,
                "total": 0,
            }
            for name in self.agents
        }
        self.final_pieces = {name: [] for name in self.agents}

        # Additional heuristics for how well an agent is doing
        self._score = {name: 0 for name in self.agents}
        self._piece_score = {name: 0 for name in self.agents}
        
    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        elif self.render_mode == "human":
            # Update the window dimensions in case of resize
            self.WINDOW_WIDTH, self.WINDOW_HEIGHT = self.window.get_size()
            self.clock.tick(self.metadata["render_fps"])

            # Use the smaller dimension to calculate block_size so the board fits in both directions
            block_size = min(self.WINDOW_WIDTH // self.board_size, self.WINDOW_HEIGHT // self.board_size)
            border_color = (0, 0, 0)
            self.colors = {
                0: (211, 211, 211),
                1: (221, 186, 151),
                2: (120, 65, 65),
                3: (128, 128, 162),
                4: (238, 221, 203),
                5: (188, 160, 160),
                6: (0, 0, 0),
                7: (233, 210, 187),
                8: (208, 189, 189),
                9: (172, 172, 195),
            }

            board_reshaped = self.board.squares.reshape(self.board_size, self.board_size)
            territory_reshaped = self.board.territory.reshape(self.board_size, self.board_size)

            # Loop using board indices rather than pixel positions
            for x in range(self.board_size):
                for y in range(self.board_size):
                    x_screen = x * block_size
                    y_screen = y * block_size

                    if (
                        territory_reshaped[x, y] > 0
                        and board_reshaped[x, y] == 0
                    ):
                        # Using territory value + 3 to choose a color
                        color = self.colors[territory_reshaped[x, y] + 3]
                    else:
                        color = self.colors[board_reshaped[x, y]]

                    border = color != self.colors[0]

                    # Draw the square (cell) for the board
                    pygame.draw.rect(self.window, color, (x_screen, y_screen, block_size, block_size), 0)
                    if border:
                        pygame.draw.rect(self.window, border_color, (x_screen, y_screen, block_size, block_size), 2)
                    else:
                        pygame.draw.rect(self.window, border_color, (x_screen, y_screen, block_size, block_size), 1)

            pygame.display.update()

        elif self.render_mode == "text":
            print("Board: \n", self.board.squares.reshape(self.board_size, self.board_size))
            print("Territory: \n", self.board.territory.reshape(self.board_size, self.board_size))

    def close(self):
        if self.render_mode == "human":
            import pygame

            pygame.quit()
            self.screen = None
            

