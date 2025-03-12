import multiprocessing
import torch
import random
import numpy as np
import torch.nn.functional as F
from cathedral_rl import cathedral_v0  
import random


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import deque
import matplotlib.pyplot as plt
from tqdm import tqdm 
import math
import copy
from cathedral_rl.game.board import Board 

class AlphaZeroNet(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super(AlphaZeroNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(obs_shape[2], 32, kernel_size=3, stride=1, padding=1),  # -> 32 x 10 x 10
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),              # -> 64 x 10 x 10
            nn.ReLU(),
            nn.Flatten()
        )
        
        dummy = torch.zeros(1, obs_shape[2], obs_shape[0], obs_shape[1])
        conv_out_size = self.conv(dummy).shape[1]
        # print(f'conv_out_size: {conv_out_size}')
        
        self.policy_fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        
        self.value_fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Tanh()  
        )
    
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        features = self.conv(x)
        policy_logits = self.policy_fc(features)
        value = self.value_fc(features)
        return policy_logits, value


def initial_state(board_size=8):
    board = Board(board_size=board_size, players=["player_0", "player_1"])
    # "player_0" starts
    return {"board": board, "current_agent": "player_0"}

def get_legal_moves(state):
    board = state["board"]
    agent = state["current_agent"]
    legal_moves = []
    for action in range(board.num_actions):
        if board.is_legal(agent, action):
            legal_moves.append(action)
    return legal_moves

def is_terminal(state):
    """
    Determine is the state is terminal.
    A state is terminal if no legal moves are available for both players.
    """
    board = state["board"]
    current_agent = state["current_agent"]
    legal_moves_current = [action for action in range(board.num_actions) if board.is_legal(current_agent, action)]
    
    opponent = "player_1" if current_agent == "player_0" else "player_0"
    legal_moves_opponent = [action for action in range(board.num_actions) if board.is_legal(opponent, action)]
    
    return (len(legal_moves_current) == 0) and (len(legal_moves_opponent) == 0)

def evaluate_terminal(state):
    """
    Evaluate a terminal state by returning +1, -1 or 0.
    We use the board.check_for_winner() function which returns:
        - 0 if player_0 wins,
        - 1 if player_1 wins,
        - -1 in case of a draw.
    The result is returned from the point of view of the agent that was active in the initial state of the self-play.
    """
    board = state["board"]
    winner, _, _ = board.check_for_winner()
    current_agent = state["current_agent"]
    if winner == -1:
        return 0  # draw
    if (winner == 0 and current_agent == "player_0") or (winner == 1 and current_agent == "player_1"):
        return 1
    else:
        return -1

def next_state(state, action):
    """
    From a state and an action, simulates the move and returns the new state.
    Here we perform a deep copy of the state to avoid altering the original.
    """
    new_state = copy.deepcopy(state)
    board = new_state["board"]
    agent = new_state["current_agent"]
    
    board.play_turn(agent, action)

    new_state["board"] = board
    
    opponent = "player_1" if agent == "player_0" else "player_0"
    legal_moves_opponent = [a for a in range(board.num_actions) if board.is_legal(opponent, a)]
    
    if len(legal_moves_opponent) > 0:
        new_state["current_agent"] = opponent
    else:
        # If the opponent has no legal moves, the current agent plays again
        new_state["current_agent"] = agent
        
    return new_state

def state_to_observation(state):
    """
    Converts the state (dictionary containing "board" and "current_agent")
    into an observation (numpy array of shape (board_size, board_size, 5)).
    Inspired by the observe function of the environment.
    """
    board = state["board"]
    agent = state["current_agent"]
    board_size = board.board_size

    board_vals = board.squares.reshape(board_size, board_size)
    board_territory = board.territory.reshape(board_size, board_size)
    
    cur_player = board.possible_agents.index(agent)
    opp_player = (cur_player + 1) % 2

    cur_p_board = np.equal(board_vals, cur_player + 1)
    opp_p_board = np.equal(board_vals, opp_player + 1)
    cathedral_board = np.equal(board_vals, 3)
    cur_p_territory = np.equal(board_territory, cur_player + 1)
    opp_p_territory = np.equal(board_territory, opp_player + 1)

    observation = np.stack(
        [cur_p_board, opp_p_board, cathedral_board, cur_p_territory, opp_p_territory],
        axis=2
    ).astype(np.float32)
    return observation
class MCTSNode:
    def __init__(self, state, parent=None, prior=0.0):
        self.state = state
        self.parent = parent
        self.children = {}  # dictionnaire : action -> MCTSNode
        self.N = 0          # number of visits
        self.W = 0.0        # sum of values
        self.Q = 0.0        # mean value
        self.P = prior      # prior probability from the policy network
        self.is_expanded = False

def select_child(node, c_puct):
    best_score = -float('inf')
    best_action = None
    best_child = None
    count = 0
    for action, child in node.children.items():
        # print(f'count select child: {count}')
        count += 1
        U = c_puct * child.P * math.sqrt(node.N) / (1 + child.N)
        score = child.Q + U
        if score > best_score:
            best_score = score
            best_action = action
            best_child = child
    
    # print(f'selecting_child => total child : {len(node.children)} / selected action: {best_action}')
    return best_action, best_child


def expand_node(node, network, device):
    legal_moves = get_legal_moves(node.state)
    # print(f'expanding current node => legal moves : {len(legal_moves)}')
    
    obs = state_to_observation(node.state)
    # print(f'obs shape : {obs.shape}')
    state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
    policy_logits, value = network(state_tensor)
    
    policy = F.softmax(policy_logits.squeeze(0), dim=0).detach().cpu().numpy()
    
    # print(f'policy shape : {policy.shape}')
    # print(f'policy : {policy}')
    for action in legal_moves:
        if action not in node.children:
            new_state = next_state(node.state, action)
            node.children[action] = MCTSNode(new_state, parent=node, prior=policy[action])
    node.is_expanded = True
    return value.item()


def mcts_search(root, network, device, n_simulations, c_puct, n_actions):
    for _ in range(n_simulations):
        # print(f'simulation mcts search: {_+1}/{n_simulations}')
        node = root
        search_path = [node]
        count = 0
        while node.is_expanded and node.children:
            count += 1
            action, node = select_child(node, c_puct)
            # print(f'exploring node => action selected: {action}')
            search_path.append(node)
            
        if is_terminal(node.state):
            value = evaluate_terminal(node.state)
        else:
            value = expand_node(node, network, device)
            
        for i in reversed(range(len(search_path))):
            node = search_path[i]
            node.N += 1
            node.W += value
            node.Q = node.W / node.N
            if i > 0:
                parent_agent = search_path[i - 1].state["current_agent"]
                current_agent = node.state["current_agent"]
                if parent_agent != current_agent:
                    value = -value

def get_policy_from_mcts(root, n_actions, temperature):
    # print(f'getting policy from root')
    counts = np.zeros(n_actions)
    for action, child in root.children.items():
        counts[action] = child.N
    if temperature == 0:
        # Deterministic policy
        best_action = np.argmax(counts)
        policy = np.zeros_like(counts)
        policy[best_action] = 1.0
    else:
        counts_temp = counts ** (1 / temperature)
        policy = counts_temp / np.sum(counts_temp)
    return policy

def self_play_game(network, device, n_simulations, c_puct, temperature, n_actions, saved_initial_node, id):
    print(f'Starting self-play game: {id}')
    game_history = []  # (state, pi, joueur)
    
    action_history = []
    current_player = 1  # par exemple, 1 pour le joueur courant, -1 pour l'adversaire
        
    root = saved_initial_node  
    state = root.state
    
    while not is_terminal(state):
        # print('State still not terminal')
        
        mcts_search(root, network, device, n_simulations, c_puct, n_actions)
        
        pi = get_policy_from_mcts(root, n_actions, temperature)
        
        obs = state_to_observation(state) 
        current_player_numeric = 1 if state["current_agent"] == "player_0" else -1
        game_history.append((obs, pi, current_player_numeric))
        
        action = np.random.choice(n_actions, p=pi)
        action_history.append(action)
        
        old_agent = state["current_agent"]
        next_state_value = next_state(state, action)
        new_agent = next_state_value["current_agent"]
        
        if new_agent != old_agent:
            current_player = -current_player
        
        root = root.children[action]
        state = next_state_value
        
        # print(f'Action choosen: {action}')
        # print(f'len(game_history): {len(game_history)}') 
    
    print(f'Game {id}: actions played: {action_history}')
    
    outcome = evaluate_terminal(state) 
    training_examples = []
    
    for s, pi, player in game_history:
        value_target = outcome if player == 1 else -outcome
        training_examples.append((s, pi, value_target))
    
    # print(f'training examples: {training_examples}')
        
    return training_examples, saved_initial_node


def run_self_play_game(args):
    id, network_state_dict, obs_shape, n_simulations, c_puct, temperature, n_actions = args
    # Créer une instance du réseau sur CPU
    model_cpu = AlphaZeroNet(obs_shape, n_actions)
    model_cpu.load_state_dict(network_state_dict)
    model_cpu.to('cpu')
    
    # Initialiser l'état de jeu pour la partie
    initial = initial_state(board_size=6)
    saved_initial_node = MCTSNode(initial)
    
    # Lancer la self-play game en utilisant le modèle sur CPU
    game_data, _ = self_play_game(model_cpu, 'cpu', n_simulations, c_puct, temperature, n_actions, saved_initial_node, id)
    return game_data

def train_alphazero_parallel(network, optimizer, device, num_iterations, n_games, n_simulations, c_puct, temperature, n_actions, batch_size, obs_shape, num_updates):
    memory = []
    losses = []
    for iteration in range(num_iterations):
        print(f'Iteration: {iteration}/{num_iterations-1}')
        
        # Récupérer le state_dict du modèle pour le transmettre aux processus de self-play
        network_state_dict = network.state_dict()
        # Préparer la liste d'arguments pour chaque partie self-play
        args_list = [
            (i, network_state_dict, obs_shape, n_simulations, c_puct, temperature, n_actions)
            for i in range(n_games)
        ]
        
        # Lancer les parties en parallèle sur CPU
        with multiprocessing.Pool(processes=n_games) as pool:
            results = pool.map(run_self_play_game, args_list)
        
        # Agréger les données de jeu
        for game_data in results:
            memory.extend(game_data)
        
        print(f'len(memory): {len(memory)}')
        
        for update in range(num_updates):
            batch = random.sample(memory, batch_size)
            states, target_policies, target_values = zip(*batch)
            
            states_tensor = torch.tensor(np.array(states), dtype=torch.float32).to(device)
            target_policies_tensor = torch.tensor(np.array(target_policies), dtype=torch.float32).to(device)
            target_values_tensor = torch.tensor(target_values, dtype=torch.float32).unsqueeze(1).to(device)
            
            pred_policies_logits, pred_values = network(states_tensor)
            
            value_loss = F.mse_loss(pred_values, target_values_tensor)
            policy_loss = -torch.mean(torch.sum(target_policies_tensor * F.log_softmax(pred_policies_logits, dim=1), dim=1))
            
            loss = value_loss + policy_loss
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(f'Iteration: {iteration}/{num_iterations-1} => Update {update}/{num_updates-1}, Loss: {loss.item():.4f}, Value Loss: {value_loss.item():.4f}, Policy Loss: {policy_loss.item():.4f}')
    
    torch.save({
        'model_state_dict': network.state_dict(),
        'losses': losses},
        f"model_alphazero_num_iterations:{num_iterations}_n_games:{n_games}_n_simulations:{n_simulations}_num_updates:{num_updates}_c_puct:{c_puct}_temperature:{temperature}.pth")

if __name__ == '__main__':
    # Définir la méthode de démarrage à 'spawn'
    multiprocessing.set_start_method('spawn', force=True)
    board_size = 6
    
    env = cathedral_v0.env(board_size=board_size, render_mode="text", per_move_rewards=False, final_reward_score_difference=False)
    env.reset()
    n_actions = env.action_space("player_0" ).n
    obs_shape = (board_size, board_size, 5)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = AlphaZeroNet(obs_shape, n_actions=n_actions).to(device)  # Assure-toi que n_actions correspond à ton environnement
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    
    train_alphazero_parallel(
        network=network,
        optimizer=optimizer,
        device=device,
        num_iterations=100,       # Par exemple
        n_games=10,               # Nombre de parties en parallèle
        n_simulations=50,         # Nombre de simulations MCTS
        c_puct=1.0,
        temperature=1.0,
        n_actions=n_actions,
        batch_size=32,
        obs_shape=obs_shape,
        num_updates = 10
    )
