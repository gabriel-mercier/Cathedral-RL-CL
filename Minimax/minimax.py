import copy
import math
import numpy as np

from cathedral_rl.game.board import Board 

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

def evaluate_non_terminal(state):
    """
    Évalue un état non terminal en se basant sur les scores des pièces restantes.
    Un score positif indique que l'état est favorable pour le joueur courant.
    
    On utilise board.get_score() qui retourne :
      - pieces_remaining : liste des pièces non placées par joueur,
      - piece_score : somme des tailles des pièces restantes pour chaque joueur.
    
    L'évaluation est calculée comme :
        evaluation = (score de l'adversaire) - (score du joueur courant)
    """
    board = state["board"]
    current_agent = state["current_agent"]
    opponent = "player_1" if current_agent == "player_0" else "player_0"
    
    _, piece_score = board.get_score()
    
    return piece_score[opponent] - piece_score[current_agent]

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
        return 100000
    else:
        return -100000

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

def copy_piece(piece):
    saved = {}
    for key, value in piece.__dict__.items():
        if isinstance(value, np.ndarray):
            saved[key] = value.copy()
        else:
            saved[key] = value
    return saved

def save_board_state(board):
    saved = {
        'squares': board.squares.copy(),
        'territory': board.territory.copy(),
        'unplaced_pieces': {agent: list(board.unplaced_pieces[agent]) for agent in board.unplaced_pieces},
        'pieces': {}
    }
    for agent in board.possible_agents:
        saved['pieces'][agent] = [copy_piece(piece) for piece in board.pieces[agent]]
    return saved

def restore_board_state(board, saved):
    board.squares[:] = saved['squares']
    board.territory[:] = saved['territory']
    for agent in board.possible_agents:
        board.unplaced_pieces[agent] = list(saved['unplaced_pieces'][agent])
        for i, piece in enumerate(board.pieces[agent]):
            saved_piece = saved['pieces'][agent][i]
            for key, value in saved_piece.items():
                if isinstance(value, np.ndarray):
                    board.pieces[agent][i].__dict__[key] = value.copy()
                else:
                    board.pieces[agent][i].__dict__[key] = value

def save_state(state):
    return {
        "board_state": save_board_state(state["board"]),
        "current_agent": state["current_agent"]
    }

def restore_state(state, saved):
    restore_board_state(state["board"], saved["board_state"])
    state["current_agent"] = saved["current_agent"]

def apply_move_in_place(state, action):
    board = state["board"]
    agent = state["current_agent"]
    board.play_turn(agent, action)
    
    opponent = "player_1" if agent == "player_0" else "player_0"
    legal_moves_opponent = [a for a in range(board.num_actions) if board.is_legal(opponent, a)]
    
    state["current_agent"] = opponent

def minimax_in_place(state, depth, maximizingPlayer):
    if depth == 0 or is_terminal(state):
        return evaluate_terminal(state), None

    legal_moves = get_legal_moves(state)
    if not legal_moves:
        return evaluate_terminal(state), None

    best_move = None
    saved = save_state(state)
    
    if maximizingPlayer:
        maxEval = -math.inf
        for move in legal_moves:
            restore_state(state, saved)
            apply_move_in_place(state, move)
            eval_score, _ = minimax_in_place(state, depth - 1, False)
            if eval_score > maxEval:
                maxEval = eval_score
                best_move = move
        restore_state(state, saved)
        return maxEval, best_move
    else:
        minEval = math.inf
        for move in legal_moves:
            restore_state(state, saved)
            apply_move_in_place(state, move)
            eval_score, _ = minimax_in_place(state, depth - 1, True)
            if eval_score < minEval:
                minEval = eval_score
                best_move = move
        restore_state(state, saved)
        return minEval, best_move

def minimax_decision_in_place(state, depth):
    _, best_move = minimax_in_place(state, depth, True)
    return best_move

def alphabeta_in_place(state, depth, alpha, beta, maximizingPlayer):
    if depth == 0 or is_terminal(state):
        if is_terminal(state):
            return evaluate_terminal(state), None
        else:
            return evaluate_non_terminal(state), None

    legal_moves = get_legal_moves(state)
    if not legal_moves:
        if is_terminal(state):
            return evaluate_terminal(state), None
        else:
            return evaluate_non_terminal(state), None

    best_move = None
    saved = save_state(state)

    if maximizingPlayer:
        maxEval = -math.inf
        for move in legal_moves:
            restore_state(state, saved)
            apply_move_in_place(state, move)
            eval_score, _ = alphabeta_in_place(state, depth - 1, alpha, beta, False)
            if eval_score > maxEval:
                maxEval = eval_score
                best_move = move
            alpha = max(alpha, maxEval)
            if alpha >= beta:
                break
        restore_state(state, saved)
        return maxEval, best_move
    else:
        minEval = math.inf
        for move in legal_moves:
            restore_state(state, saved)
            apply_move_in_place(state, move)
            eval_score, _ = alphabeta_in_place(state, depth - 1, alpha, beta, True)
            if eval_score < minEval:
                minEval = eval_score
                best_move = move
            beta = min(beta, minEval)
            if alpha >= beta:
                break
        restore_state(state, saved)
        return minEval, best_move

def alphabeta_decision_in_place(state, depth):
    _, best_move = alphabeta_in_place(state, depth, -math.inf, math.inf, True)
    return best_move

if __name__ == "__main__":
    board_size = 8
    state = initial_state(board_size=board_size)
    
    while not is_terminal(state):
        current_agent = state["current_agent"]
        print("État du plateau :")
        print(state["board"].squares.reshape(board_size, board_size))
        legal_moves = get_legal_moves(state)
        if current_agent == "player_0":
            if len(legal_moves) < 100:
                print("depth = 2")
                print(f"{len(legal_moves)} moves available")
                best_action = alphabeta_decision_in_place(state, depth=2)
            else:
                print(f"{len(legal_moves)} moves available")
                best_action = alphabeta_decision_in_place(state, depth=1)
            print(f"Alpha-beta Minimax choisit l'action : {best_action}")
            state = next_state(state, best_action)
        else:
            print(f"{len(legal_moves)} moves available")
            moves = legal_moves
            action = moves[0] if moves else None
            print(f"Random joue l'action : {action}")
            state = next_state(state, action)
    
    print("Jeu terminé.")
    print("État final du plateau :")
    print(state["board"].squares.reshape(board_size, board_size))
    print(f"Résultat : {evaluate_terminal(state)}")
