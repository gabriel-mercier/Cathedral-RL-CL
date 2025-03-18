# cathedral-rl
[![PyPI version](https://badge.fury.io/py/cathedral-rl.svg?branch=master&kill_cache=1)](https://badge.fury.io/py/cathedral-rl)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](https://github.com/elliottower/cathedral-rl/blob/main/LICENSE)

Interactive Multi-Agent Reinforcement Learning Environment for the board game Cathedral using PettingZoo

<h1 style="text-align: center;width: 80%">
    <img alt="Cathedral board game" src="assets/cathedral_game.jpg" width="45%">
    &nbsp; &nbsp; &nbsp; &nbsp;
    <img alt="Cathedral-rl game rendered with Pygame" src="assets/cathedral_screenshot.jpg" width="45%">
</h1>

## Installation Instructions
To set up the project, follow these steps:

1. Ensure you have **Python 3.12.8** installed. Other versions of **Python 3.12** should also work.
2. Create a virtual environment using either **venv** or **conda**:
   - Using `venv`:
     ```bash
     python -m venv venv
     source venv/bin/activate  # On macOS/Linux
     venv\Scripts\activate  # On Windows
     ```
   - Using `conda`:
     ```bash
     conda create --name myenv python=3.12.8
     conda activate myenv
     ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt

## Methods

### DQN (Deep Q-Network)

To train your model using DQN:

#### Preparation

Create an empty folder named `model_weights_DQN` to store metrics during training:

```bash
mkdir model_weights_DQN
```

#### Execution
Open the associated Jupyter Notebook, adjust the hyperparameters in the designated cell, and then start the training.

### MCTS (Monte Carlo Tree Search)
The MCTS feature is currently in a debugging phase and is not operational. You can review the implemented architecture, but it is not recommended to run it as the results will not be meaningful.

### Minimax
An alpha-beta pruning minimax algorithm is available to play against a random opponent. Note that a full execution may take approximately 5 minutes. To test this approach, run the dedicated Python script.

### PPO (Proximal Policy Optimization)
The PPO module is accessible via a Jupyter Notebook. You can choose to train against a random opponent or via self-play. Adjust the parameters directly in the Notebook according to your needs.

## Usage
Once the installation is complete, you can experiment with the different training methods:

1. DQN: 
Open the corresponding Notebook and follow the instructions to start the training.
2. Minimax:
Run the Python script to observe a game against a random player.
3. PPO:
Launch the Notebook and choose the desired training mode.

## License
This project is licensed under the MIT License. You are free to use, modify, and distribute it according to the terms of this license.