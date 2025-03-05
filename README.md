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
Before training, create an empty folder named `model_weights_DQN` to store metrics during training:
    ```bash
    mkdir model_weights_DQN
Then, run the associated Jupyter Notebook.
You can adjust the hyperparameters in the designated cell before starting the training.

### MCTS (Monte Carlo Tree Search)
Currently, MCTS is not functional (still in debugging phase).
You can explore the architecture we aim to implement, but running it at this stage is not recommended as it will not produce meaningful results.

### PPO
  Running is possible in the associated notebook. One can choose to train against a random opponent or self-play. 