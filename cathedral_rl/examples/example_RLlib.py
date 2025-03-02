import argparse
import os
import sys

import ray
from ray import air, tune
from ray.rllib.algorithms import ppo
from ray.rllib.env import PettingZooEnv
from ray.tune import register_env
from ray.tune.logger import pretty_print

import numpy as np

from ray.rllib.algorithms.ppo import PPOConfig

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cathedral_rl import cathedral_v0

import argparse
import os

import ray
from ray import air, tune
from ray.rllib.algorithms import ppo
from ray.rllib.env import PettingZooEnv
from ray.rllib.examples._old_api_stack.models.action_mask_model import (
    ActionMaskModel,
    TorchActionMaskModel,
)
from ray.rllib.policy.policy import PolicySpec
from ray.tune import register_env
from ray.tune.logger import pretty_print

from cathedral_rl import cathedral_v0

def get_cli_args():
    """Create CLI parser and return parsed arguments"""
    parser = argparse.ArgumentParser()

    # example-specific args
    parser.add_argument(
        "--no-masking",
        action="store_true",
        help="Do NOT mask invalid actions. This will likely lead to errors.",
    )

    # general args
    parser.add_argument(
        "--run", type=str, default="APPO", help="The RLlib-registered algorithm to use."
    )
    parser.add_argument("--num-cpus", type=int, default=0)
    parser.add_argument(
        "--framework",
        choices=["tf2", "torch"],
        default="torch",
        help="The DL framework specifier.",
    )
    parser.add_argument("--eager-tracing", action="store_true")
    parser.add_argument(
        "--stop-iters", type=int, default=10, help="Number of iterations to train."
    )
    parser.add_argument(
        "--stop-timesteps",
        type=int,
        default=10000,
        help="Number of timesteps to train.",
    )
    parser.add_argument(
        "--stop-reward",
        type=float,
        default=80.0,
        help="Reward at which we stop training.",
    )
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Run without Tune using a manual train loop instead. Here,"
        "there is no TensorBoard support.",
    )
    parser.add_argument(
        "--local-mode",
        action="store_true",
        help="Init Ray in local mode for easier debugging.",
    )

    args = parser.parse_args()
    print(f"Running with following CLI args: {args}")
    return args

def action_mask_policy(observation, model):
    """Policy function that considers the action mask."""
    action_mask = observation["action_mask"]
    
    # Get model output (action probabilities or Q-values)
    logits = model(observation["obs"])  # Shape (num_actions,)
    
    # Mask out illegal actions by setting their logits to a very negative value
    logits = logits * action_mask + (1 - action_mask) * -float("inf")
    
    # Select the action with the highest logit (highest probability for legal moves)
    action = np.argmax(logits)
    
    return action

if __name__ == "__main__":
    args = get_cli_args()

    ray.init(num_cpus=args.num_cpus or None, local_mode=args.local_mode)

    def env_creator(args):
        env = cathedral_v0.env()
        env = PettingZooEnv(env)
        print(env.observation_space)
        return env

    env = env_creator({})
    register_env("cathedral", env_creator)
    
    def policy_mapping_fn(agent_id, episode=None, worker=None):
        return agent_id

    print(f'observation space of player 1 {env.observation_space["player_1"]}')
    # Configure the PPO algorithm
    config = (
        PPOConfig()
        .environment("cathedral")
        .framework(args.framework, eager_tracing=args.eager_tracing)
        .resources(num_gpus=1)
        .multi_agent(
            policies={
                "player_0": PolicySpec(
                    action_space=env.action_space["player_0"],
                    observation_space=env.observation_space["player_0"],
                    # No need to use a custom model here, use the default RLlib model
                ),
                "player_1": PolicySpec(
                    action_space=env.action_space["player_1"],
                    observation_space=env.observation_space["player_1"],
                    # No need to use a custom model here, use the default RLlib model
                ),
                
            },
            policy_mapping_fn=policy_mapping_fn
        )
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        
    )
    
    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }

    # Manual training loop (no Ray tune)
    if args.no_tune:
        if args.run not in {"APPO", "PPO"}:
            raise ValueError("This example only supports APPO and PPO.")

        algo = config.build_algo()

        # Run manual training loop and print results after each iteration
        for _ in range(args.stop_iters):
            result = algo.train()
            print(pretty_print(result))
            # Stop training if the target train steps or reward are reached
            if (
                result["timesteps_total"] >= args.stop_timesteps
                or result["env_runners"]["episode_reward_mean"] >= args.stop_reward
            ):
                break

        # Manual test loop
        print("Finished training. Running manual test/inference loop.")
        # Prepare environment with max 10 steps
        config["env_config"]["max_episode_len"] = 10
        env = env_creator({})
        obs, info = env.reset()
        done = False
        # Run one iteration until done
        print(f"ActionMaskEnv with {config['env_config']}")
        while not done:
            action = algo.compute_single_action(obs)
            next_obs, reward, done, truncated, _ = env.step(action)
            # Observations contain original observations and the action mask
            # Reward is random and irrelevant here and therefore not printed
            print(f"Obs: {obs}, Action: {action}")
            obs = next_obs

    # Run with tune for auto trainer creation, stopping, TensorBoard, etc.
    else:
        tuner = tune.Tuner(
            args.run,
            param_space=config.to_dict(),
            run_config=air.RunConfig(stop=stop, verbose=2),
        )
        tuner.fit()

    print("Finished successfully without selecting invalid actions.")
    ray.shutdown()