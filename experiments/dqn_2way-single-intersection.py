import os
import sys

import gymnasium as gym
from stable_baselines3 import DQN

# Ensure SUMO_HOME is correctly set
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import traci
from sumo_rl import SumoEnvironment


if __name__ == "__main__":
    # Create SUMO environment
    env = SumoEnvironment(
        net_file="sumo_rl/nets/2way-single-intersection/single-intersection.net.xml",
        route_file="sumo_rl/nets/2way-single-intersection/single-intersection-vhvh.rou.xml",
        out_csv_name="outputs/2way-single-intersection/dqn",
        single_agent=True,
        use_gui=False,
        num_seconds=1000,
    )

    # Initialize DQN model with tuned hyperparameters
    model = DQN(
        env=env,
        policy="MlpPolicy",
        learning_rate=5e-4,
        learning_starts=1000,
        train_freq=4,
        target_update_interval=1000,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        exploration_fraction=0.1,
        buffer_size=50000,
        batch_size=64,
        verbose=1,
        # tensorboard_log="./runs/sumo_dqn"
    )

    # Train the model
    model.learn(total_timesteps=50000)

    # Save the trained model
    model.save("outputs/2way-single-intersection/dqn_model")

    # Close the environment
    env.close()
