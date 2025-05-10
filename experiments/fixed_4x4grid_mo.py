import argparse
import os
import sys

import pandas as pd

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumo_rl import SumoEnvironment

if __name__ == "__main__":
    runs = 1
    episodes = 1

    # Initialize environment with FIXED traffic lights (fixed_ts=True)
    env = SumoEnvironment(
        net_file="sumo_rl/nets/4x4-Lucas/4x4.net.xml",
        route_file="sumo_rl/nets/4x4-Lucas/4x4c1c2c1c2.rou.xml",
        use_gui=True,
        num_seconds=10000,
        reward_fn=["diff-waiting-time", "average-speed"],
        reward_weights=[1, 0.1],
        enforce_max_green=True,
        min_green=5,
        delta_time=5,
        fixed_ts=True,  # Critical: Enables fixed-time traffic signals
    )

    for run in range(1, runs + 1):
        for episode in range(1, episodes + 1):
            env.reset()  # Reset environment for each episode
            done = {"__all__": False}
            while not done["__all__"]:
                # Step through without agent actions (traffic lights follow fixed plans)
                s, r, done, info = env.step(action={})
            
            # Save metrics to compare with RL
            env.save_csv(f"outputs/4x4/fixed-4x4grid_run{run}", episode)

    env.close()