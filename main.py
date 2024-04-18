import gymnasium as gym
import numpy as np

import pandas as pd
import os
import json
from stable_baselines3 import PPO, DQN


log_dir = "tmp/"
os.makedirs(log_dir, exist_ok=True)
# gym.make arguments
# - to always get all 18 action space
#       full_action_space=True
# - variantion of observation space
#       obs_type="rgb"
#       obs_type="grayscale"
#       obs_type="ram"
# - make the environment stocastic 
#       repeat_action_probability=0.25 # 0 - 1
#       frameskip = int|turple(1x2)
# - add flavour
#       mode = int
#       difficult = int

seed = 2345
total_timesteps = 700_000

df = pd.read_csv('atari_env.csv', usecols=[i for i in range(1,6)])

id = 4 # Alien-v5
game = df.iloc[id]

env_id = game.Name # 'ALE/Alien-v5'
name = env_id.split('/')[-1]

modes = game.Modes[1:-1].split(', ')
difficulties = game.Difficulties[1:-1].split(', ')
i = 0
tasks = []
for mode in modes:

    for difficulty in difficulties:

        task = {
            'id':i,
            'name':env_id,
            'difficulty':int(difficulty),
            'mode':int(mode),
            'seed':seed
        }
        tasks.append(task)

        env = gym.make(env_id, render_mode="rgb_array", mode = int(mode), difficulty = int(difficulty), full_action_space = True)

        env.seed(seed)

        model = DQN("CnnPolicy", env, verbose=1,tensorboard_log=log_dir+"/Alien")
        model.learn(total_timesteps = total_timesteps, progress_bar=True)

        # Save the agent
        model.save(f"dqn_{name}_task-{i}")
        i += 1

        env.close()

with open('task-config.json', 'w') as f:
    json.dump(tasks, f)
# 
