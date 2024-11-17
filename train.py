import gymnasium as gym
import os
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.env_util import make_vec_env
import numpy as np

import env

env_id = "myenv-v0"
n_training_envs = 3
n_eval_envs = 5

# Create log dir where evaluation results will be saved
eval_log_dir = "./ppo_eval_logs_42/"
os.makedirs(eval_log_dir, exist_ok=True)

# Initialize a vectorized training environment with default parameters
train_env = make_vec_env(env_id, n_envs=n_training_envs, seed=0)

# Separate evaluation env, with different parameters passed via env_kwargs
# Eval environments can be vectorized to speed up evaluation.
eval_env = make_vec_env(env_id, n_envs=n_eval_envs, seed=0)

eval_callback = EvalCallback(eval_env, best_model_save_path=eval_log_dir,
                              log_path=eval_log_dir, eval_freq=max(500 // n_training_envs, 1),
                              n_eval_episodes=5, deterministic=True,
                              render=False)

if len(sys.argv) > 1:
  load_path = sys.argv[1]
  model = PPO.load(load_path, train_env, verbose=1, tensorboard_log='./tensorboard/')
else:
  model = PPO("MlpPolicy", train_env, verbose=1, tensorboard_log='./tensorboard/')
model.learn(100000, callback=eval_callback, batch_size=512, n_steps=512, n_epochs=5)