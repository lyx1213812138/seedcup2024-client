import gymnasium as gym
import os
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.env_util import make_vec_env
import numpy as np

import env

import my_train_test as my_test

env_id = "myenv-v0"
n_training_envs = 5
n_eval_envs = 10

# Create log dir where evaluation results will be saved
# eval_log_dir = "./ppo_eval_logs_47/end2"
load_path = os.path.join(os.path.dirname(__file__),  "ppo_eval_logs_42/batch256/left_perfect/best_model2.zip")
eval_log_dir = os.path.join(os.path.dirname(__file__), "ppo_eval_logs_42/batch256/left_perfect/best_model2.zip")
os.makedirs(eval_log_dir, exist_ok=True)

# Initialize a vectorized training environment with default parameters
train_env = make_vec_env(env_id, n_envs=n_training_envs, seed=0)

# Separate evaluation env, with different parameters passed via env_kwargs
# Eval environments can be vectorized to speed up evaluation.
eval_env = make_vec_env(env_id, n_envs=n_eval_envs, seed=0)

class my_log(BaseCallback):
  def __init__(self, verbose: int = 1):
    super().__init__(verbose)

  def _on_training_end(self):
    return True

  def _on_step(self):
    my_test.main(self.logger)
    return True


eval_callback = EvalCallback(eval_env, best_model_save_path=eval_log_dir,
                              log_path=eval_log_dir,
                              n_eval_episodes=5, deterministic=True,
                              render=False, eval_freq=2000)
                              # callback_on_new_best=my_log())

if len(sys.argv) > 1:
  load_path = sys.argv[1]
  model = PPO.load(load_path, train_env, verbose=1, tensorboard_log='./tensorboard/')
elif load_path:
  model = PPO.load(load_path, train_env, verbose=1, tensorboard_log='./tensorboard/')
else:
  model = PPO("MlpPolicy", train_env, verbose=1, tensorboard_log='./tensorboard/', 
              batch_size=256, n_steps=2048, n_epochs=10)
model.learn(50000000, callback=eval_callback)