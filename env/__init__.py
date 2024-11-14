import gymnasium as gym
from .env import Env

gym.register(
    id = "myenv-v0",
    entry_point=Env,
)