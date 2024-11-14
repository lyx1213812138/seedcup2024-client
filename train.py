import gymnasium as gym
import env

from stable_baselines3 import PPO

env = gym.make("myenv-v0")

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log='./tensorboard/')
model.learn(total_timesteps=100000)
model.save("ppo")
