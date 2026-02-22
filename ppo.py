import gymnasium as gym
import rustoracerpy
from stable_baselines3 import PPO
import numpy as np

env = gym.make("Rustoracer-v0", yaml="maps/berlin.yaml")
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    n_steps=4096,
)
model.learn(total_timesteps=5_000_000)
model.save("racer_ppo")
env.close()

env = gym.make("Rustoracer-v0", yaml="maps/berlin.yaml", render_mode="human")
model = PPO.load("racer_ppo")

obs, info = env.reset()
while True:
    action = model.predict(obs, deterministic=True)[0]
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        obs, info = env.reset()
