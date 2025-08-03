#import gymnasium as

import gym
import gym_laser_tracker
from stable_baselines3 import PPO

env = gym.make("LaserTracker-v0")

# Train the agent
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# Save the model
model.save("ppo_laser_tracker")

# Optional: load and test
model = PPO.load("ppo_laser_tracker")
obs = env.reset()
for _ in range(100):
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()
env.close()

