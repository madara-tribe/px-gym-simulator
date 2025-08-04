#import gymnasium as
import gym
import gym_laser_tracker  # this registers your env

env = gym.make("LaserTracker-v0")

obs = env.reset()
for _ in range(100):
    env.render()
    action = env.action_space.sample()  # random action
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()
env.close()
