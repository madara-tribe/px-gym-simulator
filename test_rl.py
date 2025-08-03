
import gym
import gym_laser_tracker
from stable_baselines3 import PPO
import matplotlib.pyplot as plt

# Load environment and model
env = gym.make("LaserTracker-v0")
model = PPO.load("ppo_laser_tracker", env=env)

# Run test episodes
obs = env.reset()
test_rewards = []
ep_reward = 0

for i in range(100):
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()
    ep_reward += reward
    if done:
        test_rewards.append(ep_reward)
        ep_reward = 0
        obs = env.reset()

env.close()

# Plotting rewards
plt.figure()
plt.plot(test_rewards, label="Episode reward")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("RL Inference Performance")
plt.legend()
plt.grid()
plt.show()
