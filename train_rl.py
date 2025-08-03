
import gym
import gym_laser_tracker
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure

# Create environment
env = gym.make("LaserTracker-v0")

# Create model
model = PPO("MlpPolicy", env, verbose=1)

# Set up TensorBoard logging
log_path = "logs/ppo_laser_tracker_tensorboard/"
new_logger = configure(log_path, ["stdout", "tensorboard"])
model.set_logger(new_logger)

# Train and save
total_timesteps = 30000
model.learn(total_timesteps=total_timesteps)

model_path = "ppo_laser_tracker"
model.save(model_path)

env.close()
