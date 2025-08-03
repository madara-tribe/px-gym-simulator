#import gymnasium as gym
import gym
from gym import spaces
import numpy as np

class LaserTrackerEnv(gym.Env):
    def __init__(self):
        super(LaserTrackerEnv, self).__init__()
        self.servo_angle = 0  # start facing forward
        self.target_position = np.random.uniform(-1, 1)

        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # Left, Stay, Right

    def step(self, action):
        # Action: 0=left, 1=stay, 2=right
        if action == 0:
            self.servo_angle -= 0.05
        elif action == 2:
            self.servo_angle += 0.05

        # Simulate target movement
        self.target_position += np.random.uniform(-0.02, 0.02)
        self.target_position = np.clip(self.target_position, -1.0, 1.0)

        # Calculate reward
        error = abs(self.servo_angle - self.target_position)
        reward = -error  # closer = better

        done = error < 0.01  # perfect alignment
        obs = np.array([self.target_position - self.servo_angle], dtype=np.float32)

        return obs, reward, done, {}

    def reset(self):
        self.servo_angle = 0
        self.target_position = np.random.uniform(-1, 1)
        return np.array([self.target_position - self.servo_angle], dtype=np.float32)

    def render(self, mode='human'):
        print(f"Servo Angle: {self.servo_angle:.2f}, Target: {self.target_position:.2f}")

