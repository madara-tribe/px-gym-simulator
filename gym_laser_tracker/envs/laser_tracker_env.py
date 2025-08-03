import gym
from gym import spaces
import numpy as np
import random
import matplotlib.pyplot as plt

class LaserTrackerEnv(gym.Env):
    def __init__(self):
        super(LaserTrackerEnv, self).__init__()

        # Observation: relative angle between target and laser (in degrees)
        self.observation_space = spaces.Box(low=-90, high=90, shape=(1,), dtype=np.float32)

        # Action: delta angle to move the laser (in degrees)
        self.action_space = spaces.Discrete(3)  # 0: left, 1: stay, 2: right

        # Internal state
        self.servo_angle = 90  # Initial servo angle (center)
        self.target_angle = self._generate_target_angle()

        self.max_steps = 50
        self.current_step = 0

        plt.ion()
        self.fig, self.ax = plt.subplots()

    def reset(self):
        self.servo_angle = 90  # Reset to center
        self.target_angle = self._generate_target_angle()
        self.current_step = 0
        return np.array([self._get_relative_angle()], dtype=np.float32)

    def step(self, action):
        self.current_step += 1

        # Simulate px3 - move servo angle based on action
        if action == 0:
            self.servo_angle = max(0, self.servo_angle - 5)
        elif action == 2:
            self.servo_angle = min(180, self.servo_angle + 5)

        # Observation is angle difference from target
        rel_angle = self._get_relative_angle()

        # Reward: the closer the servo to target, the higher the reward
        reward = -abs(rel_angle) / 90  # normalized between -1 and 0
        done = abs(rel_angle) < 3 or self.current_step >= self.max_steps

        if abs(rel_angle) < 3:
            reward += 1  # bonus for hitting the target

        return np.array([rel_angle], dtype=np.float32), reward, done, {}

    def render(self, mode="human"):
        self.ax.clear()
        self.ax.set_xlim(0, 180)
        self.ax.set_ylim(0, 1)
        self.ax.set_title(f"Step {self.current_step}")
        self.ax.plot(self.servo_angle, 0.5, 'ro', label='Laser')
        self.ax.plot(self.target_angle, 0.5, 'gx', label='Target')
        self.ax.legend()
        plt.pause(0.01)

    def _get_relative_angle(self):
        return float(self.target_angle - self.servo_angle)

    def _generate_target_angle(self):
        return random.randint(30, 150)  # Avoid edges for realism
