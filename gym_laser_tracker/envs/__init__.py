#import gymnasium as
import gym
from gym.envs.registration import register

register(
    id='LaserTracker-v0',
    entry_point='gym_laser_tracker.envs.laser_tracker_env:LaserTrackerEnv',
)

