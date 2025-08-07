# Abstract

This repository simulates a software and hardware system—such as a robotic platform—to evaluate its feasibility.

You can set up an environment for training using ***OpenAI Gym***, and perform real-time testing through a Pygame-based visual interface.

The main algorithm used is reinforcement learning. You can customize the approach depending on your specific use case.

## Library Versions

| Library          | Version       |
|------------------|----------------|
| Python           | 3.8+           |
| gym              | ≥ 0.26         |
| stable-baselines3 | ≥ 2.0         |
| pygame           | ≥ 2.1          |

# simulaor

## About Gym Training
You can train a reinforcement learning agent using OpenAI Gym and Stable-Baselines3.
Environment: LaserTrackerEnv simulates a servo-laser mounted system trying to follow a moving target.
Training: Implemented via PPO algorithm (train_rl.py)
Metrics: Training logs can be visualized in TensorBoard
Reset behavior: After each episode, the servo angle and target position are re-initialized

```bash
# when to train 
python3 train/train_rl.py
```

## About Pygame Test

You can test the trained model in a visual simulator built with Pygame.
Visualizes the tracking process of the RL agent in real time
Target moves with configurable behavior (including sudden direction shifts)
Game ends when the agent consistently aligns the laser with the target
Success is measured based on angle difference (margin of error)
```bash
# when to test on pygame
python3 test/test_with_pygame.py
```
