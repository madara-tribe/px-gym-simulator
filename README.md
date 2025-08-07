# Abstract

This repository simulates a software and hardware system—such as a robotic platform—to evaluate its feasibility.

You can set up an environment for training using **OpenAI Gym**, and perform real-time testing through a **Pygame-based visual interface**. 

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
You can train with simulated environment by reinforcement learning agent through OpenAI Gym.

```bash
# when to train 
python3 train/train_rl.py
```

## About Pygame Test

You can test the trained model in a visual simulator built with Pygame.

you can Visualize its process of the performance in real time. 

```bash
# when to test on pygame
python3 test/test_with_pygame.py
```
