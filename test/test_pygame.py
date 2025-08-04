import pygame
import numpy as np
import cv2
from stable_baselines3 import PPO
import gym
import gym_laser_tracker

# --- Config ---
WIDTH, HEIGHT = 640, 480
FPS = 30
EPISODES = 3
STEPS = 20
SAVE_VIDEO = True

# --- Pygame setup ---
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Laser Tracker RL - Minimal with Video")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)
BLACK, RED, GREEN, WHITE = (0, 0, 0), (255, 0, 0), (0, 255, 0), (255, 255, 255)

# --- Env & model ---
env = gym.make("LaserTracker-v0")
model = PPO.load("ppo_laser_tracker")

# --- Video writer ---
if SAVE_VIDEO:
    video_writer = cv2.VideoWriter(
        "laser_tracking_rl_minimal.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (WIDTH, HEIGHT)
    )

# --- Draw function ---
def draw(relative_angle, servo_angle, target_angle):
    screen.fill(BLACK)
    cx, cy = WIDTH // 2, HEIGHT // 2
    offset = int((target_angle - 90) * (WIDTH / 180))
    pygame.draw.circle(screen, RED, (cx + offset, cy), 10)

    angle_rad = np.deg2rad(servo_angle - 90)
    lx = int(cx + 300 * np.sin(angle_rad))
    ly = int(cy - 300 * np.cos(angle_rad))
    pygame.draw.line(screen, GREEN, (cx, cy), (lx, ly), 3)

    info = font.render(f"Tgt: {target_angle:.1f}°  Servo: {servo_angle:.1f}°  Err: {relative_angle[0]:.1f}°", True, WHITE)
    screen.blit(info, (20, 20))
    pygame.display.flip()

    if SAVE_VIDEO:
        frame = pygame.surfarray.array3d(screen)
        video_frame = cv2.cvtColor(np.transpose(frame, (1, 0, 2)), cv2.COLOR_RGB2BGR)
        video_writer.write(video_frame)

# --- Main loop ---
for ep in range(EPISODES):
    obs = env.reset()
    for step in range(STEPS):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)

        draw(obs, env.servo_angle, env.target_angle)
        clock.tick(FPS)

        if done:
            break

env.close()
if SAVE_VIDEO:
    video_writer.release()
pygame.quit()

