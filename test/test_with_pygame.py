import pygame
import numpy as np
import random
import time
import gym
import gym_laser_tracker
from stable_baselines3 import PPO
import cv2

# Constants
WIDTH, HEIGHT = 640, 480
MARGIN_OF_ERROR = 5
SUDDEN_SWIFT = True
MAX_STEPS = 7
FPS = 1.5
SUCCESS_TARGET = 10

# Load model and environment
model = PPO.load("ppo_laser_tracker")
env = gym.make("LaserTracker-v0")

# Pygame init
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Laser Tracker - Test")
font = pygame.font.SysFont(None, 24)
clock = pygame.time.Clock()

# Video writer setup
video_filename = "tracking_result.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter(video_filename, fourcc, 10, (WIDTH, HEIGHT))

def angle_to_pos(angle):
    return int(angle / 180 * WIDTH)

def draw_state(screen, servo_angle, target_angle, step, reward, done, success_count):
    screen.fill((255, 255, 255))
    # draw laser(red) and label
    laser_x = angle_to_pos(servo_angle)
    pygame.draw.circle(screen, (255, 0, 0), (angle_to_pos(servo_angle), HEIGHT // 2), 10)
    laser_label = font.render("Laser", True, (255, 0, 0))
    screen.blit(laser_label, (laser_x - 20, HEIGHT // 2 - 25))

    # Draw target(green) and label
    target_x = angle_to_pos(target_angle)
    pygame.draw.circle(screen, (0, 255, 0), (target_x, HEIGHT // 2), 10)
    target_label = font.render("Target", True, (0, 128, 0))
    screen.blit(target_label, (target_x - 25, HEIGHT // 2 + 15))

    # Info
    info = f"Step: {step} | Reward: {reward:.2f} | Success: {success_count}"
    text = font.render(info, True, (0, 0, 0))
    screen.blit(text, (20, 20))
    pygame.display.flip()
    
    # Save to video
    frame = pygame.surfarray.array3d(screen)
    frame = np.rot90(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    video_writer.write(frame)

def run_episode(success_count):
    obs = env.reset()
    servo_angle = 90
    target_angle = env.target_angle
    step = 0
    done = False
    success = False
    
    # Draw initial (reset) state
    draw_state(screen, servo_angle, target_angle, step, 0.0, False, success_count)
    time.sleep(0.5)
    
    while not done:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                video_writer.release()
                exit()

        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        servo_angle = env.servo_angle
        target_angle = env.target_angle
        step += 1

        if abs(target_angle - servo_angle) <= MARGIN_OF_ERROR:
            success = True

        draw_state(screen, servo_angle, target_angle, step, reward, done, success_count)

    if success:
        success_count += 1
    return success_count

def main():
    success_count = 0

    while success_count < SUCCESS_TARGET:
        success_count = run_episode(success_count)
        time.sleep(1)

    # Game cleared
    screen.fill((0, 255, 0))
    cleared_text = font.render("Game Cleared!", True, (0, 0, 0))
    screen.blit(cleared_text, (WIDTH // 2 - 80, HEIGHT // 2))
    pygame.display.flip()

    for _ in range(30):
        frame = pygame.surfarray.array3d(screen)
        frame = np.rot90(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame)
        time.sleep(1 / 10)

    video_writer.release()
    pygame.quit()

if __name__ == "__main__":
    main()

