import os
import gym
import gym_laser_tracker
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback


class LaserTrackerTrainer:
    def __init__(self,
                 model_path="ppo_laser_tracker",
                 checkpoint_dir="checkpoints",
                 best_model_dir="best_model",
                 log_path="logs/ppo_laser_tracker_tensorboard/",
                 total_timesteps=30000,
                 checkpoint_freq=10000):
        self.model_path = model_path
        self.model_file = f"{model_path}.zip"
        self.checkpoint_dir = checkpoint_dir
        self.best_model_dir = best_model_dir
        self.log_path = log_path
        self.total_timesteps = total_timesteps
        self.checkpoint_freq = checkpoint_freq

        self.env = gym.make("LaserTracker-v0")
        self.eval_env = gym.make("LaserTracker-v0")
        self.model = self._load_or_create_model()
        self._setup_logger()
        self._prepare_directories()

    def _prepare_directories(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.best_model_dir, exist_ok=True)

    def _load_or_create_model(self):
        if os.path.exists(self.model_file):
            print(f"Loading existing model from {self.model_file}")
            return PPO.load(self.model_path, env=self.env)
        else:
            print("No existing model found. Creating new model.")
            return PPO("MlpPolicy", self.env, verbose=1)

    def _setup_logger(self):
        logger = configure(self.log_path, ["stdout", "tensorboard"])
        self.model.set_logger(logger)

    def _get_eval_callback(self):
        """
        # Evaluation meaning in TensorBoard:
        # - ep_rew_mean: Mean episode reward (important in this case; reflects tracking performance)
        # - ep_len_mean: Mean episode length (not relevant in this case; fixed max steps)
        """
        return EvalCallback(
            self.eval_env,
            best_model_save_path=self.best_model_dir,
            log_path=self.log_path,
            eval_freq=self.checkpoint_freq // 2,
            deterministic=True,
            render=False,
            n_eval_episodes=5
        )

    def train(self):
        timesteps_done = 0
        eval_callback = self._get_eval_callback()

        while timesteps_done < self.total_timesteps:
            next_chunk = min(self.checkpoint_freq, self.total_timesteps - timesteps_done)
            self.model.learn(
                total_timesteps=next_chunk,
                reset_num_timesteps=False,
                callback=eval_callback
            )
            timesteps_done += next_chunk

            checkpoint_path = os.path.join(self.checkpoint_dir, f"{self.model_path}_{timesteps_done}.zip")
            self.model.save(checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

        self.model.save(self.model_path)
        print(f"Final model saved to {self.model_path}.zip")
        self.env.close()
        self.eval_env.close()


if __name__ == "__main__":
    trainer = LaserTrackerTrainer()
    trainer.train()
