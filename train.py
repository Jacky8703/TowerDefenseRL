import gymnasium_env.envs  # ensure the custom environment is registered
import gymnasium as gym
import logging
import datetime
from stable_baselines3.common.callbacks import CheckpointCallback
from sb3_contrib import MaskablePPO
from gymnasium_env.wrappers.wrap import wrap_env
from custom_callbacks.tensor_board_info import TensorboardInfoCallback

mean_time_fps = 320 # ~mean time/fps from tensor board, steps per second
mean_episode_steps = 1500 # ~mean steps per episode from tensor board

hours_to_play = 1
video_number = 10 # number of videos to record during training

training_steps = mean_time_fps*hours_to_play*3600 # total number of training steps
episode_recording_gap = (training_steps/mean_episode_steps) // video_number  # one episode = one game

env_name = "gymnasium_env/TowerDefenseWorld-v0"
prefix = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M")

logging.basicConfig(
    filename="training.log", 
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

env = gym.make(env_name)
env = wrap_env(env, episode_recording_gap, prefix)

# save 10 checkpoints
checkpoint_callback = CheckpointCallback(
  save_freq=training_steps//10,
  save_path=f"./checkpoints/{prefix}/",
  name_prefix="maskable_ppo_tower_defense",
)
# custom tensorboard callback to log wave number and tower counts
tensorboard_info_callback = TensorboardInfoCallback()

try:
    logging.info(f"--- Starting New Training Run ---")
    logging.info(f"Environment: {env_name}")
    logging.info(f"Total Timesteps: {training_steps} (~{training_steps*0.1/3600:.2f} hours of playing)")
    logging.info(f"Video Recording Period: {episode_recording_gap} games")

    model = MaskablePPO("MlpPolicy", env, verbose=1, tensorboard_log=f"./logs/")

    logging.info("Starting model training...")
    model.learn(total_timesteps=training_steps, callback=[checkpoint_callback, tensorboard_info_callback])
    logging.info("Model training completed.")

    model.save(f"./models/{prefix}_maskable_ppo_tower_defense")
    logging.info("Model saved.")
except Exception as e:
    logging.error(f"An error occurred during training: {e}")
    raise e