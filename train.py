import gymnasium as gym
import logging
from stable_baselines3.common.callbacks import CheckpointCallback
from sb3_contrib import MaskablePPO
from gymnasium_env.wrappers.wrap import wrap_env

# one episode is one game, suppose average game waves = 10, 10 waves = ~300 seconds (27s*wave) and one step = 0.1s in game so 3000 steps per game/episode.
training_period_episodes = 30 
training_steps = 100000 # total number of training steps

env_name = "gymnasium_env/TowerDefenseWorld-v0"

logging.basicConfig(
    filename="training.log", 
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

env = gym.make(env_name)
env = wrap_env(env, training_period_episodes)

# save 10 checkpoints
checkpoint_callback = CheckpointCallback(
  save_freq=training_steps//10,
  save_path="./checkpoints/",
  name_prefix="maskable_ppo_tower_defense",
)

try:
    logging.info(f"--- Starting New Training Run ---")
    logging.info(f"Environment: {env_name}")
    logging.info(f"Total Timesteps: {training_steps} (~{training_steps*0.1/3600:.2f} hours of playing)")
    logging.info(f"Video Recording Period: {training_period_episodes} games")
    
    model = MaskablePPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/")

    logging.info("Starting model training...")
    model.learn(total_timesteps=training_steps, callback=checkpoint_callback)
    logging.info("Model training completed.")

    model.save("maskable_ppo_tower_defense")
    logging.info("Model saved.")
except Exception as e:
    logging.error(f"An error occurred during training: {e}")