import json
import gymnasium_env.envs  # ensure the custom environment is registered
import gymnasium as gym
import logging
import datetime
from stable_baselines3.common.callbacks import CheckpointCallback
from sb3_contrib import MaskablePPO
from gymnasium_env.wrappers.wrap import wrap_env
from custom_callbacks.tensor_board_info import TensorboardInfoCallback
from custom_callbacks.save_agent_actions import SaveAgentActionsCallback

hours_to_play = 0.5
video_number = 10 # number of videos to record during training

mean_time_fps = 220 # ~mean time/fps from tensor board, steps per second (obviously varies)
mean_episode_steps = 700 # ~mean steps per episode from tensor board (also varies and it depends on the hours_to_play: more hours, better agent, longer episodes)

training_steps = round(mean_time_fps*hours_to_play*3600) # total number of training steps
episode_recording_gap = (training_steps/mean_episode_steps) // video_number  # one episode = one game

env_name = "gymnasium_env/TowerDefenseWorld-v0"
prefix = datetime.datetime.now().strftime("%d.%m.%Y_%H.%M")

logging.basicConfig(
    filename="training.log", 
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

env = gym.make(env_name)
env = wrap_env(env, episode_recording_gap, prefix)

# save 3 checkpoints
checkpoint_callback = CheckpointCallback(
  save_freq=training_steps//3,
  save_path=f"./models/{prefix}/checkpoints/",
  name_prefix="maskable_ppo_tower_defense",
)
# custom tensorboard callback to log wave number and tower counts
tensorboard_info_callback = TensorboardInfoCallback()
# custom callback to save best agent performance
save_actions_callback = SaveAgentActionsCallback()

try:
    logging.info(f"--- Starting New Training Run ---")
    logging.info(f"Environment: {env_name}")
    logging.info(f"Total Timesteps: {training_steps} (~{training_steps*0.1/3600:.2f} hours of playing)")
    logging.info(f"Video Recording Period: {episode_recording_gap} games")

    model = MaskablePPO("MlpPolicy", env, verbose=1, tensorboard_log=f"./logs/")

    logging.info("Starting model training...")
    start = datetime.datetime.now()
    model.learn(total_timesteps=training_steps, callback=[checkpoint_callback, tensorboard_info_callback, save_actions_callback])
    logging.info(f"Model training completed in {(datetime.datetime.now() - start).total_seconds()/3600:.2f} hours ({hours_to_play} planned).")

    model.save(f"./models/{prefix}/maskable_ppo_tower_defense.zip")
    logging.info("Model saved.")

    best_performance_data = save_actions_callback.get_best_agent_performance()
    with open(f"./models/{prefix}/best_episode_actions.json", "w") as f:
        json.dump(best_performance_data, f, indent=2)
    logging.info("Best episode actions saved.")
except Exception as e:
    logging.error(f"An error occurred during training: {e}")
    raise e