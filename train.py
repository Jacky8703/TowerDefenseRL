import json
import gymnasium_env.envs  # ensure the custom environment is registered
import gymnasium as gym
import logging
import datetime
from stable_baselines3.common.callbacks import CheckpointCallback
from sb3_contrib import MaskablePPO
from gymnasium_env.wrappers.random_map_wrapper import RandomMapWrapper
from gymnasium_env.wrappers.wrap import wrap_env
from custom_callbacks.tensor_board_info import TensorboardInfoCallback
from custom_callbacks.save_agent_actions import SaveAgentActionsCallback
import argparse

hours_to_train = 8
video_number = 10 # number of videos to record during training

mean_time_fps = 330 # ~mean time/fps from tensor board, steps per second (obviously varies)
mean_episode_steps = 2500 # ~mean steps per episode from tensor board (also varies and it depends on the hours_to_train: more hours, better agent, longer episodes)

training_steps = round(mean_time_fps*hours_to_train*3600) # total number of training steps
episode_recording_gap = (training_steps/mean_episode_steps) // video_number  # one episode = one game

env_name = "gymnasium_env/TowerDefenseWorld-v0"

def main(load_model_path, random_maps_path):
    prefix = datetime.datetime.now().strftime("%d.%m.%Y_%H.%M")

    logging.basicConfig(
        filename="training.log", 
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    env = gym.make(env_name)
    env = wrap_env(env, episode_recording_gap, prefix)

    if random_maps_path:
        with open(random_maps_path, "r") as f:
            data = json.load(f)
        env.reset(seed=87) # set seed for reproducibility (same seed -> same map sequence)
        env = RandomMapWrapper(env, map_list=data)

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

        if load_model_path:
            logging.info(f"Loading model from: {load_model_path}")
            model = MaskablePPO.load(load_model_path, env, tensorboard_log="./logs/")
        else:
            model = MaskablePPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/")

        logging.info("Starting model training...")
        start = datetime.datetime.now()
        # do not reset the timestep number if loading a model
        model.learn(total_timesteps=training_steps, callback=[checkpoint_callback, tensorboard_info_callback, save_actions_callback], reset_num_timesteps=not load_model_path)
        logging.info(f"Model training completed in {(datetime.datetime.now() - start).total_seconds()/3600:.2f} hours ({hours_to_train} planned).")

        model.save(f"./models/{prefix}/maskable_ppo_tower_defense.zip")
        logging.info("Model saved.")

        best_performance_data = save_actions_callback.get_best_agent_performance()
        with open(f"./models/{prefix}/best_episode_actions.json", "w") as f:
            json.dump(best_performance_data, f)
        logging.info("Best episode actions saved.")
    except Exception as e:
        logging.error(f"An error occurred during training: {e}")
        raise e

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a MaskablePPO agent on the Tower Defense environment.")
    parser.add_argument("--load-model", help="Optional path to the model zip file.")
    parser.add_argument("--random-maps", help="Optional path to the custom maps JSON file for random map training.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    main(args.load_model, args.random_maps)