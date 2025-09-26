# TowerDefenseRL
Reinforcement learning agent for my [Tower Defense Game](https://github.com/Jacky8703/TowerDefenseGame).

## Features

- Integration with the Tower Defense Game (custom Gymnasium environment)
- Training script using Stable Baselines3 (maskable PPO algorithm)
- Tensorboard for monitoring training progress (with custom metrics)
- Model saving and loading
- Video recording of agent gameplay
- Script for replaying the best agent game actions from training (max waves reached)
- Random map selection during training for better generalization

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/Jacky8703/TowerDefenseRL.git
    cd TowerDefenseRL
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Clone and set up the [Tower Defense Game](https://github.com/Jacky8703/TowerDefenseGame) server as described in its README.

## Usage

The tower defense game server must be running before executing any scripts.

### Train an agent
1. Set ```hours_to_train``` in ```train.py```

2. Execute script (default map):
    ```bash 
    python train.py
    ```
    Or with random maps:
    ```bash
    python train.py --random-maps custom-maps.json
    ```
3. Monitor training progress via TensorBoard (or at the end of training):
    ```bash
    tensorboard --logdir ./logs/
    ```
4. The trained model will be saved in the `models/` directory.

In addition to the final model, a json file with the best agent performance (max waves reached) and a csv file with basic training metrics (reward, episode length and training time) will be saved.

In the `models/checkpoints/` directory, you will find periodic checkpoints of the model during training.

In the `models/videos/` directory, you will find videos of the agent's gameplay recorded at intervals during training.

In the `logs/` directory, a log file containing training metrics (visible via TensorBoard) will be created.

### Load a pre-trained model
1. If you want to continue the old training logs, add the `tb_log_name` argument to the `model.learn()` function in `train.py` with the corresponding tensorboard log name, e.g.:
    ```python
    model.learn(total_timesteps=training_steps, ..., tb_log_name="PPO_1")
    ```
2. Execute script specifying the model path:
    ```bash
    python train.py --load-model ./path/to/maskable_ppo_tower_defense.zip
    ```

### Replay best agent game (works only for the default map for now)
1. Execute the replay script specifying the json file with the best agent actions:
    ```bash
    python replay_actions.py --actions-file ./models/<date_time>/best_episode_actions.json
    ```
    Optionally, you can save the frames to a `best_frames` directory next to the actions file by adding the `--save-frames` argument (for future loading purposes).

2. If you have already saved the frames, you can load them directly by using the `--load-frames` argument with the path to the `best_frames` directory (much faster):
    ```bash
    python replay_actions.py --load-dir ./models/<date_time>/best_frames
    ```