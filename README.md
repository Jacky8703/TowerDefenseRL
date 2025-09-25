# TowerDefenseRL
Reinforcement learning agent for my [Tower Defense Game](https://github.com/Jacky8703/TowerDefenseGame).

## Features

- Integration with the Tower Defense Game environment
- Training and evaluation scripts
- Visualization tools for agent performance

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

Train an agent:
1. Set ```hours_to_train``` in ```train.py```

2. Execute script:
    ```bash 
    python train.py
    ```
3. Monitor training progress via TensorBoard (or at the end of training):
    ```bash
    tensorboard --logdir=./logs/
    ```
4. The trained model will be saved in the `models/` directory.