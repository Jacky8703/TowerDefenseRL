import gymnasium as gym
import requests
import math
import numpy as np
from gymnasium import spaces

url = "http://localhost:3000/"

class TowerDefenseWorldEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}
    
    # define action_space and observation_space
    def __init__(self):
        response = requests.get(url + "info")
        if response.status_code != 200:
            raise ConnectionError(f"Failed to get game info: {response.text}")

        self.game_info = response.json()
        self.action_types = self.game_info["actions"]
        self.tower_types = self.game_info["towers"]
        cell_size = self.game_info["map"]["cell_size"]
        map_horizontal_cells = int(self.game_info["map"]["width"] / cell_size)
        map_vertical_cells = int(self.game_info["map"]["height"] / cell_size)

        self.action_space = spaces.MultiDiscrete([len(self.action_types), len(self.tower_types), map_horizontal_cells, map_vertical_cells]) # action, tower type, x, y 

        self.max_towers = int(map_horizontal_cells * map_vertical_cells - self.game_info["map"]["path_length"] / cell_size)
        self.max_enemies = self.__calculate_total_enemies()

        global_feature_count = 4 # game time, wave number, money, game over
        tower_feature_count = self.max_towers * 5 # active, type, x, y, fire rate
        enemy_feature_count = self.max_enemies * 6 # active, type, health, x, y, path progress
        total_features = global_feature_count + tower_feature_count + enemy_feature_count

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(total_features,),
            dtype=np.float32
        )

    def reset(self):
        response = requests.post(url + "reset")
        if response.status_code != 200:
            raise ConnectionError(f"Failed to reset game: {response.text}")

        self.game_state = response.json()
        observation = self.__get_observation()
        info = {}
        return observation, info

    def step(self, action):
        action_index, tower_index, x, y = action
        game_action = self.action_types[action_index]
        if game_action["type"] == "BUILD_TOWER":
            game_action["towerType"] = self.tower_types[tower_index]["type"]
            game_action["position"]["x"] = self.game_info["map"]["cell_size"]/2 + self.game_info["map"]["cell_size"]*x
            game_action["position"]["y"] = self.game_info["map"]["cell_size"]/2 + self.game_info["map"]["cell_size"]*y

        response = requests.post(url + "step", json = game_action)
        if response.status_code != 200:
            print(f"Error during step: {response.text}")
            last_observation = self.__get_observation()
            return last_observation, 0, False, True, {}

        new_game_state = response.json()
        reward = self.__calculate_reward(new_game_state)
        self.game_state = new_game_state
        observation = self.__get_observation()
        terminated = new_game_state["gameOver"]
        truncated = False
        info = {}

        return observation, reward, terminated, truncated, info


    # Worst case (assuming enemies remain alive, max number of enemies per wave and the slower spawns last):
    # - Time between waves: T = wave delay + max enemies per wave * spawn delay
    # - Number of actual waves: N = slower enemy time to complete path / T
    # - Number of total enemies: = N * max enemies per wave
    def __calculate_total_enemies(self):
        wave_delay = self.game_info["waves"]["wave_delay"]
        wave_max_enemies = self.game_info["waves"]["max_enemies"]
        spawn_delay = self.game_info["waves"]["spawn_delay"]
        slower_enemy_time = self.game_info["map"]["path_length"] / self.game_info["waves"]["slower_enemy_sample"]["currentSpeed"]
        total_enemies = int(slower_enemy_time*wave_max_enemies/(wave_delay+spawn_delay*wave_max_enemies))
        if slower_enemy_time < wave_delay:
            total_enemies = wave_max_enemies
        return total_enemies

    # encodes the game state into a tensor of shape self.observation_space.shape
    def __get_observation(self):
        shape = self.observation_space.shape
        if shape is None:
            raise ValueError("Observation space shape is not defined")
        observation = np.zeros(shape, dtype=np.float32)
        # per andare avanti devi stabilire un max dei valori globali in modo da poter normalizzare tutto

    # calculate the rewards based on the new game state
    def __calculate_reward(self, new_game_state):
        reward = 0
        old_state = self.game_state
        # positive, killing enemies and completing waves
        reward += max(0, len(old_state["enemies"]) - len(new_game_state["enemies"]))
        if new_game_state["waveNumber"] > old_state["waveNumber"]:
            reward += new_game_state["waveNumber"]*2
        # neutral, building towers (rewarded based on coverage, penalized if no coverage)
        new_towers_count = len(new_game_state["towers"]) - len(old_state["towers"])
        if new_towers_count > 0:
            for i in range(new_towers_count):
                count = self.__count_path_cells_in_range(new_game_state["towers"][-i-1]) # new towers are at the end of the list
                reward += count*2
                if count == 0:
                    reward -= 30
        # negative, spending money too often and game over
        if new_game_state["money"] < old_state["money"]:
            reward -= 4
        if new_game_state["gameOver"]:
            reward -= 100

        return reward

    def __count_path_cells_in_range(self, tower):
        count = 0
        for path_cell in self.game_info["path_cells"]:
            distance = math.sqrt((tower["position"]["x"] - path_cell["x"])**2 + (tower["position"]["y"] - path_cell["y"])**2)
            if distance < tower["range"]:
                count += 1
        return count

env = TowerDefenseWorldEnv()
#o, r, t, tt, i = env.step([1,0,5,7])