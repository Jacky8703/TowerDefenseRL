import gymnasium as gym
import requests
import math
import io
from PIL import Image
import numpy as np
from gymnasium import spaces

url = "http://localhost:3000/"

class TowerDefenseWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}
    
    # define action_space and observation_space
    def __init__(self, render_mode="rgb_array"):
        self.render_mode = render_mode
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

        self.global_feature_count = 4 # game time, wave number, money, game over
        self.features_per_tower = 4+len(self.tower_types) # active, x, y, attack cooldown, one-hot encoding type
        self.tower_feature_count = self.max_towers * self.features_per_tower
        self.features_per_enemy = 5+len(self.game_info["waves"]["enemy_types"]) # active, x, y, health, path progress, one-hot encoding type
        self.enemy_feature_count = self.max_enemies * self.features_per_enemy

        total_features_count = self.global_feature_count + self.tower_feature_count + self.enemy_feature_count
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(total_features_count,),
            dtype=np.float32
        )

        self.tower_type_to_index = {tower["type"]: idx for idx, tower in enumerate(self.tower_types)}
        self.enemy_type_to_index = {enemy_type: idx for idx, enemy_type in enumerate(self.game_info["waves"]["enemy_types"])}

    # reset the environment and return the initial observation and info
    def reset(self):
        response = requests.post(url + "reset")
        if response.status_code != 200:
            raise ConnectionError(f"Failed to reset game: {response.text}")

        self.game_state = response.json()
        observation = self.__get_observation()
        info = self.__get_info()
        return observation, info

    # perform the action and return the new observation, reward, terminated, truncated, info
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
        info = self.__get_info()

        return observation, reward, terminated, truncated, info

    # returns the game state as an rgb array
    def render(self):
        black_frame = np.zeros((self.game_info["map"]["height"], self.game_info["map"]["width"], 3), dtype=np.uint8)
        if self.render_mode == "rgb_array":
            response = requests.get(url + "render")
            if response.status_code != 200:
                print(f"Error during render: {response.text}")
                return black_frame
            image_bytes = io.BytesIO(response.content)
            image = Image.open(image_bytes)
            rgb_array = np.array(image)
            return rgb_array
        return black_frame
    
    # just to comply with the interface
    def close(self):
        pass

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

    # encodes the self game state into a tensor of shape self.observation_space.shape
    def __get_observation(self):
        shape = self.observation_space.shape
        if shape is None:
            raise ValueError("Observation space shape is not defined")
        observation = np.zeros(shape, dtype=np.float32)

        # global features normalized
        observation[0] = self.game_state["gameTime"] / self.game_info["max_global_info"]["gameTime"]
        observation[1] = self.game_state["waveNumber"] / self.game_info["max_global_info"]["waveNumber"]
        observation[2] = self.game_state["money"] / self.game_info["max_global_info"]["money"]
        observation[3] = self.game_state["gameOver"]

        # tower features normalized
        for idx, tower in enumerate(self.game_state["towers"]):
            offset = self.global_feature_count + idx * self.features_per_tower
            observation[offset] = 1 # active
            observation[offset+1] = tower["position"]["x"] / self.game_info["map"]["width"] # normalized x
            observation[offset+2] = tower["position"]["y"] / self.game_info["map"]["height"] # normalized y
            observation[offset+3] = tower["attackCooldown"] / self.game_info["slower_tower_sample"]["attackCooldown"] # normalized attack cooldown
            observation[offset+4+self.tower_type_to_index[tower["type"]]] = 1 # one-hot encoding type

        # enemy features normalized
        for idx, enemy in enumerate(self.game_state["enemies"]):
            offset = self.global_feature_count + self.tower_feature_count + idx * self.features_per_enemy
            observation[offset] = 1 # active
            observation[offset+1] = enemy["position"]["x"] / self.game_info["map"]["width"] # normalized x
            observation[offset+2] = enemy["position"]["y"] / self.game_info["map"]["height"] # normalized y
            observation[offset+3] = enemy["currentHealth"] / enemy["fullHealth"] # normalized health
            observation[offset+4] = enemy["pathProgress"]
            observation[offset+5+self.enemy_type_to_index[enemy["type"]]] = 1 # one-hot encoding type

        return observation
    
    # additional info for debugging or logging, currently empty
    def __get_info(self):
        info = {}
        return info
    
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

    # counts how many path cells are in range of the tower
    def __count_path_cells_in_range(self, tower): # si potrebbe ottimizzare partendo dalla posizione della torre e calcolando il numero di celle in range
        count = 0
        for path_cell in self.game_info["map"]["path_cells"]:
            distance = math.sqrt((tower["position"]["x"] - path_cell["x"])**2 + (tower["position"]["y"] - path_cell["y"])**2)
            tower_index = self.tower_type_to_index[tower["type"]]
            if distance < self.game_info["towers"][tower_index]["range"]:
                count += 1
        return count

env = TowerDefenseWorldEnv()