import gymnasium as gym
import numpy as np
import requests

url = "http://localhost:3000/"

class RandomMapWrapper(gym.Wrapper):
    def __init__(self, env, map_list: list[dict]):
        super().__init__(env)
        self.map_list = map_list

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        # set the seed for reproducibility (same seed -> same map sequence)
        obs, info = self.env.reset(seed=seed, options=options)
        # select a random index and set the map in the server
        selected_map_index = self.env.unwrapped.np_random.integers(0, len(self.map_list))
        response = requests.post(url + "set-map", json=self.map_list[selected_map_index]["waypoints"])
        if response.status_code != 200:
            print(f"Error setting map: {response.text}")

        return obs, info