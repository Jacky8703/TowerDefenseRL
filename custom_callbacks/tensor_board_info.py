from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class TensorboardInfoCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_wave_numbers = []
        self.episode_tower_counts = {}

    def _on_step(self) -> bool:
        if self.locals["dones"][0]: # only at the end of an episode
            info = self.locals["infos"][0]
            if "wave_number" in info:
                self.episode_wave_numbers.append(info["wave_number"])
            if "tower_counts" in info:
                for tower_type, count in info["tower_counts"].items():
                    if tower_type not in self.episode_tower_counts:
                        self.episode_tower_counts[tower_type] = [] # initialize list if not present
                    self.episode_tower_counts[tower_type].append(count)
        return True
    
    def _on_rollout_end(self) -> None: # default rollout is 2048 steps, so ~205 seconds in game time
        # log the mean wave number if we have data
        if len(self.episode_wave_numbers) > 0:
            mean_wave_number = np.mean(self.episode_wave_numbers)
            self.logger.record("rollout/custom/ep_wave_number_mean", mean_wave_number)
            self.episode_wave_numbers.clear()
        # log the mean tower counts for each type
        for tower_type, counts in self.episode_tower_counts.items():
            if len(counts) > 0:
                mean_count = np.mean(counts)
                self.logger.record(f"rollout/custom/ep_{tower_type}_count_mean", mean_count)
                counts.clear()