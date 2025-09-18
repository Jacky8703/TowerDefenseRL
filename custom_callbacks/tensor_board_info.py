from stable_baselines3.common.callbacks import BaseCallback

class TensorboardInfoCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        if self.locals["dones"][0]: # only at the end of an episode
            info = self.locals["infos"][0]
            if "wave_number" in info:
                self.logger.record("custom/wave_number", info["wave_number"])
            if "tower_counts" in info:
                for tower_type, count in info["tower_counts"].items():
                    self.logger.record(f"custom/{tower_type}_count", count)

        return True