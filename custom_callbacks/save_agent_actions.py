from stable_baselines3.common.callbacks import BaseCallback
from copy import deepcopy

class SaveAgentActionsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.best_agent_performance = {
            "game_time": -1,
            "wave_number": -1,
            "actions": []
        }

    def _on_step(self) -> bool:
        if self.locals["dones"][0]: # only at the end of an episode
            info = self.locals["infos"][0]
            if "game_time" in info and "wave_number" in info and "episode_actions" in info:
                game_time = info["game_time"]
                wave_number = info["wave_number"]
                episode_actions = info["episode_actions"]
                # check if this episode is better
                if wave_number > self.best_agent_performance["wave_number"]:
                    self.best_agent_performance["game_time"] = game_time
                    self.best_agent_performance["wave_number"] = wave_number
                    self.best_agent_performance["actions"] = deepcopy(episode_actions)

        return True

    def get_best_agent_performance(self):
        return self.best_agent_performance