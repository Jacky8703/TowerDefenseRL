from gymnasium.wrappers import RecordVideo, Autoreset
from stable_baselines3.common.monitor import Monitor

def wrap_env(env, episode_recording_gap, prefix):
    env = Monitor(env, f"./models/{prefix}/monitor.csv")
    env = RecordVideo(env, video_folder=f"./models/{prefix}/videos/", name_prefix="training", episode_trigger=lambda e: e % episode_recording_gap == 0)
    env = Autoreset(env)
    return env