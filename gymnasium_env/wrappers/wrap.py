from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo, Autoreset
import datetime

video_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M")

def wrap_env(env, episode_recording_gap):
    env = RecordEpisodeStatistics(env)
    env = RecordVideo(env, video_folder=f"videos/{video_id}", name_prefix="training", episode_trigger=lambda e: e % episode_recording_gap == 0)
    env = Autoreset(env)
    return env