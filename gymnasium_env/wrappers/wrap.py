from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo, Autoreset

def wrap_env(env, episode_recording_gap):
    env = RecordEpisodeStatistics(env)
    env = RecordVideo(env, video_folder="videos", name_prefix="training", episode_trigger=lambda e: e % episode_recording_gap == 0)
    env = Autoreset(env)
    return env