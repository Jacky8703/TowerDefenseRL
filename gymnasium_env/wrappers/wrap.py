from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo, Autoreset

def wrap_env(env, training_period):
    env = RecordEpisodeStatistics(env)
    env = RecordVideo(env, video_folder="videos", name_prefix="training", episode_trigger=lambda e: e % training_period == 0)
    env = Autoreset(env)
    return env