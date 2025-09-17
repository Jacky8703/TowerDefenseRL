import gymnasium as gym

gym.register(
    id="gymnasium_env/TowerDefenseWorld-v0",
    entry_point="gymnasium_env.envs.tower_defense_world:TowerDefenseWorldEnv"
)