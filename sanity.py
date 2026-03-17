import gymnasium as gym
import highway_env  # registers the envs

env = gym.make("highway-v0", render_mode=None)
obs, info = env.reset()
print("OK — env created. obs shape:", obs.shape)
env.close()