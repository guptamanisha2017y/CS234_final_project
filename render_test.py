import gymnasium as gym
import highway_env

env = gym.make("highway-v0", render_mode="human")
obs, info = env.reset()

for _ in range(300):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()