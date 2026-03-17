import gymnasium as gym
import highway_env
from stable_baselines3 import PPO
import numpy as np

ENV_ID = "merge-v0"

def run_episode(env, model, deterministic=True, render=False):
    obs, info = env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    collision = False

    while not done:
        if render:
            env.render()
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1
        done = terminated or truncated

        # Many highway-env envs expose a crash flag via vehicle.crashed
        try:
            collision = collision or bool(env.unwrapped.vehicle.crashed)
        except Exception:
            pass

    # “Merged” heuristic: if the episode lasted a while without crash, call it a success.
    # We can refine this once we inspect lane_index.
    success = (not collision) and (steps > 50)

    return total_reward, steps, collision, success

def main():
    model = PPO.load("ppo_merge_final")
    env = gym.make(ENV_ID, render_mode=None)

    N = 20
    rewards, lengths = [], []
    collisions, successes = 0, 0

    for _ in range(N):
        r, L, col, succ = run_episode(env, model, render=False)
        rewards.append(r); lengths.append(L)
        collisions += int(col)
        successes += int(succ)

    print(f"Eval over {N} episodes:")
    print(f"  avg reward:  {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"  avg length:  {np.mean(lengths):.1f}")
    print(f"  collisions:  {collisions}/{N}  ({collisions/N:.0%})")
    print(f"  successes:   {successes}/{N}  ({successes/N:.0%})")

    env.close()

    # Render a short demo
    demo_env = gym.make(ENV_ID, render_mode="human")
    for _ in range(3):
        run_episode(demo_env, model, render=True)
    demo_env.close()

if __name__ == "__main__":
    main()