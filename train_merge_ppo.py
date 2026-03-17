import gymnasium as gym
import highway_env  # registers envs
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback

ENV_ID = "merge-v0"

def make_env():
    def _init():
        env = gym.make(ENV_ID)
        return env
    return _init

train_env = VecMonitor(DummyVecEnv([make_env()]))
eval_env  = VecMonitor(DummyVecEnv([make_env()]))

model = PPO(
    "MlpPolicy",
    train_env,
    verbose=1,
    tensorboard_log="runs",
    n_steps=2048,
    batch_size=256,
    learning_rate=3e-4,
    gamma=0.99,
)

eval_cb = EvalCallback(
    eval_env,
    best_model_save_path="checkpoints_merge",
    log_path="checkpoints_merge",
    eval_freq=10_000,
    n_eval_episodes=10,
    deterministic=True,
)

model.learn(total_timesteps=300_000, callback=eval_cb)
model.save("ppo_merge_final")
print("Done. Saved to ppo_merge_final.zip")