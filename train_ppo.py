import os
import gymnasium as gym
import highway_env  # registers envs

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

ENV_ID = "highway-v0"

# ---- Settings you can tweak safely ----
N_ENVS = 8                 # was 1; PPO learns smoother with 8
TOTAL_TIMESTEPS = 300_000  # keep your target
SEED = 0

def make_env(rank: int, seed: int = 0):
    def _init():
        env = gym.make(ENV_ID)  # IMPORTANT: no render_mode during training
        env.reset(seed=seed + rank)
        return env
    return _init

# ---- Build vectorized envs ----
train_env = DummyVecEnv([make_env(i, SEED) for i in range(N_ENVS)])
train_env = VecMonitor(train_env, filename=os.path.join("runs", "monitor_train.csv"))

eval_env = DummyVecEnv([make_env(0, SEED + 10_000)])
eval_env = VecMonitor(eval_env, filename=os.path.join("runs", "monitor_eval.csv"))

# ---- Quick sanity prints ----
tmp_env = gym.make(ENV_ID)
print("ENV_ID =", ENV_ID)
print("obs_space =", tmp_env.observation_space)
print("act_space =", tmp_env.action_space)
tmp_env.close()

# ---- Model ----
model = PPO(
    "MlpPolicy",
    train_env,
    verbose=1,
    tensorboard_log="runs",
    n_steps=2048,        # per env rollout length
    batch_size=256,
    learning_rate=3e-4,
    gamma=0.99,
)

# NOTE: EvalCallback eval_freq is in *environment steps* for VecEnv.
# With N_ENVS parallel envs, training advances N_ENVS steps per call.
# So we scale eval_freq so evaluation happens at a similar wall-clock cadence.
eval_cb = EvalCallback(
    eval_env,
    best_model_save_path="checkpoints",
    log_path="checkpoints",
    eval_freq=10_000 // N_ENVS,   # important change for VecEnv
    n_eval_episodes=10,
    deterministic=True,
    render=False,
)

ckpt_cb = CheckpointCallback(
    save_freq=50_000 // N_ENVS,
    save_path="checkpoints",
    name_prefix="ppo_highway",
)

model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[eval_cb, ckpt_cb])
model.save("ppo_highway_final")
print("Done. Saved to ppo_highway_final.zip")