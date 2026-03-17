import numpy as np
import torch
from Single_Agent_Lane_Change_ppo_model import ActorCriticNetwork
from Single_Agent_Lane_Change_ppo_model import PPOTrainer
from Single_Agent_Lane_Change_ppo_model import PPOUtils
from Single_Agent_Lane_Change_environment import env
from Single_Agent_Lane_Change_environment import Hyperparameters


# print(Hyperparameters.all())
DEVICE, n_episodes, print_freq, policy_lr, value_lr, target_kl_div, max_policy_train_iters, value_train_iters, obs_space, action_space = Hyperparameters.all()[:10]

model = ActorCriticNetwork(obs_space, action_space)
model = model.to(DEVICE)

# Init Trainer
ppo = PPOTrainer(
    actor_critic=model,
    ppo_clip_val=0.2,
    target_kl_div=target_kl_div,
    train_iters=max_policy_train_iters,
    lr=policy_lr,
    value_coef=0.5,
    entropy_coef=0.001,
    max_grad_norm=0.5
)

print(env.config) #MG added print for debug

print("lane_change_reward =", env.config.get("lane_change_reward"))
print("high_speed_reward =", env.config.get("high_speed_reward"))
print("right_lane_reward =", env.config.get("right_lane_reward"))
print("collision_reward =", env.config.get("collision_reward"))

# Training loop
ep_rewards = []
ep_lengths = []      # NEW
ep_crashed = []      # NEW (bools)

for episode_idx in range(n_episodes):
  # Perform rollout
  train_data, reward,ep_len,crashed = PPOUtils.rollout(model, env)
  ep_rewards.append(reward)
  ep_lengths.append(ep_len)      # NEW
  ep_crashed.append(crashed)     # NEW

  # Shuffle
  permute_idxs = np.random.permutation(len(train_data[0]))

  # Policy data
  obs = torch.tensor(train_data[0][permute_idxs],
                     dtype=torch.float32, device=DEVICE)
  acts = torch.tensor(train_data[1][permute_idxs],
                      dtype=torch.int32, device=DEVICE)
  gaes = torch.tensor(train_data[3][permute_idxs],
                      dtype=torch.float32, device=DEVICE)
  act_log_probs = torch.tensor(train_data[4][permute_idxs],
                               dtype=torch.float32, device=DEVICE)

  # Value data
  returns = PPOUtils.discount_rewards(train_data[2])[permute_idxs]
  returns = torch.tensor(returns, dtype=torch.float32, device=DEVICE)
  
  unique, counts = torch.unique(acts, return_counts=True)

  # Train model
  ppo.train(obs, acts, act_log_probs, returns, gaes)

  if (episode_idx + 1) % print_freq == 0:
    print('=========================================')
    print('Episode {} | Avg Reward {:.1f}'.format(
        episode_idx + 1, np.mean(ep_rewards[-print_freq:])))
    print('Episode {} | Avg Length {:.1f}'.format(
        episode_idx + 1, np.mean(ep_lengths[-print_freq:])))
    print('Episode {} | Crash Rate {:.2f}'.format(
        episode_idx + 1, np.mean(ep_crashed[-print_freq:])))
    print('=========================================')

# Save models and the reward plot
PPOUtils.save_models(model, ep_rewards, ppo)

