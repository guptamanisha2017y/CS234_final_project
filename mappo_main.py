import numpy as np
import torch
from mappo_utils import MAPPOTrainer
from mappo_environment import make_env
from mappo_environment import Hyperparameters
from mappo_utils import MAPPOActorCritic
from mappo_utils import MAPPOUtils


def main():
    env = make_env()
    device = Hyperparameters.DEVICE

    model = MAPPOActorCritic(
        local_obs_dim=Hyperparameters.local_obs_dim(),
        joint_obs_dim=Hyperparameters.joint_obs_dim(),
        action_space_size=Hyperparameters.ACTION_SPACE,
    ).to(device)

    trainer = MAPPOTrainer(
        model,
        policy_lr=Hyperparameters.POLICY_LR,
        value_lr=Hyperparameters.VALUE_LR,
        ppo_clip_val=Hyperparameters.PPO_CLIP,
        target_kl_div=Hyperparameters.TARGET_KL_DIV,
        max_policy_train_iters=Hyperparameters.MAX_POLICY_TRAIN_ITERS,
        value_train_iters=Hyperparameters.VALUE_TRAIN_ITERS,
        entropy_coef=Hyperparameters.ENTROPY_COEF,
    )

    ep_rewards, ep_lengths, ep_crashed = [], [], []

    for episode_idx in range(Hyperparameters.N_EPISODES):
        train_data, reward, ep_len, crashed = MAPPOUtils.rollout(
            model,
            env,
            max_steps=Hyperparameters.MAX_STEPS,
        )

        ep_rewards.append(reward)
        ep_lengths.append(ep_len)
        ep_crashed.append(int(crashed))

        perm = np.random.permutation(len(train_data['local_obs']))
        local_obs = torch.tensor(train_data['local_obs'][perm], dtype=torch.float32, device=device)
        joint_obs = torch.tensor(train_data['joint_obs'][perm], dtype=torch.float32, device=device)
        acts = torch.tensor(train_data['actions'][perm], dtype=torch.int64, device=device)
        advantages = torch.tensor(train_data['advantages'][perm], dtype=torch.float32, device=device)
        returns = torch.tensor(train_data['returns'][perm], dtype=torch.float32, device=device)
        log_probs = torch.tensor(train_data['log_probs'][perm], dtype=torch.float32, device=device)

        trainer.train_policy(local_obs, acts, log_probs, advantages)
        trainer.train_value(joint_obs, returns)

        if (episode_idx + 1) % Hyperparameters.PRINT_FREQ == 0:
            print('=========================================')
            print(f'Episode {episode_idx + 1} | Avg Team Reward {np.mean(ep_rewards[-Hyperparameters.PRINT_FREQ:]):.2f}')
            print(f'Episode {episode_idx + 1} | Avg Length {np.mean(ep_lengths[-Hyperparameters.PRINT_FREQ:]):.2f}')
            print(f'Episode {episode_idx + 1} | Crash Rate {np.mean(ep_crashed[-Hyperparameters.PRINT_FREQ:]):.2f}')
            print('=========================================')

    MAPPOUtils.save_models(model, ep_rewards)
    env.close()


if __name__ == '__main__':
    main()
