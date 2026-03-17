import datetime
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.distributions.categorical import Categorical
from mappo_environment import Hyperparameters
from torch import nn
from torch import optim



class MAPPOActorCritic(nn.Module):
    def __init__(self, local_obs_dim: int, joint_obs_dim: int, action_space_size: int):
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(local_obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_space_size),
        )

        self.critic = nn.Sequential(
            nn.Linear(joint_obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def policy(self, local_obs):
        return self.actor(local_obs)

    def value(self, joint_obs):
        return self.critic(joint_obs).squeeze(-1)





class MAPPOUtils:
    @staticmethod
    def flatten_agent_obs(agent_obs):
        return np.asarray(agent_obs, dtype=np.float32).reshape(-1)

    @staticmethod
    def flatten_joint_obs(obs_tuple):
        return np.concatenate([MAPPOUtils.flatten_agent_obs(o) for o in obs_tuple], axis=0).astype(np.float32)

    @staticmethod
    def _lane_id(vehicle):
        li = getattr(vehicle, 'lane_index', None)
        if li is None:
            return None
        try:
            return li[2]
        except Exception:
            try:
                return li[-1]
            except Exception:
                return None

    @staticmethod
    def _front_distance(env, vehicle):
        if hasattr(env, 'road') and hasattr(env.road, 'neighbour_vehicles'):
            front, _ = env.road.neighbour_vehicles(vehicle)
            if front is None:
                return None
            d = float(front.position[0] - vehicle.position[0])
            return d if d >= 0 else None
        return None

    @staticmethod
    def team_reward(env, base_reward, prev_lanes):
        rewards = list(base_reward) if isinstance(base_reward, (tuple, list, np.ndarray)) else [float(base_reward)]
        team_reward = float(np.mean(rewards))

        lane_change_bonus = 0.01
        unsafe_dist_threshold = 12.0
        unsafe_dist_penalty = 0.3
        spacing_bonus = 0.3
        crash_penalty = 4.0

        lane_changes = 0
        crashes = 0
        for i, vehicle in enumerate(env.unwrapped.controlled_vehicles):
            new_lane = MAPPOUtils._lane_id(vehicle)
            if prev_lanes[i] is not None and new_lane is not None and new_lane != prev_lanes[i]:
                lane_changes += 1
            prev_lanes[i] = new_lane

            d_front = MAPPOUtils._front_distance(env.unwrapped, vehicle)
            if d_front is not None:
                if d_front < unsafe_dist_threshold:
                    closeness = (unsafe_dist_threshold - d_front) / unsafe_dist_threshold
                    team_reward -= unsafe_dist_penalty * closeness
                elif d_front > 18.0:
                    team_reward += spacing_bonus

            if getattr(vehicle, 'crashed', False):
                crashes += 1

        team_reward += lane_change_bonus * lane_changes
        team_reward -= crash_penalty * crashes
        return float(team_reward), prev_lanes, crashes

    @staticmethod
    def calculate_gae(rewards, values, gamma=0.99, lam=0.95):
        values_ext = np.append(values, 0.0)
        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values_ext[t + 1] - values_ext[t]
            gae = delta + gamma * lam * gae
            advantages[t] = gae
        returns = advantages + values
        return advantages, returns
    
    @staticmethod
    def rollout(model, env, max_steps=500):
        device = Hyperparameters.DEVICE
        obs, _ = env.reset()
        n_agents = len(obs)

        # Per-agent buffers for actor training
        local_obs_buffer = []
        joint_obs_buffer = []
        actions_buffer = []
        log_probs_buffer = []

        # Per-timestep buffers for critic / GAE
        timestep_rewards = []
        timestep_values = []

        prev_lanes = [MAPPOUtils._lane_id(v) for v in env.unwrapped.controlled_vehicles]
        ep_reward = 0.0
        steps = 0
        crashed_any = False

        for _ in range(max_steps):
            local_obs = [MAPPOUtils.flatten_agent_obs(o) for o in obs]
            joint_obs = MAPPOUtils.flatten_joint_obs(obs)
            joint_obs_tensor = torch.tensor(joint_obs, dtype=torch.float32, device=device).unsqueeze(0)

            actions = []
            log_probs = []

            # Actor acts from local observation
            for agent_obs in local_obs:
                local_obs_tensor = torch.tensor(agent_obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    logits = model.policy(local_obs_tensor)
                    dist = Categorical(logits=logits)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)

                actions.append(int(action.item()))
                log_probs.append(float(log_prob.item()))

            # Centralized critic: one value per timestep
            with torch.no_grad():
                value = float(model.value(joint_obs_tensor).item())

            next_obs, reward, terminated, truncated, _ = env.step(tuple(actions))
            team_reward, prev_lanes, crashes = MAPPOUtils.team_reward(env, reward, prev_lanes)
            crashed_any = crashed_any or (crashes > 0)

            if Hyperparameters.RENDER and env.render_mode == 'human':
                env.render()
                time.sleep(1 / 15)

            # Store timestep-level reward/value ONCE
            timestep_rewards.append(team_reward)
            timestep_values.append(value)

            # Store per-agent samples
            for agent_idx in range(n_agents):
                local_obs_buffer.append(local_obs[agent_idx])
                joint_obs_buffer.append(joint_obs)
                actions_buffer.append(actions[agent_idx])
                log_probs_buffer.append(log_probs[agent_idx])

            ep_reward += team_reward
            steps += 1
            obs = next_obs

            done = terminated or truncated
            if isinstance(done, (tuple, list, np.ndarray)):
                done = any(done)
            if done:
                break

        # GAE over timesteps, not flattened agents
        timestep_rewards = np.asarray(timestep_rewards, dtype=np.float32)
        timestep_values = np.asarray(timestep_values, dtype=np.float32)

        advantages_t, returns_t = MAPPOUtils.calculate_gae(
            timestep_rewards,
            timestep_values,
            gamma=Hyperparameters.GAMMA,
            lam=Hyperparameters.GAE_LAMBDA,
        )

        # Repeat timestep advantages/returns for each agent
        advantages = np.repeat(advantages_t, n_agents).astype(np.float32)
        returns = np.repeat(returns_t, n_agents).astype(np.float32)

        train_data = {
            'local_obs': np.asarray(local_obs_buffer, dtype=np.float32),
            'joint_obs': np.asarray(joint_obs_buffer, dtype=np.float32),
            'actions': np.asarray(actions_buffer, dtype=np.int64),
            'advantages': advantages,
            'returns': returns,
            'log_probs': np.asarray(log_probs_buffer, dtype=np.float32),
        }
        return train_data, ep_reward, steps, crashed_any


    @staticmethod
    def create_directory():
        now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        return f'{now}_highway_v0_mappo'

    @staticmethod
    def save_models(model, ep_rewards):
        model_folder = 'models'
        os.makedirs(model_folder, exist_ok=True)
        output_folder = os.path.join(model_folder, MAPPOUtils.create_directory())
        os.makedirs(output_folder, exist_ok=True)

        model_path = os.path.join(output_folder, 'mappo_actor_critic.pth')
        torch.save(model.state_dict(), model_path)

        plt.figure(figsize=(10, 5))
        plt.plot(ep_rewards, label='Episode team reward')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('MAPPO Training Progress')
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(output_folder, 'training_progress.png')
        plt.savefig(plot_path)
        plt.close()

        print('Saved model to:', model_path)
        print('Saved training plot to:', plot_path)
        return model_path, plot_path

    @staticmethod
    def load_model(model_path):
        model = MAPPOActorCritic(
            local_obs_dim=Hyperparameters.local_obs_dim(),
            joint_obs_dim=Hyperparameters.joint_obs_dim(),
            action_space_size=Hyperparameters.ACTION_SPACE,
        )
        state_dict = torch.load(model_path, map_location=Hyperparameters.DEVICE)
        model.load_state_dict(state_dict)
        model.eval()
        return model
    
    
class MAPPOTrainer:
    def __init__(
        self,
        actor_critic,
        policy_lr=3e-4,
        value_lr=1e-3,
        ppo_clip_val=0.2,
        target_kl_div=0.02,
        max_policy_train_iters=40,
        value_train_iters=60,
        entropy_coef=0.01,
    ):
        self.ac = actor_critic
        self.ppo_clip_val = ppo_clip_val
        self.target_kl_div = target_kl_div
        self.max_policy_train_iters = max_policy_train_iters
        self.value_train_iters = value_train_iters
        self.entropy_coef = entropy_coef

        self.policy_optim = optim.Adam(self.ac.actor.parameters(), lr=policy_lr)
        self.value_optim = optim.Adam(self.ac.critic.parameters(), lr=value_lr)

    def train_policy(self, local_obs, acts, old_log_probs, advantages):
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.max_policy_train_iters):
            self.policy_optim.zero_grad()
            logits = self.ac.policy(local_obs)
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(acts)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs)
            clipped_ratio = ratio.clamp(1 - self.ppo_clip_val, 1 + self.ppo_clip_val)
            loss_unclipped = ratio * advantages
            loss_clipped = clipped_ratio * advantages
            policy_loss = -torch.min(loss_unclipped, loss_clipped).mean() - self.entropy_coef * entropy

            policy_loss.backward()
            self.policy_optim.step()

            kl_div = (old_log_probs - new_log_probs).mean()
            if kl_div >= self.target_kl_div:
                break

    def train_value(self, joint_obs, returns):
        for _ in range(self.value_train_iters):
            self.value_optim.zero_grad()
            values = self.ac.value(joint_obs)
            value_loss = ((returns - values) ** 2).mean()
            value_loss.backward()
            self.value_optim.step()
