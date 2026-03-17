import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.distributions.categorical import Categorical
from Single_Agent_Lane_Change_environment import env
from Single_Agent_Lane_Change_environment import Hyperparameters
from torch import nn
from torch import optim


class PPOTrainer:
    def __init__(
        self,
        actor_critic,
        ppo_clip_val=0.2,
        target_kl_div=0.01,
        train_iters=5,
        lr=1e-4,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
    ):
        self.ac = actor_critic
        self.ppo_clip_val = ppo_clip_val
        self.target_kl_div = target_kl_div
        self.train_iters = train_iters
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        self.optim = optim.Adam(self.ac.parameters(), lr=lr)

    def train(self, obs, acts, old_log_probs, returns, gaes):
        gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)

        for _ in range(self.train_iters):
            self.optim.zero_grad()

            logits = self.ac.policy(obs)
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(acts)
            entropy = dist.entropy().mean()

            values = self.ac.value(obs).squeeze(-1)

            ratio = torch.exp(new_log_probs - old_log_probs)
            clipped_ratio = ratio.clamp(1 - self.ppo_clip_val, 1 + self.ppo_clip_val)

            policy_loss_1 = ratio * gaes
            policy_loss_2 = clipped_ratio * gaes
            policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

            value_loss = ((returns - values) ** 2).mean()

            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
            self.optim.step()

            kl_div = (old_log_probs - new_log_probs).mean().item()
            if kl_div >= self.target_kl_div:
                break


class ActorCriticNetwork(nn.Module):
    def __init__(self, obs_space_size, action_space_size):
        super().__init__()

        self.shared_layers = nn.Sequential(
            nn.Linear(obs_space_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        self.policy_layers = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_space_size)
        )

        self.value_layers = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def value(self, obs):
        z = self.shared_layers(obs)
        return self.value_layers(z)

    def policy(self, obs):
        z = self.shared_layers(obs)
        return self.policy_layers(z)

    def forward(self, obs):
        z = self.shared_layers(obs)
        policy_logits = self.policy_layers(z)
        value = self.value_layers(z)
        return policy_logits, value


action_type, obs_type = Hyperparameters.all()[10:12]
DEVICE = Hyperparameters.all()[0]


class PPOUtils:
    @staticmethod
    def rollout(model, env, max_steps=1000):
        """
        Performs a single rollout.
        Returns:
            train_data = [obs, act, reward, gaes, act_log_probs]
            ep_reward
            steps
            crashed
        """
        train_data = [[], [], [], [], []]  # obs, act, reward, values, act_log_probs

        seed = int(np.random.randint(20))
        obs, _ = env.reset(seed=seed, options={})
        obs = np.array(obs, dtype=np.float32)
        Hyperparameters.OBS_SPACE = obs.shape[0]

        base_env = env.unwrapped

        def _lane_id(vehicle):
            li = getattr(vehicle, "lane_index", None)
            if li is None:
                return None
            try:
                return li[2]
            except Exception:
                try:
                    return li[-1]
                except Exception:
                    return None

        def _front_rear_in_lane(base_env, lane_id):
            ego = base_env.vehicle
            ego_x = float(ego.position[0])

            front_vehicle = None
            rear_vehicle = None
            min_front_dist = float("inf")
            min_rear_dist = float("inf")

            for veh in base_env.road.vehicles:
                if veh is ego:
                    continue

                veh_lane = _lane_id(veh)
                if veh_lane != lane_id:
                    continue

                dx = float(veh.position[0] - ego_x)

                if dx >= 0 and dx < min_front_dist:
                    min_front_dist = dx
                    front_vehicle = veh
                elif dx < 0 and abs(dx) < min_rear_dist:
                    min_rear_dist = abs(dx)
                    rear_vehicle = veh

            return front_vehicle, rear_vehicle

        def _is_safe_lane(base_env, target_lane, min_front_gap=15.0, min_rear_gap=12.0):
            ego = base_env.vehicle
            ego_x = float(ego.position[0])

            front_vehicle, rear_vehicle = _front_rear_in_lane(base_env, target_lane)

            front_gap = float("inf") if front_vehicle is None else float(front_vehicle.position[0] - ego_x)
            rear_gap = float("inf") if rear_vehicle is None else float(ego_x - rear_vehicle.position[0])

            safe_front = front_gap >= min_front_gap
            safe_rear = rear_gap >= min_rear_gap

            return safe_front and safe_rear, front_gap, rear_gap

        def _front_distance_current_lane(base_env):
            ego = base_env.vehicle
            current_lane = _lane_id(ego)

            front_vehicle, _ = _front_rear_in_lane(base_env, current_lane)

            if front_vehicle is None:
                return None

            d = float(front_vehicle.position[0] - ego.position[0])
            return d if d >= 0 else None

        # Reward shaping
        lane_change_bonus = 0.00
        unsafe_lane_change_penalty = 0.25
        unnecessary_lane_change_penalty = 0.08
        blocked_front_penalty = 0.1
        blocked_lane_penalty = 0.12

        rapid_lane_change_penalty = 0.10
        return_to_previous_lane_penalty = 0.12
        lane_change_cooldown = 50

        unsafe_dist_threshold = 30.0
        unsafe_dist_penalty = 0.15
        spacing_bonus = 0.005
        blocked_threshold = 30.0

        ep_reward = 0.0
        steps = 0

        # Anti-oscillation state
        last_lane_change_step = -100
        prev_lane_before_change = None

        for _ in range(max_steps):
            obs_input = obs.reshape(1, -1)
            obs_tensor = torch.tensor(obs_input, dtype=torch.float32, device=DEVICE)

            logits, val = model(obs_tensor)
            act_distribution = Categorical(logits=logits)
            act = act_distribution.sample()
            act_log_prob = act_distribution.log_prob(act).item()

            act = act.item()
            val = val.item()

            # Before action
            old_lane = _lane_id(base_env.vehicle)
            old_front_dist = _front_distance_current_lane(base_env)

            current_lane = old_lane
            left_safe = False
            right_safe = False

            if current_lane is not None:
                left_lane = current_lane - 1
                right_lane = current_lane + 1

                if left_lane >= 0:
                    left_safe, _, _ = _is_safe_lane(base_env, left_lane)

                if right_lane < base_env.config["lanes_count"]:
                    right_safe, _, _ = _is_safe_lane(base_env, right_lane)

            # Environment step
            next_obs, reward, terminated, truncated, _ = env.step(act)
            reward = float(reward)

            # After action
            new_lane = _lane_id(base_env.vehicle)
            new_front_dist = _front_distance_current_lane(base_env)

            lane_changed = (
                old_lane is not None and
                new_lane is not None and
                new_lane != old_lane
            )

            # Safe / unnecessary / oscillating lane change logic
            if lane_changed:
                safe_change, front_gap, rear_gap = _is_safe_lane(base_env, new_lane)

                # Penalize rapid repeated lane changes
                if steps - last_lane_change_step < lane_change_cooldown:
                    reward -= rapid_lane_change_penalty

                # Penalize returning to previous lane too quickly
                if prev_lane_before_change is not None and new_lane == prev_lane_before_change:
                    if steps - last_lane_change_step < lane_change_cooldown:
                        reward -= return_to_previous_lane_penalty

                # Reward only if old lane was actually blocked and new lane is clearly better
                if old_front_dist is not None and old_front_dist < unsafe_dist_threshold:
                    if safe_change and (new_front_dist is None or new_front_dist > old_front_dist + 10.0):
                        reward += lane_change_bonus
                    else:
                        reward -= unsafe_lane_change_penalty
                else:
                    reward -= unnecessary_lane_change_penalty

                prev_lane_before_change = old_lane
                last_lane_change_step = steps

            # Penalize staying behind blocked front vehicle
            if old_front_dist is not None and old_front_dist < unsafe_dist_threshold:
                if new_lane == old_lane:
                    reward -= blocked_front_penalty

            # Blocked lane incentive
            lane_blocked = old_front_dist is not None and old_front_dist < blocked_threshold
            if lane_blocked:
                if (left_safe or right_safe) and new_lane == current_lane:
                    reward -= blocked_lane_penalty

            # Dense safety signal
            d_front = _front_distance_current_lane(base_env)
            if d_front is not None and d_front < unsafe_dist_threshold:
                closeness = (unsafe_dist_threshold - d_front) / unsafe_dist_threshold
                reward -= unsafe_dist_penalty * closeness
            elif d_front is not None and d_front > 20.0:
                reward += spacing_bonus

            env.render()

            for i, item in enumerate((obs_input.squeeze(0), act, reward, val, act_log_prob)):
                train_data[i].append(item)

            obs = np.array(next_obs, dtype=np.float32)
            ep_reward += reward
            steps += 1

            if terminated or truncated:
                break

        crashed = getattr(base_env.vehicle, "crashed", False)

        train_data = [np.asarray(x) for x in train_data]
        train_data[3] = PPOUtils.calculate_gaes(train_data[2], train_data[3])

        return train_data, ep_reward, steps, crashed

    @staticmethod
    def discount_rewards(rewards, gamma=0.99):
        new_rewards = [float(rewards[-1])]
        for i in reversed(range(len(rewards) - 1)):
            new_rewards.append(float(rewards[i]) + gamma * new_rewards[-1])
        return np.array(new_rewards[::-1])

    @staticmethod
    def calculate_gaes(rewards, values, gamma=0.99, decay=0.97):
        next_values = np.concatenate([values[1:], [0]])
        deltas = [rew + gamma * next_val - val for rew, val, next_val in zip(rewards, values, next_values)]

        gaes = [deltas[-1]]
        for i in reversed(range(len(deltas) - 1)):
            gaes.append(deltas[i] + decay * gamma * gaes[-1])

        return np.array(gaes[::-1])

    @staticmethod
    def create_directory():
        now = datetime.datetime.now()
        date_time = now.strftime("%Y-%m-%d_%H-%M-%S")
        act_n_obs = f"{action_type}_{obs_type}"
        env_folder = env.spec.id.replace("-", "_")
        folder_name = f"{date_time}_{env_folder}_{act_n_obs}"
        return folder_name

    @staticmethod
    def save_models(model, ep_rewards, ppo):
        model_folder = "models"
        os.makedirs(model_folder, exist_ok=True)

        output_folder = os.path.join(model_folder, PPOUtils.create_directory())
        os.makedirs(output_folder, exist_ok=True)

        plt.figure(figsize=(10, 5))
        plt.plot(ep_rewards, label="Episode Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Training Progress")
        plt.legend()
        plt.grid(True)

        actor_model_path = os.path.join(output_folder, "actor_model.pth")
        torch.save(model.state_dict(), actor_model_path)

        critic_model_path = os.path.join(output_folder, "critic_model.pth")
        torch.save(ppo.ac.state_dict(), critic_model_path)

        plot_path = os.path.join(output_folder, "training_progress.png")
        plt.savefig(plot_path)

        print("Saved actor model to:", actor_model_path)
        print("Saved critic model to:", critic_model_path)
        print("Saved training plot to:", plot_path)

        plt.show()

    @staticmethod
    def load_models(model_path, obs_space, action_space):
        actor_model = ActorCriticNetwork(obs_space, action_space)
        critic_model = ActorCriticNetwork(obs_space, action_space)

        actor_model.load_state_dict(torch.load(model_path))
        critic_model.load_state_dict(torch.load(model_path))

        actor_model.eval()
        critic_model.eval()

        return actor_model, critic_model