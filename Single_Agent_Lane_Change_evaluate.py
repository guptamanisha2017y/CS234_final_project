import argparse
import numpy as np
import torch
from torch.distributions.categorical import Categorical
from Single_Agent_Lane_Change_environment import env
from Single_Agent_Lane_Change_ppo_model import PPOUtils
from Single_Agent_Lane_Change_environment import Hyperparameters


obs_space, action_space = Hyperparameters.all()[8:10]

# Define the command-line arguments
parser = argparse.ArgumentParser(description='Perform inference with PPO model')
parser.add_argument('-mp', type=str, required=True, help='Path to the pre-trained model')
parser.add_argument('-i', type=int, default=10, help='Number of inference iterations')

# Parse the command-line arguments
args = parser.parse_args()

# Load the model
actor_model, critic_model = PPOUtils.load_models(args.mp, obs_space, action_space)

rewards = []
avg_rewards = []

# Track lane for lane-change reward (lane_index is often a tuple; lane id is last element)
def _lane_id(vehicle):
    li = getattr(vehicle, "lane_index", None)
    if li is None:
        return None
    # lane_index often looks like (road_id, lane_id, lane) depending on version;
    # commonly lane id is the last element or index 2.
    try:
        return li[2]
    except Exception:
        try:
            return li[-1]
        except Exception:
            return None

prev_lane = _lane_id(env.unwrapped.vehicle)

num_crashes = 0

# Perform inference in the environment
for i in range(args.i): # Inference Iterations
    terminated = False
    truncated = False
    env.np_random
    observation, _ = env.reset()
    observation = observation.squeeze()
    steps = 0
    lane_change=0
    while not (terminated or truncated): #MG fix bracket
        with torch.no_grad():
            observation_tensor = torch.tensor(observation, dtype=torch.float32).reshape(1, -1)
            policy_logits = actor_model.policy(observation_tensor)  # Extract policy logits
            action_distribution = Categorical(logits=policy_logits)
            action = action_distribution.sample().item()

        observation, reward, terminated, truncated, info = env.step(action)

        # Implement a way to detect Collisions.
        rewards.append(reward)
        env.render()
        
        
        steps+= 1
        crashed = getattr(env.unwrapped.vehicle, "crashed", False)
        # =========================
        # 1) Lane-change count
        # =========================
        new_lane = _lane_id(env.unwrapped.vehicle)
        if (prev_lane is not None) and (new_lane is not None) and (new_lane != prev_lane):
            lane_change+=1
            prev_lane = new_lane        
        
        if isinstance(info, dict):
            if info.get("crashed", False) or info.get("collision", False):
                crashed = True
                
    if crashed:
        num_crashes += 1
    
    print(f'Episode: {i+1}  steps: {steps}')
    print(f'Episode: {i+1} lane change: {lane_change}')
    avg_reward = np.mean(rewards)
    avg_rewards.append(avg_reward)
    print(f'Episode: {i+1} Avg Reward: {avg_reward}')

print(f'Avg Reward for {i+1} Iteration(s): {np.mean(avg_rewards)}')
crash_rate = num_crashes / args.i if args.i > 0 else 0.0
print(f'Crash rate: {crash_rate:.2f} ({num_crashes}/{args.i})')

env.close()
