import argparse
import time
import torch
from torch.distributions.categorical import Categorical
from mappo_environment import make_env
from mappo_utils import MAPPOUtils


parser = argparse.ArgumentParser(description='Perform MAPPO inference with cooperative highway-env')
parser.add_argument('-mp', type=str, required=True, help='Path to the trained MAPPO model')
parser.add_argument('-i', type=int, default=5, help='Number of inference episodes')
args = parser.parse_args()

model = MAPPOUtils.load_model(args.mp)
env = make_env(render='human')

num_crashes = 0

for episode in range(args.i):
    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    crashed = False

    while not done:
        actions = []
        for agent_obs in obs:
            local_obs = MAPPOUtils.flatten_agent_obs(agent_obs)
            local_obs_tensor = torch.tensor(local_obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits = model.policy(local_obs_tensor)
                dist = Categorical(logits=logits)
                action = dist.sample().item()
            actions.append(action)

        obs, reward, terminated, truncated, info = env.step(tuple(actions))
        team_reward = float(sum(reward) / len(reward)) if isinstance(reward, (tuple, list)) else float(reward)
        total_reward += team_reward
        steps += 1
        env.render()
        time.sleep(1 / 15)
        
        if isinstance(info, dict):
            if info.get("crashed", False) or info.get("collision", False):
                crashed = True 

        done = terminated or truncated
        if isinstance(done, (tuple, list)):
            done = any(done)
    
    if crashed:
        num_crashes += 1

    print(f'Episode {episode + 1}: steps={steps}, team_reward={total_reward:.2f}')

crash_rate = num_crashes / args.i if args.i > 0 else 0.0
print(f'Crash rate: {crash_rate:.2f} ({num_crashes}/{args.i})')

env.close()
