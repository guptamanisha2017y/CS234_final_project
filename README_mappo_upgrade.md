# MAPPO upgrade for cooperative highway-env driving

This version upgrades your project from shared-policy PPO to a **MAPPO-style setup**:

- **Decentralized actors**: each controlled car chooses its action from its own local observation.
- **Centralized critic**: the value network sees the concatenated observations of **all 3 controlled cars**.
- **Shared team reward**: all agents optimize the same cooperative objective.

## What changed

### 1. Multi-agent environment
`mappo_environment.py` uses:
- `controlled_vehicles = 3`
- `MultiAgentAction`
- `MultiAgentObservation`

### 2. New model
`mappo_model.py` has:
- an **actor** that consumes one car's local observation
- a **critic** that consumes the joint observation of all controlled cars

### 3. New rollout logic
`mappo_utils.py`:
- samples one action per controlled car
- steps the environment with a tuple of actions
- computes a **single team reward**
- stores local observations for actor updates and joint observations for critic updates

### 4. Rendering
Training and inference both render in `human` mode.
If you are on a remote VM with no display, the window will not appear.

## Run training
```bash
python mappo_main.py
```

## Run inference
```bash
python mappo_inference.py -mp models/<run_folder>/mappo_actor_critic.pth -i 5
```

## Why this is more cooperative than parameter-sharing PPO
Your previous cooperative PPO mainly used:
- shared policy
- shared reward

This MAPPO version adds a **centralized critic**. That means the value estimate is based on the **full team context**, so the policy gradient is guided by what helps the group, not just what seems good from one car's local view.

## Important requirement
This needs a version of `highway-env` with multi-agent support. If your installed version is too old, upgrade it first.
