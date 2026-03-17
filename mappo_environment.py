import gymnasium as gym


class Hyperparameters:
    DEVICE = 'cpu'

    N_EPISODES = 5000
    PRINT_FREQ = 20
    MAX_STEPS = 200

    POLICY_LR = 3e-4
    VALUE_LR = 1e-3
    PPO_CLIP = 0.2
    TARGET_KL_DIV = 0.02
    MAX_POLICY_TRAIN_ITERS = 5
    VALUE_TRAIN_ITERS = 10

    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    ENTROPY_COEF = 0.001

    CONTROLLED_VEHICLES = 2
    VEHICLES_COUNT = 20
    LANES_COUNT = 6

    ACTION_TYPE = 'DiscreteMetaAction'
    LOCAL_OBS_TYPE = 'Kinematics'
    FEATURES = ['presence', 'x', 'y', 'vx', 'vy', 'cos_h', 'sin_h']
    VEHICLES_OBSERVED = 6

    RENDER_MODE = 'human'
    RENDER = False
    RENDER_EVERY = 1

    @classmethod
    def local_obs_dim(cls):
        return cls.VEHICLES_OBSERVED * len(cls.FEATURES)

    @classmethod
    def joint_obs_dim(cls):
        return cls.CONTROLLED_VEHICLES * cls.local_obs_dim()

    ACTION_SPACE = 5




def make_env(render=None):
    render_mode = Hyperparameters.RENDER_MODE if render is None else render
    env = gym.make('highway-v0', render_mode=render_mode)
    env.configure({
        'controlled_vehicles': Hyperparameters.CONTROLLED_VEHICLES,
        'vehicles_count': Hyperparameters.VEHICLES_COUNT,
        'lanes_count': Hyperparameters.LANES_COUNT,
        'policy_frequency': 5,
        'vehicles_density': 2, # Changed from 2 to 1 MG
        'simulation_frequency': 15,
        #'duration': 20,
        'show_trajectories': True,
        'screen_height': 320,
        'screen_width': 1100,
        'scaling': 4.2,
        'centering_position': [0.35, 0.5],
        'normalize_reward': False,
        'collision_reward': -10,
        'high_speed_reward': 0.0,
        'right_lane_reward': 0.00,
        'lane_change_reward': 0.0,
        'action': {
            'type': 'MultiAgentAction',
            'action_config': {'type': Hyperparameters.ACTION_TYPE},
        },
        'observation': {
            'type': 'MultiAgentObservation',
            'observation_config': {
                'type': Hyperparameters.LOCAL_OBS_TYPE,
                'vehicles_count': Hyperparameters.VEHICLES_OBSERVED,
                'features': Hyperparameters.FEATURES,
                'absolute': False,
                'normalize': True,
                'clip': False,
                'see_behind': True,
            },
        },
    })
    return env
