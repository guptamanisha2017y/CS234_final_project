import pprint
import gymnasium as gym

class Hyperparameters():

    # Define Device
    DEVICE = 'cpu'

    # Define training params
    N_EPISODES = 2000 #MG changed from 300 to 500 to 5000
    PRINT_FREQ = 50 # Reduced the print frequency from 20 to 50 per episode

    # Define agent params
    POLICY_LR = 3E-4
    VALUE_LR = 1E-3
    TARGET_KL_DIV = 0.02
    MAX_POLICY_TRAIN_ITERS = 5
    VALUE_TRAIN_ITERS = 10

    # Define observation and action space
    OBS_SPACE = 70 #10*7
    ACTION_SPACE = 5
    ACTION_TYPE = 'DiscreteMetaAction'
    OBS_TYPE = 'Kinematics'
    

    @classmethod
    def all(cls):
        return [value for name, value in vars(cls).items() if name.isupper()]

action_type, obs_type = Hyperparameters.all()[10:12]

env = gym.make('highway-v0', render_mode='rgb_array')

env.configure({
    'action': {'type': action_type},
    'observation': {
        'type': 'Kinematics',
        'vehicles_count': 10,   # ego + nearby vehicles
        'features': ['presence', 'x', 'y', 'vx', 'vy','cos_h', 'sin_h'],
        'features_range': {
            'x': [-100, 100],
            'y': [-12, 12],
            'vx': [-20, 20],
            'vy': [-20, 20],
        },
        'absolute': False,      # relative to ego vehicle
        'flatten': True,
        'observe_intentions': False,
    },
    'policy_frequency': 5,
    'show_trajectories': True,
    'lanes_count': 6,
    'vehicles_density': 1,
    'vehicles_count': 20,
    'screen_height': 300,
    'screen_width': 700,
    'collision_reward': -10,
    'high_speed_reward': 0.00,
    'right_lane_reward': 0.00,
    'lane_change_reward': 0.0,
    'normalize_reward': False,
})

obs_space_size = env.observation_space
action_space_size = env.action_space


