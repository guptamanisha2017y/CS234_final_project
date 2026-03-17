1. Install the required dependencies:

   cd ppo-highway-env_optim
   pip install -r requirements.txt


## Usage

2. Train the PPO agent:

   python main.py


2. Evaluate the trained agent:

   python evaluate.py -mp /path/to/pre-trained-model -i 10
   ```

   The evaluation script will load the trained model and test the agent's performance in the environment.

    - -mp or --model-path: Path to the pre-trained model (required).
    - -i or --inference-iterations: Number of inference iterations (default: 10).

