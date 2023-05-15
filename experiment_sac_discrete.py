import gym
import torch
import gym_trading
from collections import deque
import wandb
from agent.algorithms.SAC_discrete.agent import SAC
from agent.algorithms.SAC_discrete.buffer import ReplayBuffer

env_args = {
    "symbol": 'XBTUSD',
    "fitting_file": 'XBTUSD_20200101_20200102_merge.csv.xz',
    "testing_file": 'XBTUSD_2020-01-03.csv.xz',
    "max_position": 10,
    "window_size": 100,
    "seed": 1,
    "action_repeats": 5,
    "training": True,
    "format_3d": False,
    "reward_type": 'trade_completion',
    "transaction_fee": True,
    "id": 'market-maker-v0'    
}

agent_args = {
    "env": "market-maker-v0",
    "numer_of_training_steps": 1e3
}

if __name__ == '__main__':
    print(100_000)
    env = gym.make(**env_args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    steps = 0
    average10 = deque(maxlen=10)
    total_steps = 0

    with wandb.init(project='SAC_disc', name='sac', config= env_args | agent_args):
        
        agent = SAC(state_size=env.observation_space.shape[0],
                    action_size=env.action_space.n,
                    device=device)
        
        wandb.watch(agent, log='gradients', log_freq=10)

        buffer = ReplayBuffer