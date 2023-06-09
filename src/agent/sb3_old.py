from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from datetime import datetime as dt

import wandb

from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback

import gym
from gym.envs.registration import register
from gym_trading.envs.market_maker import MarketMaker

# define args for env init
env_args = {
    "symbol": 'XBTUSD',
    "fitting_file": 'XBTUSD_20200101_20200102_merge.csv.xz',
    "testing_file": 'XBTUSD_2020-01-03.csv.xz',
    "max_position": 10.,
    "window_size": 100,
    "seed": 1,
    "action_repeats": 5,
    "training": True,
    "format_3d": False,
    "reward_type": 'trade_completion',
    "transaction_fee": True  
}

# register custom env as gym env
register(id='market-maker-v0', 
         entry_point = MarketMaker,
         kwargs = env_args)

# configs for the agent
config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 100_000,
    "env_name": "market-maker-v0",
    "save_interval": 100_000 #steps not episodes!
}

# wandb init
run = wandb.init(
    project="thesis",
    config={**config, **env_args},
    sync_tensorboard=True,
    save_code=True
)

wandb.run.log_code(".")

# vectorize env
def make_env(): 
    env = gym.make(id='market-maker-v0')
    env = Monitor(env)
    return env

env = DummyVecEnv([make_env])

# define agent
model = PPO(config['policy_type'], 
    env, verbose=1, 
    #buffer_size=10_000,
    tensorboard_log=f"./runs/{run.id}"
)

# class to track additional params in wandb
class TensorboardCallback(BaseCallback):
    def __init__(self, verbose = 0):
        super(TensorboardCallback, self).__init__(verbose)
        self.tb_episode_reward = 0
        self.tb_episode_pnl = 0
        self.tb_episode_avg_pnl = 0

    def _on_rollout_start(self) -> None:
        self.logger.record("logger/episode_reward", self.tb_episode_reward)
        self.logger.record("logger/episode_pnl", self.tb_episode_pnl)
        self.logger.record("logger/episode_avg_trade_pnl", self.tb_episode_avg_pnl)

    def _on_step(self) -> bool:
        self.tb_episode_reward = self.training_env.get_attr("tb_episode_reward")[0]    
        self.tb_episode_pnl = self.training_env.get_attr("tb_episode_pnl")[0]
        self.tb_episode_avg_pnl = self.training_env.get_attr("tb_episode_avg_pnl")[0]
        return True
    
# define callback to save agent periodically
checkpoint_callback = CheckpointCallback(
    save_freq=config['save_interval'],
    save_path="saved_agents/"+str(dt.now()).split()[0]+"_"+str(run.id)+"/"
)

# concatenate all defined callbacks
callback_list = CallbackList([TensorboardCallback(), checkpoint_callback])

# train agent
model.learn(
    total_timesteps=config['total_timesteps'],
    callback=callback_list,
    log_interval=1
)

# close wandb run
run.finish()


