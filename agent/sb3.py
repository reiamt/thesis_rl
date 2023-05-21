from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
print(sys.path)

import wandb
from wandb.integration.sb3 import WandbCallback

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat

import gym
from gym.envs.registration import register
from gym_trading.envs.market_maker import MarketMaker

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

register(id='market-maker-v0', 
         entry_point = MarketMaker,
         kwargs = env_args)

config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 500_000,
    "env_name": "market-maker-v0"
}

run = wandb.init(
    project="thesis",
    config={**config, **env_args},
    sync_tensorboard=True
)

def make_env(): 
    env = gym.make(id='market-maker-v0')
    env = Monitor(env)
    return env

env = DummyVecEnv([make_env])

model = DQN(
    config['policy_type'], 
    env, verbose=1, 
    buffer_size=10_000,
    tensorboard_log=f"./runs/{run.id}"
)

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose = 1):
        super(TensorboardCallback, self).__init__(verbose)
        self.reward = 0

    def _on_rollout_end(self) -> None:
        self.logger.record("logger/reward", self.reward)

        #reset vars
        self.reward = 0

    def _on_step(self) -> bool:
        self.reward += self.training_env.get_attr("reward")[0]

        return True


model.learn(
    total_timesteps=config['total_timesteps'],
    callback=TensorboardCallback()
)




'''model.learn(
    total_timesteps=config['total_timesteps'],
    callback=WandbCallback(
        gradient_save_freq=10,
        model_save_path=f"models/{run.id}",
        verbose=1
    )
)'''

run.finish()

'''
vec_env = model.get_env()
obs = vec_env.reset()
done = False
for i in range(10_000_000):
    action, _states = model.predict(obs, deterministic=False)
    obs, reward, done, info = vec_env.step(action)
    #print(done)
    #vec_env.render()
    #if i%1000==0: print(i)

env.close()
'''



#model.save("dqn_mm")