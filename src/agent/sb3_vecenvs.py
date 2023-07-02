from datetime import datetime as dt
from typing import Callable
import pickle

from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn

import wandb
import matplotlib.pyplot as plt

from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecEnvWrapper
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import TensorBoardOutputFormat

import gym
from gym.envs.registration import register

from gym_trading.envs.market_maker import MarketMaker
from configurations import LOGGER

class Agent:
    def __init__(self, config, log_code=True, algorithm="dqn",
                 save_model=True):
        self.config = config
        self.log_code = log_code
        self.algorithm = algorithm
        self.save_model = save_model
        self.log_interval = 1
        self._now = None
        self.vanilla_env = None

    def train(self, envs_dict):
        
        run = self.setup_wandb(envs_dict[0])

        # Register custom environment
        self.register_custom_envs(envs_dict)

        self._now = dt.now().strftime("%Y_%m_%d"+"_at_"+"%H_%M")

        # Create vectorized environment
        env = self.create_vectorized_envs(len(envs_dict))
        
        # Define agent
        model = self.create_agent(env)

        # Define callbacks
        callback_list = self.create_callbacks()

        # Train agent
        LOGGER.info('Starting training now...')
        model.learn(
            total_timesteps=self.config['total_timesteps'] * 2, #* len(envs_dict),
            callback=callback_list,
            log_interval=self.log_interval
        )

        with open('saved_statistics_dict.pkl','wb') as f:
            pickle.dump(callback_list.statistics, f)

        wandb.save('saved_statistics_dict.pkl')

        self.plot_statistics(callback_list.statistics)

        plot_types = list(callback_list.statistics[0].keys()) 
        for t in plot_types:
            wandb.log({t: wandb.Image(f"wandb_episode_{t}.png")})

        run.finish()
        
        # Save final model
        if self.save_model:
            #model.save('trained_model'+self._now)
            reward_type = envs_dict[0]['reward_type']
            model.save(f'models/{self.algorithm}/{reward_type}/{self._now}')

        LOGGER.info('Finished training...')

    def test(self, env_args, load_model_path=None):
        
        #load_model_path = 'models/ppo/trade_completion/2023_06_04'
        env_id = 'market-maker-v101'

        if 'a2c' == self.algorithm:
            model = A2C.load(load_model_path)
        elif 'ppo' == self.algorithm:
            model = PPO.load(load_model_path)
        elif 'dqn' == self.algorithm:
            model = DQN.load(load_model_path)
        else:
            raise ValueError('Not implemented, check load model path.')

        # Set up Wandb
        run = self.setup_wandb(env_args)

        register(
            id=env_id,
            entry_point=MarketMaker,
            kwargs=env_args
        )

        env = self.create_env(env_id)

        obs = env.reset()
        while True:
            action, _states = model.predict(obs)
            obs, rewards, dones, infos = env.step(action)
    
            if dones[0] == True:
                break
        

        infos[0].pop('terminal_observation', None)
        # Log run statistics and plot in wandb
        wandb.log(infos[0])
        wandb.log({'plot': wandb.Image('wandb_plot.png')})

        # Finish Wandb run
        run.finish()

        return infos[0]
    
    
    def setup_wandb(self, env_args):
        
        # Initialize Wandb
        wandb.init(
            project="thesis",
            config={
                **self.config, 
                **env_args, 
                "algorithm": self.algorithm
            },
            sync_tensorboard=True,
            save_code=self.log_code
        )
        if self.log_code:
            wandb.run.log_code(".")

        return wandb.run
    

    def register_custom_envs(self, envs_dict):
        # Register custom environment
        for i in range(len(envs_dict)):
            register(
                id='market-maker-v'+str(i),
                entry_point=MarketMaker,
                kwargs=envs_dict[i]
            )

    def create_vectorized_envs(self, n_envs):
      
        # Create vectorized environment
        def make_env(env_id: str):
            '''
            Utility function for multiprocessed env.
            '''
            def _init() -> gym.Env:
                env = gym.make(id=env_id)
                env = Monitor(env)
                return env
            return _init
    
        envs = [make_env('market-maker-v'+str(i)) for i in range(n_envs)]
        #envs = Monitor(envs)
        env = SubprocVecEnv(envs, start_method='fork')
        #env = DummyVecEnv(envs)
        env = VecMonitor(env)
        return env
    
    def create_env(self, env_id):
        def make_env(): 
            env = gym.make(id=env_id)
            self.vanilla_env = env
            return env
        
        env = DummyVecEnv([make_env])

        return env

    def create_agent(self, env):
        # Define agent
        if self.algorithm == "dqn":
            LOGGER.info('Initializing DQN')
            model = DQN(
                self.config['policy_type'],
                env,
                verbose=0,
                buffer_size=10_000,
                #tensorboard_log=f"./runs/{self.algorithm}_{self._now}"
            )
        elif self.algorithm == "ppo":
            LOGGER.info('Initializing PPO')
            model = PPO(
                self.config['policy_type'],
                env,
                verbose=0,
                #tensorboard_log=f"./runs/{self.algorithm}_{self._now}",
                gamma=0.99,
                gae_lambda=0.97,
                n_steps=256 # as reported in paper
            )
        elif self.algorithm == "a2c":
            LOGGER.info('Initializing A2C')
            model = A2C(
                self.config['policy_type'],
                env,
                verbose=0,
                #tensorboard_log=f"./runs/{self.algorithm}_{self._now}",
                gamma=0.99,
                gae_lambda=0.97,
                use_rms_prop=False, # use Adam as optim
                n_steps=40 # as reported in paper
            )
        else:
            raise ValueError("Invalid algorithm specified.")

        return model

    def create_callbacks(self):
        # Define callback to track additional params in Wandb
        '''
        class TensorboardCallback(BaseCallback):
            def __init__(self, verbose=0):
                super(TensorboardCallback, self).__init__(verbose)
                self.wandb_logs = {}

            def _on_rollout_start(self) -> None:
                self.logger.record("logger/episode_reward", self.wandb_logs['episode reward'])
                self.logger.record("logger/episode_pnl", self.wandb_logs['episode pnl'])
                self.logger.record("logger/episode_avg_trade_pnl", self.wandb_logs['episode avg pnl'])

            def _on_step(self) -> bool:
                self.tb_episode_reward = self.training_env.get_attr("wandb_logs['episode reward']")[0]
                self.tb_episode_pnl = self.training_env.get_attr("wandb_logs['episode pnl']")[0]
                self.tb_episode_avg_pnl = self.training_env.get_attr("wandb_logs['episode avg pnl']")[0]
                return True
        
        # Define callback to save agent periodically
        checkpoint_callback = CheckpointCallback(
            save_freq=self.config['save_interval'],
            #save_path="saved_agents/" + str(dt.now()).split()[0] + "_" + str(run_id) + "/"
            save_path="saved_agents/" + self.algorithm + "_" + self._now + "/"
        )

        # Concatenate all defined callbacks
        callback_list = CallbackList([TensorboardCallback(), checkpoint_callback])
        '''
        '''
        class TensorboardCallback(BaseCallback):
            def __init__(self, verbose=0):
                super(TensorboardCallback, self).__init__(verbose)
                self.wandb_logs = []
             
            def _on_rollout_start(self):
                if len(self.wandb_logs) != 0:
                    for i in range(8):
                        for k in ['episode avg pnl','episode pnl', 'episode reward']:
                            self.logger.record(f"logger/env{i}_{k}",self.wandb_logs[i][k])
                
                #self.logger.record("logger/reward", self.wandb_logs)
                #print(self.wandb_logs)
            
            def _on_step(self) -> bool:
                
                #Log my_custom_reward every _log_freq(th) to tensorboard for each environment
                
                if len(self.wandb_logs) != 0:
                    for i in range(8):
                        for k in ['episode avg pnl','episode pnl', 'episode reward']:
                            self.logger.record(f"logger/env{i}_{k}",self.wandb_logs[i][k])
                #self.wandb_logs = self.training_env.get_attr('wandb_logs')
                #print(self.wandb_logs)
                return True
                
        return TensorboardCallback() #callback_list
    
        

        class MultiEnvRewardCallback(BaseCallback):
            def __init__(self, verbose=0):
                super(MultiEnvRewardCallback, self).__init__(verbose)

            def _on_step(self) -> bool:
                rewards = [e.get_attr('reward') for e in env]
                wandb.log({'Rewards': rewards})

        return MultiEnvRewardCallback()
        '''

        class TensorboardCallback(BaseCallback):
            def __init__(self, verbose=0):
                super(TensorboardCallback, self).__init__(verbose)
                self.statistics = {}
                for i in range(8):
                    self.statistics[i] = dict.fromkeys(['reward','pnl','avg_pnl'])
                    self.statistics[i]['reward'] = [0]
                    self.statistics[i]['pnl'] = [0]
                    self.statistics[i]['avg_pnl'] = [0]

            def _on_rollout_end(self) -> None:
                for k in self.statistics.keys():
                    env_reward = round(self.locals["infos"][k]["episode reward"], 2)
                    env_pnl = self.locals["infos"][k]["episode pnl"]
                    env_avg_pnl = self.locals["infos"][k]["episode avg pnl"]

                    if env_reward != self.statistics[k]['reward'][-1]:
                        self.statistics[k]['reward'].append(env_reward)
                    if env_pnl not in self.statistics[k]['pnl']:
                        self.statistics[k]['pnl'].append(env_pnl)
                    if env_avg_pnl not in self.statistics[k]['avg_pnl']:
                        self.statistics[k]['avg_pnl'].append(env_avg_pnl)

            def _on_step(self) -> bool:
                return True
            
        return TensorboardCallback()
    
    def plot_statistics(self, statistics_dict, save=True):
    
        labels = list(statistics_dict.keys())
        plot_types = list(statistics_dict[labels[0]].keys())
        for type in plot_types:
            fig = plt.figure(figsize=(20,12))
            plt.style.use('classic')
            plt.title('episode ' + type)
            for i in labels:
                plt.plot(statistics_dict[i][type], label = f'env_{i}')
            plt.legend()

            if not save:
                plt.show()
            else:
                plt.savefig(f"wandb_episode_{type}.png")
                plt.close(fig)          
