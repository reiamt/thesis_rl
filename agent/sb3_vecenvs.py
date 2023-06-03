from datetime import datetime as dt
from typing import Callable

import wandb

from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy

import gym
from gym.envs.registration import register

from gym_trading.envs.market_maker import MarketMaker
from configurations import LOGGER

class Agent:
    def __init__(self, envs_dict, config, test_env_args, log_code=True, algorithm="dqn",
                 test_params = None, save_model=True):
        self.envs_dict = envs_dict
        self.config = config
        self.test_env_args = test_env_args
        self.log_code = log_code
        self.algorithm = algorithm
        self.test_params = test_params
        self.save_model = save_model
        self.log_interval = 1
        self.vanilla_env = None
        self._now = None

    def start(self):
        
        # Register custom environment
        self.register_custom_envs()

        self._now = dt.now().strftime("%Y_%m_%d"+"_at_"+"%H_%M")

        # Create vectorized environment
        env = self.create_vectorized_envs()

        # Define agent
        model = self.create_agent(env)

        # Define callbacks
        callback_list = self.create_callbacks()

        # Train agent
        LOGGER.info('Starting training now...')
        model.learn(
            total_timesteps=self.config['total_timesteps']*len(self.envs_dict),
            callback=callback_list,
            log_interval=self.log_interval
        )

        # Save final model
        if self.save_model:
            model.save('trained_model'+self._now)

        LOGGER.info('Finished training...')

    def test(self):

        model = A2C.load('trained_model2023_06_03_at_13_05')

        # Set up Wandb
        run = self.setup_wandb()

        register(
            id='market-maker-v10',
            entry_point=MarketMaker,
            kwargs=self.test_env_args
        )

        def make_env(): 
            env = gym.make(id='market-maker-v10')
            self.vanilla_env = env
            return env
        env = DummyVecEnv([make_env])

        obs = env.reset()
        while True:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
    
            if dones[0] == True:
                break
        
        # Log run statistics and plot in wandb
        wandb.log(self.vanilla_env.wandb_logs)
        wandb.log({'plot': wandb.Image('wandb_plot.png')})

        # Finish Wandb run
        run.finish()
    
    
    def setup_wandb(self):
        # Initialize Wandb
        wandb.init(
            project="thesis",
            config={
                **self.config, 
                **self.test_env_args, 
                "algorithm": self.algorithm
            },
            sync_tensorboard=True,
            save_code=self.log_code
        )
        if self.log_code:
            wandb.run.log_code(".")

        return wandb.run
    

    def register_custom_envs(self):
        # Register custom environment
        for i in range(len(self.envs_dict)):
            register(
                id='market-maker-v'+str(i),
                entry_point=MarketMaker,
                kwargs=self.envs_dict[i]
            )

    def create_vectorized_envs(self):
        # Create vectorized environment
        def make_env(env_id: str):
            '''
            Utility function for multiprocessed env.
            '''
            def _init() -> gym.Env:
                env = gym.make(id=env_id)
                return env
            return _init
    
        envs = [make_env('market-maker-v'+str(i)) for i in range(len(self.envs_dict))]
        #envs = Monitor(envs)
        env = SubprocVecEnv(envs, start_method='fork')
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
                tensorboard_log=f"./runs/{self.algorithm}_{self._now}"
            )
        elif self.algorithm == "ppo":
            LOGGER.info('Initializing PPO')
            model = PPO(
                self.config['policy_type'],
                env,
                verbose=0,
                tensorboard_log=f"./runs/{self.algorithm}_{self._now}",
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
                tensorboard_log=f"./runs/{self.algorithm}_{self._now}",
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
        class TensorboardCallback(BaseCallback):
            def __init__(self, verbose=0):
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

        # Define callback to save agent periodically
        checkpoint_callback = CheckpointCallback(
            save_freq=self.config['save_interval'],
            #save_path="saved_agents/" + str(dt.now()).split()[0] + "_" + str(run_id) + "/"
            save_path="saved_agents/" + self.algorithm + "_" + self._now + "/"
        )

        # Concatenate all defined callbacks
        callback_list = CallbackList([TensorboardCallback(), checkpoint_callback])

        return callback_list
    

