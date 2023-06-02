from datetime import datetime as dt

import wandb

from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy

import gym
from gym.envs.registration import register

from gym_trading.envs.market_maker import MarketMaker
from configurations import LOGGER

class Agent:
    def __init__(self, env_args, config, log_code=True, algorithm="dqn",
                 test_params = None, save_model=False):
        self.env_args = env_args
        self.config = config
        self.log_code = log_code
        self.algorithm = algorithm
        self.test_params = test_params
        self.save_model = save_model
        self.log_interval = 1#config['total_timesteps']/1000
        self.vanilla_env = None

    def start(self):
        # Set up Wandb
        run = self.setup_wandb()

        # Register custom environment
        self.register_custom_env()

        # Create vectorized environment
        env = self.create_vectorized_env()

        # Define agent
        model = self.create_agent(env, run.id)

        # Define callbacks
        callback_list = self.create_callbacks(run.id)

        if self.test_params is None:
            # Train agent
            LOGGER.info('Starting training now...')
            model.learn(
                total_timesteps=self.config['total_timesteps'],
                callback=callback_list,
                log_interval=self.log_interval
            )
        else:
            # Test agent
            LOGGER.info('Starting testing now...')
            path = 'saved_agents/' + self.test_params['run_id']
            model.load(path, env=env)
            mean_reward, std_reward = evaluate_policy(
                model, model.get_env(), 
                n_eval_episodes = self.test_params['n_eval_episodes']
            )


        self.vanilla_env.plot_trade_history(save_filename='wandb_picture.png')
        wandb.log({'plot': wandb.Image('wandb_picture.png')})
        # Finish Wandb run
        run.finish()
    
        # Save final model
        if self.save_model:
            model.save('models/' + run.id)

    def setup_wandb(self):
        # Initialize Wandb
        wandb.init(
            project="thesis",
            config={
                **self.config, 
                **self.env_args, 
                "algorithm": self.algorithm
            },
            sync_tensorboard=True,
            save_code=self.log_code
        )
        if self.log_code:
            wandb.run.log_code(".")

        return wandb.run

    def register_custom_env(self):
        # Register custom environment
        register(
            id='market-maker-v0',
            entry_point=MarketMaker,
            kwargs=self.env_args
        )

    def create_vectorized_env(self):
        # Create vectorized environment
        def make_env():
            self.vanilla_env = gym.make(id='market-maker-v0')
            env = Monitor(self.vanilla_env)
            return env

        env = DummyVecEnv([make_env])
        return env

    def create_agent(self, env, run_id):
        # Define agent
        if self.algorithm == "dqn":
            LOGGER.info('Initializing DQN')
            model = DQN(
                self.config['policy_type'],
                env,
                verbose=0,
                buffer_size=10_000,
                tensorboard_log=f"./runs/{run_id}"
            )
        elif self.algorithm == "ppo":
            LOGGER.info('Initializing PPO')
            model = PPO(
                self.config['policy_type'],
                env,
                verbose=0,
                tensorboard_log=f"./runs/{run_id}",
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
                tensorboard_log=f"./runs/{run_id}",
                gamma=0.99,
                gae_lambda=0.97,
                use_rms_prop=False, # use Adam as optim
                n_steps=40 # as reported in paper
            )
        else:
            raise ValueError("Invalid algorithm specified.")

        return model

    def create_callbacks(self, run_id):
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
            save_path="saved_agents/" + self.algorithm + "_" + str(run_id) + "/"
        )

        # Concatenate all defined callbacks
        callback_list = CallbackList([TensorboardCallback(), checkpoint_callback])

        return callback_list
    

