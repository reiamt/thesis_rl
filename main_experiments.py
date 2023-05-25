from agent.sb3 import Agent

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
    "reward_type": 'default',
    "transaction_fee": True  
}

config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 1_000_000,
    "save_interval": 1_000_000 #steps not episodes!
}

test_params = {
    "run_id": 'ppo_dqe71rh3/rl_model_1000000_steps.zip',
    "n_eval_episodes": 5
}

algos = ['dqn', 'ppo', 'a2c']
reward_types = ['default', 'default_with_fills', 'asymmetrical', 'realized_pnl',
                'differential_sharpe_ratio', 'trade_completion']

for algo in algos:
    agent = Agent(
        env_args, config, algorithm=algo, log_code=False, 
        test_params=None
    )
    agent.start()