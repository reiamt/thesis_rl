from agent.sb3_gpt import Agent

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

config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 500_000,
    "save_interval": 100_000 #steps not episodes!
}

agent = Agent(env_args, config)
agent.start()