from agent.sb3 import Agent
from configurations import *

env_args = {
    "symbol": 'XBTUSD',
    #"fitting_file": 'XBTUSD_20200101_20200108_merge.csv.xz', 
    "fitting_file": 'XBTUSD_20200101_20200102_merge.csv.xz',
    #"testing_file": 'XBTUSD_20200109_20200120_merge.csv.xz',
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

global_vars = {
    "max_book_rows": MAX_BOOK_ROWS,
    "include_oderflow": INCLUDE_ORDERFLOW,
    "market_order_fee": MARKET_ORDER_FEE,
    "limit_order_fee": LIMIT_ORDER_FEE,
    "slippage": SLIPPAGE,
    "indicator_window": INDICATOR_WINDOW,
    "indicator_window_max": INDICATOR_WINDOW_MAX,
    "indicator_window_features": INDICATOR_WINDOW_FEATURES,
    "ema_alpha": EMA_ALPHA
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
        test_params=None, save_model=False
    )
    agent.start()