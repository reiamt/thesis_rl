from datetime import datetime as dt
from datetime import timedelta

from agent.sb3_vecenvs import Agent
from configurations import *

start_date = dt(2020,1,1) #of fitting, ie training starts on one day later
num_days = 8

paths = ['XBTUSD_' + (start_date+timedelta(i)).strftime("%Y-%m-%d") + '.csv.xz' 
         for i in range(num_days+1)]

envs_dict = {}.fromkeys(range(num_days))

for i in range(num_days):
    env_args = {
        "symbol": 'XBTUSD',
        "fitting_file": paths[i],
        "testing_file": paths[i+1],
        "max_position": 10.,
        "window_size": 100,
        "seed": i,
        "action_repeats": 5, #set to 1 if price data is used, else 5
        "training": True,
        "format_3d": False,
        "reward_type": 'trade_completion',
        "transaction_fee": True,
        "include_imbalances": False
    }
    envs_dict[i] = env_args

test_start_date = dt(2020,1,10) #of fitting, ie training starts on one day later
test_num_days = 1

test_paths = ['XBTUSD_' + (test_start_date+timedelta(i)).strftime("%Y-%m-%d") + '.csv.xz' 
         for i in range(test_num_days+1)]

for i in range(test_num_days):
    test_env_args = {
        "symbol": 'XBTUSD',
        "fitting_file": test_paths[i],
        "testing_file": test_paths[i+1],
        "max_position": 10.,
        "window_size": 100,
        "seed": i,
        "action_repeats": 1, #set to 1 if price data is used, else 5
        "training": False,
        "format_3d": False,
        "reward_type": 'trade_completion',
        "transaction_fee": True,
        "include_imbalances": False
    }

# to pass into wandb
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
    "save_interval": 1_000_000 
}

algos = ['a2c']#['dqn', 'ppo', 'a2c']
reward_types = ['default', 'default_with_fills', 'asymmetrical', 'realized_pnl',
                'differential_sharpe_ratio', 'trade_completion']



for algo in algos:
    agent = Agent(
        envs_dict, config, algorithm=algo, test_env_args=test_env_args,
        log_code=True, save_model=True
    )
    agent.test()
    #agent.start()
    