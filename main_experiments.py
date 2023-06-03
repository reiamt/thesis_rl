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
        "action_repeats": 1, #set to 1 if price data is used, else 5
        "training": True,
        "format_3d": False,
        "reward_type": 'trade_completion',
        "transaction_fee": True,
        "include_imbalances": False
    }
    envs_dict[i] = env_args

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

test_params = {
    "run_id": 'ppo_dqe71rh3/rl_model_1000000_steps.zip',
    "n_eval_episodes": 5
}

algos = ['a2c']#['dqn', 'ppo', 'a2c']
reward_types = ['default', 'default_with_fills', 'asymmetrical', 'realized_pnl',
                'differential_sharpe_ratio', 'trade_completion']

for algo in algos:
    agent = Agent(
        envs_dict, config, algorithm=algo, log_code=True, 
        test_params=None, save_model=False
    )
    agent.start()
    