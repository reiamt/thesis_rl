from datetime import datetime as dt
from datetime import timedelta
import pickle

import hydra
from omegaconf import DictConfig

import pandas as pd
import wandb

#orig_cwd = hydra.utils.get_original_cwd()

#from pathlib import Path
import sys
#sys.path.append(orig_cwd)

from agent.sb3_vecenvs import Agent
from configurations import *

start_date = dt(2020,1,1) #of fitting, ie training starts on one day later
num_days = 8

paths = ['XBTUSD_' + (start_date+timedelta(i)).strftime("%Y-%m-%d") + '.csv.xz' 
         for i in range(num_days+1)]

#paths = [p for p in paths if '8' not in p]

envs_dict = {}.fromkeys(range(len(paths)-1))

for i in range(len(paths)-1):
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
        "reward_type": 'differential_sharpe_ratio',
        "transaction_fee": True,
        "include_imbalances": True,
    }
    envs_dict[i] = env_args

# tests for the whole month of jan which wasn't used for training
test_start_date = dt(2020,1,9) #of fitting, ie training starts on one day later
test_num_days = 22

test_paths = ['XBTUSD_' + (test_start_date+timedelta(i)).strftime("%Y-%m-%d") + '.csv.xz' 
         for i in range(test_num_days+1)]

test_envs_dict = {}.fromkeys(range(num_days))
for i in range(test_num_days):
    test_env_args = {
        "symbol": 'XBTUSD',
        "fitting_file": test_paths[i],
        "testing_file": test_paths[i+1],
        "max_position": 10.,
        "window_size": 100,
        "seed": 101,
        "action_repeats": 1,
        "training": False,
        "format_3d": False,
        "reward_type": 'differential_sharpe_ratio',
        "transaction_fee": True,
        "include_imbalances": True,
    }
    test_envs_dict[i] = test_env_args

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
    "policy_type": "MlpPolicy", #"MlpPolicy",
    "total_timesteps": 1_000_000 
}

algos = ['a2c']#['dqn', 'ppo', 'a2c']
reward_types = ['default', 'default_with_fills', 'asymmetrical', 'realized_pnl',
                'differential_sharpe_ratio', 'trade_completion']

train = True
train_and_test = True

if __name__ == "__main__":
    if train:
        for algo in algos:
            agent = Agent(
                config, algorithm=algo,
                log_code=True, save_model=True
            )
            model_path = agent.train(envs_dict)
    else:
        statistics_dict = {}
        for algo in algos:
            statistics_dict[algo] = {}
            for i in range(len(test_envs_dict)):
                LOGGER.info(f"testing the trained algorithm on {(test_start_date+timedelta(i+1)).date()}")
                agent = Agent(
                    config, algorithm=algo,
                    log_code=False, save_model=False
                )
                if '2020-01-14' not in test_envs_dict[i]['fitting_file'] and '2020-01-14' not in test_envs_dict[i]['testing_file'] \
                    and '2020-02-09' not in test_envs_dict[i]['fitting_file'] and '2020-02-09' not in test_envs_dict[i]['testing_file']:
                    #model_path = 'models/a2c/trade_completion/0201-0901_2023_06_04'
                    model_path = 'models/ppo/differential_sharpe_ratio/2023_07_07_at_11_44'

                    #returns statistics dict
                    run_stats = agent.test(test_envs_dict[i], model_path)
                    
                    statistics_dict[algo][test_start_date+timedelta(i+1)] = run_stats

            with open('test_statistics_dict.pkl','wb') as f:
                pickle.dump(statistics_dict, f)
            
            #save statistics dict as dataframe
            data = []
            columns = ['algo','date','episode reward', 'episode pnl', 'episode avg pnl', 'episode hodl pnl']
            for algo in statistics_dict.keys():
                for date in statistics_dict[algo].keys():
                    data.append([algo,date] + list(statistics_dict[algo][date].values()))

            statistics_df = pd.DataFrame(data=data, columns=columns)

            print(statistics_df)
            statistics_df['episode reward'].plot()
            LOGGER.info(f"mean of episode pnl {statistics_df['episode pnl'].mean()}, and mean of hodl pnl {statistics_df['episode hodl pnl'].mean()}")
