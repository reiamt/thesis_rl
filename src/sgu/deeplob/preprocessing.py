from pathlib import Path
import sys
sys.path.append('/Users/tam/Documents/thesis/thesis_rl/src')

from datetime import datetime as dt
from datetime import timedelta

import pandas as pd

from gym_trading.utils.data_pipeline import DataPipeline

datapipeline = DataPipeline(None)

start_date = dt(2020,1,1) #of fitting, ie training starts on one day later
num_days = 2

paths = ['XBTUSD_' + (start_date+timedelta(i)).strftime("%Y-%m-%d") + '.csv.xz' 
         for i in range(num_days+1)]

normalized_dfs = pd.DataFrame()
for i in range(num_days):
    _, _, normalized_data = \
        datapipeline.load_environment_data(
            fitting_file=paths[i],#'XBTUSD_2020-01-02.csv.xz',
            testing_file=paths[i+1],#'XBTUSD_2020-01-03.csv.xz',
            include_imbalances=False,
            as_pandas=True
        )
    
    normalized_dfs = pd.concat([normalized_dfs, normalized_data])

# Keep only 10 Levels of LOB
def filter_strings(strings):
    filtered_strings = []
    for string in strings:
        last_part = string.split('_')[-1]
        if ('ofi' in string) or (not last_part.isdigit() or int(last_part) > 10):
            continue
        filtered_strings.append(string)
    return filtered_strings

cols = normalized_dfs.columns
cols_red = filter_strings(cols)
normalized_dfs_red = normalized_dfs[cols_red]

