import pandas as pd
import numpy as np
from datetime import datetime as dt
from datetime import timedelta
from gym_trading.utils.data_pipeline import DataPipeline

start_date = dt(2020,1,1)
num_days = 2
paths = [
    'XBTUSD_' + (start_date+timedelta(i)).strftime("%Y-%m-%d") + '.csv.xz' 
    for i in range(num_days+1)
]
data_pipeline = DataPipeline(alpha=None)

midpoints = pd.Series(dtype=np.float64)
data = pd.DataFrame(dtype=np.float32)
norm_data = pd.DataFrame(dtype=np.float32)

day_indices_dict = {}.fromkeys(range(num_days))
print(day_indices_dict)

for i in range(num_days):
    input_args = {
        "fitting_file": paths[i],
        "testing_file": paths[i+1],
        "include_imbalances": False,
        "as_pandas": True
    }
    tmp_midpoints, tmp_data, tmp_norm_data = \
        data_pipeline.load_environment_data(**input_args)
    day_indices_dict[i] = tmp_norm_data.index
    print(f"midpoint index {tmp_midpoints.shape}")

    
    midpoints = pd.concat([midpoints, tmp_midpoints])
    data = pd.concat([data, tmp_data])
    norm_data = pd.concat([norm_data, tmp_norm_data])
    

