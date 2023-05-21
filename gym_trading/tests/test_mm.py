from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
print(sys.path)
import gym

from gym_trading.envs.market_maker import MarketMaker


env_args = {
    "symbol": 'XBTUSD',
    "fitting_file": 'XBTUSD_20200101_20200102_merge.csv.xz',
    "testing_file": 'XBTUSD_2020-01-03.csv.xz',
    "max_position": 5.,
    "window_size": 100,
    "seed": 1,
    "action_repeats": 5,
    "training": True,
    "format_3d": False,
    "reward_type": 'trade_completion',
    "transaction_fee": True
}
#"id": 'market-maker-v0' 
#mm = gym.make(**env_args)
mm = MarketMaker(**env_args)

obs = mm.reset()
print(len(obs))
print(len(obs[0]))

obss = mm._get_observation()
print((obss==mm.data_buffer).all())


#mm.reset()

#obs = mm._get_observation()
#print(obs)