import logging
import os

import pytz as tz

# singleton for logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s, %(filename)s:%(lineno)s] %(message)s')
LOGGER = logging.getLogger('crypto_rl_log')

# ./recorder.py
SNAPSHOT_RATE = 1.0  # For example, 0.25 = 4x per second
BASKET = [('BTC-USD', 'tBTCUSD'),
          # ('ETH-USD', 'tETHUSD'),
          # ('LTC-USD', 'tLTCUSD')
          ]
#BASKET = 'tBTCUSD' #bitfinex

# ./data_recorder/connector_components/client.py
COINBASE_ENDPOINT = 'wss://ws-feed.pro.coinbase.com'
COINBASE_BOOK_ENDPOINT = 'https://api.pro.coinbase.com/products/%s/book'
BITFINEX_ENDPOINT = 'wss://api.bitfinex.com/ws/2'
MAX_RECONNECTION_ATTEMPTS = 100

# ./data_recorder/connector_components/book.py
MAX_BOOK_ROWS = 20 #5
INCLUDE_ORDERFLOW = True

# ./data_recorder/database/database.py
BATCH_SIZE = 100_000
RECORD_DATA = False
MONGO_ENDPOINT = 'mongodb://127.0.0.1:27017/?directConnection=true&serverSelectionTimeoutMS=2000&appName=mongosh+1.8.1' #'mongodb+srv://reiamt:tGs4FTMFPDALDy1X@cryptotick.euhjswo.mongodb.net/cryptotick'
ARCTIC_NAME = 'TickStore'
TIMEZONE = tz.utc

# ./data_recorder/database/simulator.py
SNAPSHOT_RATE_IN_MICROSECONDS = 1000000  # 1 second

# ./gym_trading/utils/broker.py
MARKET_ORDER_FEE = 0.00075 #ie 0.075% 0.0020
LIMIT_ORDER_FEE = - 0.00025 #ie -0.025%
SLIPPAGE = 0.0005

# ./indicators/*
INDICATOR_WINDOW = [60 * i for i in [5, 15]]  # Convert minutes to seconds
INDICATOR_WINDOW_MAX = max(INDICATOR_WINDOW)
INDICATOR_WINDOW_FEATURES = [f'_{i}' for i in [5, 15]]  # Create labels
EMA_ALPHA = None #0.99  # [0.9, 0.99, 0.999, 0.9999]

# agent penalty configs
ENCOURAGEMENT = 0.000000000001

# Data Directory
ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(ROOT_PATH, 'data_recorder', 'database', 'data_exports')
