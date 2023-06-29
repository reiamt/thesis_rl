import ccxt
from time import sleep
from datetime import datetime
from multiprocessing import Process, Queue
import pandas as pd

'''
{'symbol': 'XBTUSD', 'bids': [[27151.0, 38000.0], [27150.0, 18300.0], 
[27149.5, 11100.0], [27149.0, 17800.0], [27148.5, 15100.0], [27148.0, 23800.0], 
[27147.5, 7300.0], [27147.0, 5800.0], [27146.5, 6400.0], [27146.0, 6700.0], 
[27145.5, 10300.0], [27143.5, 31200.0], [27143.0, 51200.0], [27142.5, 22400.0], 
[27142.0, 18600.0], [27141.5, 30100.0], [27140.5, 41000.0], [27140.0, 107100.0], 
[27139.5, 78700.0], [27139.0, 68800.0]], 'asks': [[27151.5, 30300.0], 
[27154.0, 100.0], [27155.0, 1900.0], [27155.5, 3000.0], [27156.5, 17800.0], 
[27157.0, 24700.0], [27157.5, 50900.0], [27158.0, 17900.0], [27158.5, 10500.0], 
[27159.0, 11000.0], [27159.5, 17800.0], [27160.0, 18000.0], [27160.5, 10000.0], 
[27161.0, 17400.0], [27161.5, 28700.0], [27162.5, 17800.0], [27163.0, 59300.0], 
[27163.5, 103500.0], [27164.0, 59800.0], [27164.5, 22600.0]], 'timestamp': None, 
'datetime': None, 'nonce': None}
'''



def receive_orderbook(queue, exchange):
    while True:
        queue.put(exchange.fetch_order_book('XBTUSD', limit = 20))

def merge_and_save_orderbook(queue):
    columns = [datetime.now()]
    columns += [side+'level_'+level for side, level in zip(['bid']*20+['ask']*20)]
    df = pd.DataFrame()
    while True:
        orderbook_snapshot = queue.get()
        
        



# Create a BitMEX exchange instance
bitmex = ccxt.bitmex()

while True:
    ob = bitmex.fetch_order_book('XBTUSD', limit = 20)
    print(ob)



