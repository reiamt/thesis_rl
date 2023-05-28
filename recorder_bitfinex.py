import asyncio
import time
from datetime import datetime as dt
from multiprocessing import Process
from threading import Timer

from configurations import BASKET, LOGGER, SNAPSHOT_RATE
from data_recorder.bitfinex_connector.bitfinex_client import BitfinexClient

class Recorder(Process):

    def __init__(self, symbols):
        """
        Constructor of Recorder.

        :param symbols: basket of securities to record...
                        Example: symbols = [('BTC-USD, 'tBTCUSD')]
        """
        super(Recorder, self).__init__()
        self.symbols = symbols
        self.timer_frequency = SNAPSHOT_RATE
        self.workers = dict()
        self.current_time = dt.now()
        self.daemon = False

    def run(self) -> None:
        """
        New process created to instantiate limit order books for Bitfinex.

        Connections made to each exchange are made asynchronously thanks to asyncio.

        :return: void
        """
        bitfinex = self.symbols
        self.workers[bitfinex] = BitfinexClient(sym=bitfinex)

        self.workers[bitfinex].start()

        Timer(5.0, self.timer_worker,
              args=(self.workers[bitfinex],)).start()

        tasks = asyncio.gather(*[self.workers[sym].subscribe()
                                 for sym in self.workers.keys()])
        loop = asyncio.get_event_loop()
        LOGGER.info(f'Recorder: Gathered {len(self.workers.keys())} tasks')

        try:
            loop.run_until_complete(tasks)
            loop.close()
            [self.workers[sym].join() for sym in self.workers.keys()]
            LOGGER.info(f'Recorder: loop closed for {bitfinex}.')

        except KeyboardInterrupt as e:
            LOGGER.info(f"Recorder: Caught keyboard interrupt. \n{e}")
            tasks.cancel()
            loop.close()
            [self.workers[sym].join() for sym in self.workers.keys()]

        finally:
            loop.close()
            LOGGER.info(f'Recorder: Finally done for {bitfinex}.')

    def timer_worker(self,
                     bitfinexClient: BitfinexClient) -> None:
        """
        Thread worker to be invoked every N seconds (e.g., configurations.SNAPSHOT_RATE)

        :param coinbaseClient: CoinbaseClient
        :param bitfinexClient: BitfinexClient
        :return: void
        """
        Timer(self.timer_frequency, self.timer_worker,
              args=(bitfinexClient)).start()
        self.current_time = dt.now()

        if bitfinexClient.book.done_warming_up:
            """
            This is the place to insert a trading model. 
            You'll have to create your own.
            
            Example:
                orderbook_data = tuple(coinbaseClient.book, bitfinexClient.book)
                model = agent.dqn.Agent()
                fix_api = SomeFixAPI() 
                action = model(orderbook_data)
                if action is buy:
                    buy_order = create_order(pair, price, etc.)
                    fix_api.send_order(buy_order)
            
            """
            LOGGER.info(f'{bitfinexClient.book} is ready')
            # The `render_book()` method returns a numpy array of the LOB's current state,
            # as well as resets the Order Flow Imbalance trackers.
            # The LOB snapshot is in a tabular format with columns as defined in
            # `render_lob_feature_names()`
            _ = bitfinexClient.book.render_book()
        else:
            LOGGER.info('Bitfinex is still warming up...')


def main():
    LOGGER.info(f'Starting recorder with basket = {BASKET}')
    Recorder(BASKET).start()
    LOGGER.info(f'Process started up for {BASKET}')
    time.sleep(9)


if __name__ == "__main__":
    """
    Entry point of application
    """
    main()