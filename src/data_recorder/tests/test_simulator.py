from datetime import datetime as dt

from configurations import TIMEZONE
from data_recorder.database.simulator import Simulator


def test_get_tick_history() -> None:
    """
    Test case to query Arctic TickStore
    """
    start_time = dt.now(tz=TIMEZONE)

    sim = Simulator()
    query = {
        'ccy': ['BTC-USD'],
        'start_date': 20181231,
        'end_date': 20190102
    }
    tick_history = sim.db.get_tick_history(query=query)
    print('\n{}\n'.format(tick_history))

    elapsed = (dt.now(tz=TIMEZONE) - start_time).seconds
    print('Completed %s in %i seconds' % (__name__, elapsed))
    print('DONE. EXITING %s' % __name__)


def test_get_orderbook_snapshot_history() -> None:
    """
    Test case to export testing/training data for reinforcement learning
    """
    start_time = dt.now(tz=TIMEZONE)

    sim = Simulator()
    query = {
        'ccy': ['BTC-USD'],
        'start_date': 20230409,
        'end_date': 20230413
    }
    orderbook_snapshot_history = sim.get_orderbook_snapshot_history(query=query)
    if orderbook_snapshot_history is None:
        print('Exiting: orderbook_snapshot_history is NONE')
        return

    filename = 'test_' + '{}_{}'.format(query['ccy'][0], query['start_date'])
    sim.export_to_csv(data=orderbook_snapshot_history,
                      filename=filename, compress=True)

    elapsed = (dt.now(tz=TIMEZONE) - start_time).seconds
    print('Completed %s in %i seconds' % (__name__, elapsed))
    print('DONE. EXITING %s' % __name__)


def test_extract_features() -> None:
    """
    Test case to export *multiple* testing/training data sets for reinforcement learning
    """
    start_time = dt.now(tz=TIMEZONE)

    sim = Simulator()

    for ccy in ['BTC-USD']:
        # for ccy, ccy2 in [('LTC-USD', 'tLTCUSD')]:
        query = {
            'ccy': [ccy],  # ccy2],  # parameter must be a list
            'start_date': 20230409,  # parameter format for dates
            'end_date': 20230412,  # parameter format for dates
        }
        sim.extract_features(query)

    elapsed = (dt.now(tz=TIMEZONE) - start_time).seconds
    print('Completed %s in %i seconds' % (__name__, elapsed))
    print('DONE. EXITING %s' % __name__)


def test_merge_files() -> None:
    """
    test case to merge multiple files to train and test the agent on more than just one day
    """
    start_time = dt.now(tz=TIMEZONE)

    sim = Simulator()
    ccy = ['XBTUSD']
    sample_path = 'data_recorder/database/data_exports/paper_data/XBTUSD_'
    start_date, end_date = dt(2020,1,1), dt(2020,1,2)

    sim.merge_files(ccy, sample_path, start_date, end_date)

    elapsed = (dt.now(tz=TIMEZONE) - start_time).seconds
    print('Completed %s in %i seconds' % (__name__, elapsed))
    print('DONE. EXITING %s' % __name__)

if __name__ == '__main__':
    """
    Entry point of tests application
    """
    # test_get_tick_history()
    #test_get_orderbook_snapshot_history()
    #test_extract_features()
    test_merge_files()
