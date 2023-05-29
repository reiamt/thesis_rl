import pandas as pd
import schedule
import time
import datetime
import json
import requests
import os


DATA_DIR = 'orderbook_data'


def fetch_order_book():
    url = 'https://www.bitmex.com/api/v1/orderBook/L2'
    params = {
        'symbol': 'XBTUSD',
        'depth': 20
    }
    response = requests.get(url, params=params)
    order_book = json.loads(response.text)
    return order_book


def create_dataframe():
    order_book = fetch_order_book()
    timestamp = datetime.datetime.now()
    df = pd.DataFrame(order_book)
    df['timestamp'] = timestamp
    return df


def save_dataframe(df, filename):
    print('saving dataframe...')
    filename = DATA_DIR + filename
    df.to_csv(filename, index=False, compression='xz')
    print(f'Saved {filename}')


def record_order_book(x):
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    progress_file = os.path.join(DATA_DIR, 'progress.txt')

    df = pd.DataFrame()
    row_count = 0
    last_timestamp = None

    # Load progress if it exists
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as file:
            progress = file.read().split(',')
            last_timestamp = datetime.datetime.strptime(progress[0], "%Y-%m-%d %H:%M:%S")
            row_count = int(progress[1])
            print(f"Resuming recording from {last_timestamp}, row count: {row_count}")

    def job():
        nonlocal df, row_count, last_timestamp
        if row_count == x:
            filename = f'orderbook_{last_timestamp.strftime("%Y%m%d%H%M%S")}.csv.xz'
            save_dataframe(df, filename)
            df = pd.DataFrame()
            row_count = 0

        order_book_df = create_dataframe()
        df = pd.concat([df, order_book_df])
        row_count += 1
        last_timestamp = order_book_df['timestamp'].iloc[-1]

    schedule.every(1).second.do(job)

    while True:
        try:
            schedule.run_pending()
            time.sleep(1)
        except Exception as e:
            print(f"An error occurred: {str(e)}")

            # Save progress
            with open(progress_file, 'w') as file:
                file.write(f"{last_timestamp},{row_count}")

            # Wait for a few seconds before retrying
            time.sleep(10)
            print("Restarting script...")


if __name__ == '__main__':
    record_order_book(100)
