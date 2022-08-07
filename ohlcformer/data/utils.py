import json
import os
import random
from pathlib import Path
from datetime import datetime


def read_stocks_dir(path: Path, verbose=False):
    stocks = []

    for stock in os.listdir(path):
        stock_path = path / str(stock)
        stock_data = []

        with open(stock_path) as file:
            stock_data = json.load(file)

        stocks.append(stock_data)

        if verbose:
            print(f'{stock} : {len(stock_data)}')

    return stocks


def convert_stock_data(stocks_path: Path, preprocessed_stocks_path: Path, verbose=False):
    for stock in os.listdir(stocks_path):
        stock = str(stock)
        stock_dir = stocks_path / stock
        stock_data = []

        for stock_period in os.listdir(stock_dir):
            stock_period_path = stock_dir / str(stock_period)
            data = []

            with open(stock_period_path) as file:
                data = json.load(file)

            for row in data:
                row['date'] = datetime.strptime(row['date'], '%Y-%m-%dT%H:%M:%S.%fZ').timestamp()
                stock_data.append(row)

        if verbose:
            print(f'{stock} : {len(stock_data)}')

        output_file = (preprocessed_stocks_path / stock).with_suffix('.json')
        output_file.parent.mkdir(exist_ok=True, parents=True)

        with open(output_file, 'w', encoding='utf8') as outfile:
            json.dump(stock_data, outfile, ensure_ascii=False)


def split_time_series(stocks, lens=None, seed: int = None) -> list:
    dataset = []
    if lens is None:
        lens = [60, 100, 120, 240, 480, 960, 1000, 1440, 1920, 2000]

    if seed is not None:
        rand = random.Random(seed)
    else:
        rand = random

    for stock in stocks:
        i = 0

        while i < len(stock):
            j = rand.choice(lens)
            dataset.append(stock[i:i + j])
            i = i + j

    return dataset


def convert_series_to_relative(dataset) -> list:
    relative_dataset = []

    for row in dataset:
        if len(row) < 2:
            continue

        def diff(a, b):
            return (b - a) / a

        series = [[0, 0, 0, 0]]
        prev = [row[0]['open'], row[0]['high'], row[0]['low'], row[0]['close']]

        for elem in row[1:]:
            curr = [elem['open'], elem['high'], elem['low'], elem['close']]
            series.append([diff(prev[0], curr[0]),
                           diff(prev[1], curr[1]),
                           diff(prev[2], curr[2]),
                           diff(prev[3], curr[3])
                           ])
            prev = curr

        relative_dataset.append(series)

    return relative_dataset
