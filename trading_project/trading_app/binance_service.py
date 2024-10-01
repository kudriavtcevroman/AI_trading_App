import requests
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine

class BinanceModel:
    def __init__(self, db_url='sqlite:///binance_data.db'):
        self.engine = create_engine(db_url)

    def get_symbols(self):
        # Получение всех торговых пар
        url = "https://api.binance.com/api/v3/exchangeInfo"
        response = requests.get(url)
        data = response.json()
        symbols = [item['symbol'] for item in data['symbols']]
        return symbols

    def get_available_dates(self, symbol, interval):
        # Получение первого доступного timestamp и текущей даты
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit=1&startTime=0"
        response = requests.get(url)
        data = response.json()

        if not data or 'code' in data:
            return None, None

        start_time = pd.to_datetime(data[0][0], unit='ms').date()  # Первый доступный день
        end_time = datetime.now().date()  # Текущая дата

        return start_time, end_time

    def get_historical_data(self, symbol, interval, start_time, end_time):
        # Получаем исторические данные
        url = f"https://api.binance.com/api/v3/klines"
        all_data = []
        current_start_time = int(pd.Timestamp(start_time).timestamp() * 1000)
        current_end_time = int(pd.Timestamp(end_time).timestamp() * 1000)

        while True:
            params = {
                'symbol': symbol,
                'interval': interval,
                'startTime': current_start_time,
                'endTime': current_end_time,
                'limit': 1000
            }
            response = requests.get(url, params=params)
            data = response.json()
            if not data:
                break

            all_data.extend(data)
            current_start_time = data[-1][6]  # close_time последней строки

            if len(data) < 1000:
                break

        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(all_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.strftime('%d.%m.%Y %H:%M:%S')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms').dt.strftime('%d.%m.%Y %H:%M:%S')
        return df

    def save_to_db(self, df, table_name):
        if df.empty:
            print(f"Пустой DataFrame. Таблица {table_name} не будет создана.")
            return
        df.to_sql(table_name, con=self.engine, if_exists='replace', index=False)
