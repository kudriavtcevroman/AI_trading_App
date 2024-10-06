import requests
import pandas as pd

class BinanceModel:
    def __init__(self):
        pass  # Убираем инициализацию базы данных

    def get_symbols(self):
        try:
            url = "https://api.binance.com/api/v3/exchangeInfo"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                symbols = [item['symbol'] for item in data['symbols']]
                return symbols
            else:
                print(f"Ошибка получения символов: {response.status_code}")
                return []
        except Exception as e:
            print(f"Ошибка при запросе к Binance API: {str(e)}")
            return []

    def get_available_date_range(self, symbol, interval):
        try:
            # Получаем самую раннюю дату (первую свечу)
            url = "https://api.binance.com/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': 1,
                'startTime': 0  # Начало времён
            }
            response = requests.get(url, params=params)
            data = response.json()
            earliest_time = pd.to_datetime(data[0][0], unit='ms')

            # Получаем самую последнюю дату (последнюю свечу)
            params['startTime'] = None
            params['endTime'] = int(pd.Timestamp.now().timestamp() * 1000)
            response = requests.get(url, params=params)
            data = response.json()
            latest_time = pd.to_datetime(data[-1][6], unit='ms')

            return earliest_time, latest_time

        except Exception as e:
            print(f"Ошибка при получении диапазона дат: {str(e)}")
            return None, None

    def get_historical_data(self, symbol, interval, start_time, end_time):
        url = "https://api.binance.com/api/v3/klines"
        all_data = []
        start_timestamp = int(pd.Timestamp(start_time).timestamp() * 1000)
        end_timestamp = int(pd.Timestamp(end_time).timestamp() * 1000)

        while True:
            params = {
                'symbol': symbol,
                'interval': interval,
                'startTime': start_timestamp,
                'endTime': end_timestamp,
                'limit': 1000
            }
            response = requests.get(url, params=params)
            if response.status_code != 200:
                print(f"Ошибка API Binance: {response.status_code} - {response.text}")
                break

            data = response.json()
            if not data:
                break

            all_data.extend(data)

            last_close_time = data[-1][6]
            start_timestamp = last_close_time + 1  # Избегаем дублирования последней свечи

            if start_timestamp >= end_timestamp:
                break

            if len(data) < 1000:
                break

        if not all_data:
            print(f"Нет данных для периода {start_time} - {end_time} для {symbol}.")
            return pd.DataFrame()

        df = pd.DataFrame(all_data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])

        # Преобразование временных меток и числовых значений
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_columns] = df[numeric_columns].astype(float)

        return df
