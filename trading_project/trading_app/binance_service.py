"""
binance_service.py

Модуль для работы с API Binance. Содержит класс BinanceModel, который предоставляет методы для:
- Получения списка доступных символов на бирже Binance.
- Получения диапазона доступных дат для конкретного торгового символа.
- Получения исторических данных по выбранному символу и интервалу в заданный период.

Класс BinanceModel позволяет легко интегрировать работу с биржей Binance в другие части приложения, упрощая доступ к рыночным данным.

Используемые библиотеки:
- requests: для выполнения HTTP-запросов к API Binance.
- pandas: для обработки и структурирования полученных данных.

Ключевые методы:
- get_symbols: Возвращает список всех торговых символов с биржи Binance.
- get_available_date_range: Возвращает самый ранний и самый поздний доступный временной диапазон для символа и интервала.
- get_historical_data: Получает исторические данные по символу и интервалу в указанный временной период.
"""

import requests
import pandas as pd

class BinanceModel:
    """
    Класс для работы с API Binance, предоставляющий методы для получения символов,
    диапазона доступных дат и исторических данных по выбранному активу и интервалу.
    """

    def __init__(self):
        """
        Конструктор класса BinanceModel.
        """
        pass

    def get_symbols(self):
        """
        Метод для получения списка всех доступных торговых символов с биржи Binance.

        Возвращает:
            list: Список строк, каждая из которых представляет торговый символ (например, 'BTCUSDT').
            В случае ошибки возвращается пустой список.
        """
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
        """
        Метод для получения диапазона доступных дат (от первой до последней свечи) для выбранного символа и интервала.

        Аргументы:
            symbol (str): Торговый символ (например, 'BTCUSDT').
            interval (str): Интервал свечей (например, '1d', '1h').

        Возвращает:
            tuple: Кортеж из двух значений pandas.Timestamp (начальная дата, конечная дата).
            В случае ошибки возвращает (None, None).
        """
        try:
            # Получаем самую раннюю свечу
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

            # Получаем самую последнюю свечу
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
        """
        Метод для получения исторических данных по выбранному символу и интервалу за указанный период.

        Аргументы:
            symbol (str): Торговый символ (например, 'BTCUSDT').
            interval (str): Интервал свечей (например, '1d', '1h').
            start_time (str): Начальная дата в формате 'YYYY-MM-DD'.
            end_time (str): Конечная дата в формате 'YYYY-MM-DD'.

        Возвращает:
            pd.DataFrame: Датафрейм с историческими данными (время открытия, закрытия, цены, объемы и т.д.).
            В случае отсутствия данных возвращается пустой DataFrame.
        """
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

            # Избегаем дублирования последней свечи
            last_close_time = data[-1][6]
            start_timestamp = last_close_time + 1

            # Прерываем, если достигли конца диапазона или данных меньше максимального лимита
            if start_timestamp >= end_timestamp or len(data) < 1000:
                break

        if not all_data:
            print(f"Нет данных для периода {start_time} - {end_time} для {symbol}.")
            return pd.DataFrame()

        # Преобразуем данные в DataFrame
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
