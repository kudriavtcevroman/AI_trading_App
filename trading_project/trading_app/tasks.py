"""
tasks.py

Модуль содержит задачи для асинхронного выполнения с использованием Celery. Включает задачи для обучения 
и тестирования моделей машинного обучения на основе данных, а также сохранение и восстановление данных модели 
из MongoDB.

Основные задачи:
- train_model: Обучение модели на основе набора данных и сохранение результатов.
- test_model: Тестирование обученной модели на новом наборе данных и генерация торговых сигналов.

Используемые технологии:
- PyTorch для создания и обучения LSTM-модели.
- MongoDB для сохранения и восстановления данных модели.
- Celery для асинхронного выполнения задач.
"""

import logging
import io
from celery import shared_task
from django.db import transaction
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pymongo import MongoClient
from bson.objectid import ObjectId
import ta
from .models import TrainingSession, TrainedModel, TradeSignal, DataSet, AssetHistory
from importlib.metadata import version, PackageNotFoundError
from django.utils import timezone

logger = logging.getLogger(__name__)

# Настройка соединения с MongoDB
client = MongoClient('mongodb://localhost:27017/')
mongodb = client['AI_Trading_app']
model_data_collection = mongodb['model_data']

def save_model_data(model_data):
    """
    Сохраняет данные модели в MongoDB и возвращает идентификатор сохранённых данных.
    """
    result = model_data_collection.insert_one({'model_data': model_data})
    return str(result.inserted_id)

def get_model_data(model_data_id):
    """
    Извлекает данные модели из MongoDB по идентификатору.
    """
    data = model_data_collection.find_one({'_id': ObjectId(model_data_id)})
    return data['model_data'] if data else None

@shared_task
def train_model(training_session_id):
    """
    Задача для обучения модели LSTM на наборе данных. Использует PyTorch для обучения модели
    и сохраняет результаты в базу данных. Также добавляет выбранные пользователем индикаторы.

    Параметры:
    - training_session_id: Идентификатор сессии обучения.
    """
    training_session = TrainingSession.objects.get(id=training_session_id)
    training_session.status = 'running'
    training_session.save()

    try:
        # Логирование версии библиотеки 'ta' для индикаторов
        try:
            ta_version = version("ta")
        except PackageNotFoundError:
            ta_version = "Unknown"
        logger.info(f"ta version: {ta_version}")

        # Логирование пути модуля 'ta' для диагностики
        logger.info(f"ta module path: {getattr(ta, '__file__', 'Unknown')}")

        # Логирование доступных атрибутов модуля 'ta'
        logger.info(f"ta attributes: {dir(ta)}")

        # Загрузка данных
        dataset = training_session.dataset
        candles = dataset.candles.all().values()
        df = pd.DataFrame(list(candles))

        # Логирование загруженных колонок
        logger.info(f"Loaded columns: {df.columns.tolist()}")

        # Переименование столбцов для согласованности
        rename_mapping = {
            'open_price': 'open',
            'high_price': 'high',
            'low_price': 'low',
            'close_price': 'close',
            'volume': 'volume'
        }
        df.rename(columns=rename_mapping, inplace=True)
        logger.info(f"Renamed columns: {df.columns.tolist()}")

        # Проверка наличия необходимых столбцов
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Отсутствует необходимый столбец: {col}")

        df[required_columns] = df[required_columns].astype(float)
        logger.info("Converted numeric columns to float.")

        # Добавление индикаторов, если они выбраны
        indicators = training_session.indicators
        if indicators:
            for indicator in indicators:
                if indicator == 'EMA':
                    df['EMA'] = df['close'].ewm(span=14, adjust=False).mean()
                elif indicator == 'SMA':
                    df['SMA'] = df['close'].rolling(window=14).mean()
                elif indicator == 'RSI':
                    delta = df['close'].diff()
                    gain = delta.where(delta > 0, 0).fillna(0)
                    loss = (-delta.where(delta < 0, 0)).fillna(0)
                    avg_gain = gain.rolling(window=14).mean()
                    avg_loss = loss.rolling(window=14).mean()
                    rs = avg_gain / avg_loss
                    df['RSI'] = 100 - (100 / (1 + rs))
                elif indicator == 'On-Balance Volume':
                    df['On-Balance Volume'] = (df['volume'] * ((df['close'] - df['open']) > 0).astype(int)).cumsum()
                elif indicator == 'Stochastic Oscillator':
                    low_min = df['low'].rolling(window=14).min()
                    high_max = df['high'].rolling(window=14).max()
                    df['Stochastic Oscillator'] = 100 * ((df['close'] - low_min) / (high_max - low_min))
                elif indicator == 'Bollinger Bands':
                    df['SMA_20'] = df['close'].rolling(window=20).mean()
                    df['Bollinger_High'] = df['SMA_20'] + (df['close'].rolling(window=20).std() * 2)
                    df['Bollinger_Low'] = df['SMA_20'] - (df['close'].rolling(window=20).std() * 2)
                elif indicator == 'MACD':
                    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
                    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
                    df['MACD'] = ema_12 - ema_26
                elif indicator == 'Average Directional Index':
                    adx = ta.trend.ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
                    df['ADX'] = adx.adx()
                elif indicator == 'Standard Deviation':
                    df['Standard Deviation'] = df['close'].rolling(window=14).std()
                else:
                    logger.warning(f"Неизвестный индикатор: {indicator}")

        # Подготовка данных для модели
        feature_columns = ['open', 'high', 'low', 'close', 'volume'] + [ind for ind in indicators if ind != 'Volume']

        # Проверка наличия всех feature_columns
        for col in feature_columns:
            if col not in df.columns:
                logger.warning(f"Столбец {col} отсутствует после добавления индикаторов. Заполнение значениями по умолчанию.")
                df[col] = 0.0

        data = df[feature_columns].fillna(0).values

        # Нормализация данных
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_normalized = scaler.fit_transform(data)

        # Подготовка данных для LSTM
        sequence_length = 50
        result = []
        for index in range(len(data_normalized) - sequence_length):
            result.append(data_normalized[index: index + sequence_length])

        result = np.array(result)

        # Разделение на тренировочные и тестовые данные
        train_size = int(len(result) * 0.8)
        train_data = result[:train_size]
        test_data = result[train_size:]

        X_train = train_data[:, :-1]
        y_train = train_data[:, -1, 3]  # Предсказание 'close' цены
        X_test = test_data[:, :-1]
        y_test = test_data[:, -1, 3]

        X_train = torch.from_numpy(X_train).float()
        y_train = torch.from_numpy(y_train).float()
        X_test = torch.from_numpy(X_test).float()
        y_test = torch.from_numpy(y_test).float()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X_train, y_train, X_test, y_test = X_train.to(device), y_train.to(device), X_test.to(device), y_test.to(device)

        # Определение LSTM-модели
        class LSTMModel(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, output_size):
                super(LSTMModel, self).__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                self.fc = nn.Linear(hidden_size, output_size)

            def forward(self, x):
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
                c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
                out, _ = self.lstm(x, (h0, c0))
                out = self.fc(out[:, -1, :])
                return out

        input_size = X_train.shape[2]
        model = LSTMModel(input_size, hidden_size=64, num_layers=2, output_size=1)
        model.to(device)

        # Обучение модели
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=training_session.learning_rate)
        epochs = training_session.epochs
        batch_size = training_session.batch_size

        training_session.history = []
        training_session.save()

        for epoch in range(epochs):
            model.train()
            training_session.current_epoch = epoch + 1
            training_session.progress = (epoch + 1) / epochs * 100
            training_session.save()

            permutation = torch.randperm(X_train.size()[0])
            epoch_loss = 0

            for i in range(0, X_train.size(0), batch_size):
                indices = permutation[i:i + batch_size]
                batch_X, batch_y = X_train[indices], y_train[indices]

                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            # Логирование результатов обучения
            logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

            # Тестирование модели
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test).squeeze().cpu().numpy()
                y_test_cpu = y_test.cpu().numpy()
                test_mse = mean_squared_error(y_test_cpu, test_outputs)
                test_mae = mean_absolute_error(y_test_cpu, test_outputs)
                test_rmse = np.sqrt(test_mse)

                # Преобразование метрик в стандартные типы Python
                test_mse = float(test_mse)
                test_mae = float(test_mae)
                test_rmse = float(test_rmse)

                # Сохранение метрик в историю
                training_session.history.append({
                    'epoch': epoch + 1,
                    'mse': test_mse,
                    'mae': test_mae,
                    'rmse': test_rmse
                })
                training_session.mse = test_mse
                training_session.mae = test_mae
                training_session.rmse = test_rmse
                training_session.save()

                logger.info(f"Epoch {epoch + 1}/{epochs}, Test MSE: {test_mse:.4f}, Test MAE: {test_mae:.4f}, Test RMSE: {test_rmse:.4f}")

        # После обучения сохраните модель
        buffer = io.BytesIO()
        torch.save({
            'model_state_dict': model.state_dict(),
            'scaler': scaler,
            'input_size': input_size,
            'hidden_size': 64,
            'num_layers': 2,
            'indicators': indicators,
            'feature_columns': feature_columns,
        }, buffer)
        model_data = buffer.getvalue()
        model_data_id = save_model_data(model_data)

        # Сохранение модели в базе данных PostgreSQL
        with transaction.atomic():
            model_name = f"Model #{training_session.id}_MSE{training_session.mse:.4f}_RMSE{training_session.rmse:.4f}_MAE{training_session.mae:.4f}"
            trained_model = TrainedModel.objects.create(
                user=training_session.user,
                name=model_name,
                training_session=training_session,
                model_data_id=model_data_id,
            )
            trained_model.save()

            training_session.status = 'completed'
            training_session.save()

        logger.info(f"Training session {training_session_id} completed successfully.")

    except ValueError as ve:
        logger.error(f"ValueError in training_model task: {ve}")
        training_session.status = 'failed'
        training_session.save()
        raise ve
    except TypeError as te:
        logger.error(f"TypeError in training_model task: {te}")
        training_session.status = 'failed'
        training_session.save()
        raise te
    except Exception as e:
        logger.error(f"Unexpected error in training_model task: {e}")
        training_session.status = 'failed'
        training_session.save()
        raise e


@shared_task
def test_model(trained_model_id, dataset_id):
    """
    Задача для тестирования обученной модели на наборе данных и генерации торговых сигналов.

    Параметры:
    - trained_model_id: Идентификатор обученной модели.
    - dataset_id: Идентификатор набора данных для тестирования.
    """
    logger.info(f"Starting test_model task with trained_model_id={trained_model_id}, dataset_id={dataset_id}")
    try:
        # Получение обученной модели и набора данных
        trained_model = TrainedModel.objects.get(id=trained_model_id)
        dataset = DataSet.objects.get(id=dataset_id)
        candles = AssetHistory.objects.filter(dataset=dataset).order_by('timestamp')
        logger.info(f"Retrieved {candles.count()} candles from dataset '{dataset.name}'")

        # Очистка существующих сигналов
        TradeSignal.objects.filter(model=trained_model).delete()
        logger.info("Existing signals deleted.")

        # Загрузка сохраненной модели из MongoDB
        model_data = get_model_data(trained_model.model_data_id)
        if model_data is None:
            logger.error("Model data not found in MongoDB")
            return

        buffer = io.BytesIO(model_data)
        checkpoint = torch.load(buffer, map_location=torch.device('cpu'))

        # Восстановление модели
        input_size = checkpoint['input_size']
        hidden_size = checkpoint['hidden_size']
        num_layers = checkpoint['num_layers']
        indicators = checkpoint['indicators']
        feature_columns = checkpoint['feature_columns']
        scaler = checkpoint['scaler']

        class LSTMModel(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, output_size):
                super(LSTMModel, self).__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                self.fc = nn.Linear(hidden_size, output_size)

            def forward(self, x):
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
                c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
                out, _ = self.lstm(x, (h0, c0))
                out = self.fc(out[:, -1, :])
                return out

        model = LSTMModel(input_size, hidden_size, num_layers, output_size=1)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Подготовка данных для тестирования
        df = pd.DataFrame(list(candles.values()))
        df['open'] = df['open_price'].astype(float)
        df['high'] = df['high_price'].astype(float)
        df['low'] = df['low_price'].astype(float)
        df['close'] = df['close_price'].astype(float)
        df['volume'] = df['volume'].astype(float)

        # Добавление индикаторов
        if indicators:
            if 'EMA' in indicators:
                df['EMA'] = df['close'].ewm(span=20, adjust=False).mean()
            if 'RSI' in indicators:
                delta = df['close'].diff()
                up = delta.clip(lower=0)
                down = -delta.clip(upper=0)
                roll_up = up.rolling(window=14).mean()
                roll_down = down.rolling(window=14).mean()
                rs = roll_up / roll_down
                df['RSI'] = 100.0 - (100.0 / (1.0 + rs))

        # Проверка наличия всех необходимых столбцов
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0.0

        data = df[feature_columns].fillna(0).values
        data_normalized = scaler.transform(data)

        # Подготовка последовательностей
        sequence_length = 50
        result = []
        for index in range(len(data_normalized) - sequence_length):
            result.append(data_normalized[index: index + sequence_length])
        result = np.array(result)
        X_test = torch.from_numpy(result).float()

        # Прогнозирование
        with torch.no_grad():
            predictions = model(X_test).squeeze().numpy()

        # Генерация торговых сигналов
        logger.info("Generating trade signals")
        current_position = None
        entry_price = 0.0
        STOP_LOSS_PERCENTAGE = 0.02
        min_candles_between_signals = 5
        last_open_signal_index = -min_candles_between_signals

        for i in range(len(predictions) - 1):
            if current_position is None:
                if i - last_open_signal_index < min_candles_between_signals:
                    continue
                if predictions[i + 1] > predictions[i]:
                    trade_type = 'LONG OPEN'
                    current_position = 'LONG'
                    entry_price = df['open'].iloc[i + sequence_length]
                    last_open_signal_index = i
                elif predictions[i + 1] < predictions[i]:
                    trade_type = 'SHORT OPEN'
                    current_position = 'SHORT'
                    entry_price = df['open'].iloc[i + sequence_length]
                    last_open_signal_index = i
                else:
                    trade_type = 'HOLD'
            elif current_position == 'LONG':
                current_price = df['close'].iloc[i + sequence_length]
                price_change = (current_price - entry_price) / entry_price
                if price_change <= -STOP_LOSS_PERCENTAGE:
                    trade_type = 'STOP LOSS'
                    current_position = None
                elif predictions[i + 1] < predictions[i]:
                    trade_type = 'LONG CLOSE'
                    current_position = None
                else:
                    trade_type = 'HOLD'
            elif current_position == 'SHORT':
                current_price = df['close'].iloc[i + sequence_length]
                price_change = (entry_price - current_price) / entry_price
                if price_change <= -STOP_LOSS_PERCENTAGE:
                    trade_type = 'STOP LOSS'
                    current_position = None
                elif predictions[i + 1] > predictions[i]:
                    trade_type = 'SHORT CLOSE'
                    current_position = None
                else:
                    trade_type = 'HOLD'
            else:
                trade_type = 'HOLD'

            if trade_type == 'HOLD':
                continue

            timestamp = df['timestamp'].iloc[i + sequence_length]
            if isinstance(timestamp, pd.Timestamp):
                timestamp = timestamp.to_pydatetime()
            if timezone.is_naive(timestamp):
                timestamp = timezone.make_aware(timestamp, timezone.get_current_timezone())

            stop_loss_reached = trade_type == 'STOP LOSS'

            existing_signals = TradeSignal.objects.filter(
                model=trained_model,
                date=timestamp,
                trade_type=trade_type
            )
            if not existing_signals.exists():
                signal = TradeSignal(
                    model=trained_model,
                    date=timestamp,
                    trade_type=trade_type,
                    open_price=df['open'].iloc[i + sequence_length],
                    close_price=df['close'].iloc[i + sequence_length],
                    stop_loss_reached=stop_loss_reached
                )
                signal.save()

        logger.info("Trade signal generation completed successfully.")

    except Exception as e:
        logger.error(f"Error in test_model task: {e}", exc_info=True)
        raise e
