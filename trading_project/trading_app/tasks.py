import logging
import subprocess
import sys
import io  # Добавьте импорт io

from celery import shared_task
from django.conf import settings
from django.core.files.base import ContentFile
from django.db import transaction

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error  # Добавьте импорт метрик

import ta  # Импортируйте 'ta' только здесь

from .models import TrainingSession, TrainedModel

# Импорт для получения версии
from importlib.metadata import version, PackageNotFoundError  # Для Python 3.8+

logger = logging.getLogger(__name__)

@shared_task
def train_model(training_session_id):
    training_session = TrainingSession.objects.get(id=training_session_id)
    training_session.status = 'running'
    training_session.save()

    try:
        # Логирование версии 'ta' с безопасным доступом
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
            'volume': 'volume'  # Уже корректное название
        }
        df.rename(columns=rename_mapping, inplace=True)
        logger.info(f"Renamed columns: {df.columns.tolist()}")

        # Проверка наличия необходимых столбцов
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Отсутствует необходимый столбец: {col}")

        # Преобразование столбцов в float
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_columns] = df[numeric_columns].astype(float)
        logger.info("Converted numeric columns to float.")

        # Добавление выбранных индикаторов в DataFrame
        indicators = training_session.indicators
        if indicators:
            for indicator in indicators:
                if indicator == 'EMA':
                    df['EMA'] = df['close'].ewm(span=14, adjust=False).mean()
                elif indicator == 'SMA':
                    df['SMA'] = df['close'].rolling(window=14).mean()
                elif indicator == 'RSI':
                    delta = df['close'].diff()
                    gain = (delta.where(delta > 0, 0)).fillna(0)
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
                    try:
                        adx = ta.trend.ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
                        df['ADX'] = adx.adx()
                    except ImportError:
                        logger.warning("Библиотека 'ta' не установлена. Установка...")
                        subprocess.check_call([sys.executable, "-m", "pip", "install", "ta"])
                        adx = ta.trend.ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
                        df['ADX'] = adx.adx()
                elif indicator == 'Standard Deviation':
                    df['Standard Deviation'] = df['close'].rolling(window=14).std()
                else:
                    logger.warning(f"Неизвестный индикатор: {indicator}")

        # Выбор необходимых столбцов для обучения
        feature_columns = ['open', 'high', 'low', 'close', 'volume'] + [ind for ind in indicators if ind != 'Volume']

        # Проверка наличия всех feature_columns
        for col in feature_columns:
            if col not in df.columns:
                logger.warning(
                    f"Столбец {col} отсутствует после добавления индикаторов. Заполнение значениями по умолчанию.")
                df[col] = 0.0  # Или другое значение по умолчанию

        # Конвертация всех числовых столбцов в float (на всякий случай)
        numeric_columns = ['open', 'high', 'low', 'close', 'volume'] + [ind for ind in indicators if ind != 'Volume']
        for col in numeric_columns:
            df[col] = df[col].astype(float)
        logger.info("Converted numeric columns to float.")

        data = df[feature_columns].fillna(0).values

        # Нормализация данных
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_normalized = scaler.fit_transform(data)

        # Подготовка данных для LSTM
        sequence_length = 50  # Длина последовательности для LSTM
        result = []
        for index in range(len(data_normalized) - sequence_length):
            result.append(data_normalized[index: index + sequence_length])

        result = np.array(result)

        # Разделение на тренировочные и тестовые данные
        train_size = int(len(result) * 0.8)
        train_data = result[:train_size]
        test_data = result[train_size:]

        # Разделение на входы и целевые значения
        X_train = train_data[:, :-1]
        y_train = train_data[:, -1, 3]  # Предположим, что предсказываем 'close' цену
        X_test = test_data[:, :-1]
        y_test = test_data[:, -1, 3]

        # Преобразование в тензоры PyTorch
        X_train = torch.from_numpy(X_train).float()
        y_train = torch.from_numpy(y_train).float()
        X_test = torch.from_numpy(X_test).float()
        y_test = torch.from_numpy(y_test).float()

        # Перенос на устройство (CPU или GPU)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X_train = X_train.to(device)
        y_train = y_train.to(device)
        X_test = X_test.to(device)
        y_test = y_test.to(device)

        # Создание LSTM-модели
        input_size = X_train.shape[2]
        hidden_size = 64  # Можно сделать параметром в TrainingSession
        num_layers = 2    # Можно сделать параметром в TrainingSession
        output_size = 1

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

        model = LSTMModel(input_size, hidden_size, num_layers, output_size)
        model.to(device)

        # Настройка обучения
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=training_session.learning_rate)

        epochs = training_session.epochs
        batch_size = training_session.batch_size

        # Инициализация истории метрик
        training_session.history = []
        training_session.save()

        # Обучение модели
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

            logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

            # Расчёт метрик на тестовых данных
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

                logger.info(
                    f"Epoch {epoch + 1}/{epochs}, Test MSE: {test_mse:.4f}, Test MAE: {test_mae:.4f}, Test RMSE: {test_rmse:.4f}")

        # После обучения сохраните модель
        buffer = io.BytesIO()
        torch.save({
            'model_state_dict': model.state_dict(),
            'scaler': scaler,
            'input_size': input_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'indicators': indicators,
            'feature_columns': feature_columns,
        }, buffer)
        model_file = ContentFile(buffer.getvalue())

        # Сохранение модели в базе данных
        with transaction.atomic():
            trained_model = TrainedModel.objects.create(
                user=training_session.user,
                name=f"Model {training_session.id}",
                training_session=training_session,
            )
            trained_model.model_file.save(f"model_{trained_model.id}.pt", model_file)

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

