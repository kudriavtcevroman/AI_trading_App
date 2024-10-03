from celery import shared_task
from .models import TrainingSession, TrainedModel
from django.core.files.base import ContentFile
import torch
import torch.nn as nn
import torch.optim as optim
import io
import numpy as np
import pandas as pd

@shared_task
def train_model(training_session_id):
    training_session = TrainingSession.objects.get(id=training_session_id)
    training_session.status = 'running'
    training_session.save()

    try:
        # Загрузите данные
        dataset = training_session.dataset
        candles = dataset.candles.all().values()
        df = pd.DataFrame(list(candles))

        # Примените выбранные индикаторы
        # Например, если пользователь выбрал EMA и RSI
        indicators = training_session.indicators
        # Добавьте код для расчета индикаторов и добавления их в DataFrame

        # Подготовка данных для LSTM-модели
        # Разделите данные на X и y, нормализуйте, создайте последовательности для LSTM

        # Создание LSTM-модели
        class LSTMModel(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super(LSTMModel, self).__init__()
                self.hidden_size = hidden_size
                self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
                self.fc = nn.Linear(hidden_size, output_size)

            def forward(self, x):
                out, _ = self.lstm(x)
                out = out[:, -1, :]
                out = self.fc(out)
                return out

        input_size = ...  # Зависит от количества признаков
        hidden_size = 64
        output_size = 1  # Прогнозируемое значение (например, вероятность роста цены)

        model = LSTMModel(input_size, hidden_size, output_size)

        # Настройка обучения
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=training_session.learning_rate)

        # Обучение модели
        epochs = training_session.epochs
        for epoch in range(epochs):
            training_session.current_epoch = epoch + 1
            # Обучение на батчах
            # Обновление прогресса обучения
            training_session.progress = (epoch + 1) / epochs * 100

            # Добавьте код обучения модели

            # Расчет точности на валидационном наборе
            accuracy = ...  # Вычислите точность модели
            training_session.accuracy = accuracy
            training_session.save()

        # После обучения сохраните модель
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        model_file = ContentFile(buffer.getvalue())

        trained_model = TrainedModel.objects.create(
            user=training_session.user,
            name=f"Model {training_session.id}",
            training_session=training_session,
            model_file=model_file
        )

        training_session.status = 'completed'
        training_session.save()

    except Exception as e:
        training_session.status = 'failed'
        training_session.save()
        raise e