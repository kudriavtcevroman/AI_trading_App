"""
models.py

Модуль содержит модели для хранения данных пользователей, наборов данных, истории активов,
сессий обучения и торговых сигналов.

Модели:
- UserProfile: Основные данные пользователя.
- UserAdditionalInfo: Дополнительная информация о пользователе.
- DataSet: Набор данных, содержащий информацию об активах.
- AssetHistory: Исторические данные по активам (свечи).
- TrainingSession: Сессии обучения моделей машинного обучения.
- TrainedModel: Модель, обученная на сессии обучения.
- TradeSignal: Торговые сигналы, генерируемые обученной моделью.
"""

from django.db import models
from django.core.validators import RegexValidator
from django.contrib.auth.models import User


class UserProfile(models.Model):
    """
    Модель для хранения основной информации о пользователе.
    Связана с встроенной моделью пользователя Django.
    """
    user = models.OneToOneField(User, on_delete=models.CASCADE)

    # Телефон с валидатором
    phone_regex = RegexValidator(
        regex=r'^\+?1?\d{9,15}$',
        message="Номер телефона должен быть в формате: '+999999999'. До 15 цифр."
    )
    phone_number = models.CharField(
        validators=[phone_regex],
        max_length=17,
        blank=True
    )

    # Дата рождения и пол
    birth_date = models.DateField()
    gender = models.CharField(max_length=10, choices=[('M', 'Мужчина'), ('F', 'Женщина')])

    def __str__(self):
        return f'{self.user.first_name} {self.user.last_name} ({self.user.username})'


class UserAdditionalInfo(models.Model):
    """
    Модель для хранения дополнительной информации о пользователе, такой как отчество, никнейм, город и т.д.
    """
    user_profile = models.OneToOneField(UserProfile, on_delete=models.CASCADE)

    # Необязательные поля
    middle_name = models.CharField(max_length=100, blank=True, null=True)
    nickname = models.CharField(max_length=100, blank=True, null=True)
    telegram = models.CharField(max_length=100, blank=True, null=True)
    country = models.CharField(max_length=100, blank=True, null=True)
    city = models.CharField(max_length=100, blank=True, null=True)

    def __str__(self):
        return f"Доп. информация: {self.nickname or self.user_profile.first_name}"


class DataSet(models.Model):
    """
    Модель для хранения наборов данных, загруженных пользователем.
    Содержит информацию о названии, активе и временных рамках.
    """
    user = models.ForeignKey(UserProfile, on_delete=models.CASCADE)
    name = models.CharField(max_length=255)
    asset_name = models.CharField(max_length=100)
    interval = models.CharField(max_length=10)
    start_date = models.DateField()
    end_date = models.DateField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.name} - {self.asset_name} ({self.interval})"


class AssetHistory(models.Model):
    """
    Модель для хранения исторических данных по активам (свечи).
    Включает цены открытия, закрытия, минимальные и максимальные цены, а также объемы.
    """
    dataset = models.ForeignKey(DataSet, on_delete=models.CASCADE, related_name='candles')
    timestamp = models.DateTimeField()
    open_price = models.DecimalField(max_digits=20, decimal_places=8)
    high_price = models.DecimalField(max_digits=20, decimal_places=8)
    low_price = models.DecimalField(max_digits=20, decimal_places=8)
    close_price = models.DecimalField(max_digits=20, decimal_places=8)
    volume = models.DecimalField(max_digits=20, decimal_places=8)

    def __str__(self):
        return f"{self.dataset.asset_name} ({self.timestamp})"


class TrainingSession(models.Model):
    """
    Модель для хранения сессий обучения моделей.
    Содержит информацию о параметрах обучения и статусе выполнения.
    """
    user = models.ForeignKey(UserProfile, on_delete=models.CASCADE)
    dataset = models.ForeignKey(DataSet, on_delete=models.CASCADE)
    long = models.BooleanField(default=False)
    short = models.BooleanField(default=False)
    stop_loss = models.FloatField(default=0.0)
    indicators = models.JSONField(default=list)  # Список индикаторов, выбранных для обучения
    epochs = models.IntegerField(default=10)
    batch_size = models.IntegerField(default=32)
    learning_rate = models.FloatField(default=0.001)
    status = models.CharField(max_length=20, choices=[
        ('pending', 'Pending'),
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ], default='pending')
    progress = models.FloatField(default=0.0)  # Процент выполнения
    accuracy = models.FloatField(null=True, blank=True)  # Процент успешности обучения
    created_at = models.DateTimeField(auto_now_add=True)
    current_epoch = models.IntegerField(default=0)
    mse = models.FloatField(null=True, blank=True)  # Mean Squared Error (среднеквадратичная ошибка)
    mae = models.FloatField(null=True, blank=True)  # Mean Absolute Error (средняя абсолютная ошибка)
    rmse = models.FloatField(null=True, blank=True)  # Root Mean Squared Error (корень из среднеквадратичной ошибки)
    history = models.JSONField(default=list, blank=True)  # История обучения по эпохам

    def __str__(self):
        return f"TrainingSession {self.id} for {self.user.user.username} ({self.status})"


class TrainedModel(models.Model):
    """
    Модель для хранения обученных моделей, связанных с сессией обучения.
    """
    user = models.ForeignKey(UserProfile, on_delete=models.CASCADE)
    name = models.CharField(max_length=255)
    training_session = models.ForeignKey(TrainingSession, on_delete=models.CASCADE)
    model_data_id = models.CharField(max_length=255, null=True, blank=True)  # Идентификатор данных модели в хранилище
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name


class TradeSignal(models.Model):
    """
    Модель для хранения торговых сигналов, генерируемых обученной моделью.
    Включает тип сигнала, цену открытия и закрытия, и информацию о срабатывании Stop-Loss.
    """
    model = models.ForeignKey(TrainedModel, on_delete=models.CASCADE, related_name='signals')
    date = models.DateTimeField()
    trade_type = models.CharField(max_length=20, choices=[
        ('LONG OPEN', 'LONG OPEN'),
        ('LONG CLOSE', 'LONG CLOSE'),
        ('SHORT OPEN', 'SHORT OPEN'),
        ('SHORT CLOSE', 'SHORT CLOSE'),
        ('HOLD', 'HOLD'),
        ('EXIT', 'EXIT'),
        ('STOP LOSS', 'STOP LOSS'),
    ])
    open_price = models.DecimalField(max_digits=20, decimal_places=8)
    close_price = models.DecimalField(max_digits=20, decimal_places=8, null=True, blank=True)
    stop_loss_reached = models.BooleanField(default=False)

    def __str__(self):
        return f"{self.trade_type} at {self.date}"
