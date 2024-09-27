from django.db import models
from django.core.validators import EmailValidator, RegexValidator
from django.contrib.auth.models import User  # Используем встроенную модель User для хранения имени пользователя и пароля

# Модель для основных данных пользователя
class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)  # Связь с встроенной моделью User
    first_name = models.CharField(max_length=100)  # Имя
    last_name = models.CharField(max_length=100)  # Фамилия
    email = models.EmailField(
        unique=True,
        validators=[EmailValidator(message="Введите корректный email.")]
    )  # Почта с валидацией
    phone_regex = RegexValidator(
        regex=r'^\+?1?\d{7,15}$',
        message="Номер телефона должен быть введен в формате: '+999999999'. Допустимо от 7 до 15 цифр."
    )
    phone_number = models.CharField(validators=[phone_regex], max_length=17, blank=True)  # Номер телефона с валидацией
    birth_date = models.DateField()  # Дата рождения
    gender = models.CharField(max_length=10, choices=[('M', 'Мужчина'), ('F', 'Женщина')])  # Пол

    def __str__(self):
        return f"{self.first_name} {self.last_name} ({self.user.username})"

# Модель для необязательных данных пользователя
class UserAdditionalInfo(models.Model):
    user_profile = models.OneToOneField(UserProfile, on_delete=models.CASCADE)  # Связь с основной таблицей профиля
    middle_name = models.CharField(max_length=100, blank=True, null=True)  # Отчество
    nickname = models.CharField(max_length=100, blank=True, null=True)  # Никнейм
    telegram = models.CharField(max_length=100, blank=True, null=True)  # Telegram
    country = models.CharField(max_length=100, blank=True, null=True)  # Страна
    city = models.CharField(max_length=100, blank=True, null=True)  # Город

    def __str__(self):
        return f"Доп. информация: {self.nickname or self.user_profile.first_name}"

# Модель для хранения истории активов
class AssetHistory(models.Model):
    user = models.ForeignKey(UserProfile, on_delete=models.CASCADE)  # Связь с профилем пользователя
    asset_name = models.CharField(max_length=100)  # Название актива (например, BTC/USDT)
    timestamp = models.DateTimeField()  # Время (timestamp)
    open_price = models.DecimalField(max_digits=20, decimal_places=8)  # Цена открытия
    high_price = models.DecimalField(max_digits=20, decimal_places=8)  # Высшая цена
    low_price = models.DecimalField(max_digits=20, decimal_places=8)  # Низшая цена
    close_price = models.DecimalField(max_digits=20, decimal_places=8)  # Цена закрытия
    volume = models.DecimalField(max_digits=20, decimal_places=8)  # Объем

    def __str__(self):
        return f"{self.asset_name} ({self.timestamp})"

