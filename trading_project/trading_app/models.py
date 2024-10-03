from django.db import models
from django.core.validators import EmailValidator, RegexValidator
from django.contrib.auth.models import User


# Модель для основных данных пользователя
class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)

    # Основные поля
    first_name = models.CharField(max_length=100)
    last_name = models.CharField(max_length=100)

    # Email с валидатором
    email = models.EmailField(
        unique=True,
        validators=[EmailValidator(message="Введите корректный email.")]
    )

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
        return f'{self.first_name} {self.last_name} ({self.user.username})'


# Модель для необязательных данных пользователя
class UserAdditionalInfo(models.Model):
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
    dataset = models.ForeignKey(DataSet, on_delete=models.CASCADE, related_name='candles')
    timestamp = models.DateTimeField()
    open_price = models.DecimalField(max_digits=20, decimal_places=8)
    high_price = models.DecimalField(max_digits=20, decimal_places=8)
    low_price = models.DecimalField(max_digits=20, decimal_places=8)
    close_price = models.DecimalField(max_digits=20, decimal_places=8)
    volume = models.DecimalField(max_digits=20, decimal_places=8)

    def __str__(self):
        return f"{self.dataset.asset_name} ({self.timestamp})"