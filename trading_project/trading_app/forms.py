"""
forms.py

Модуль содержит формы для работы с пользователями, профилями, обучением моделей и тестированием данных. 
Формы используют модели приложения и обеспечивают валидацию данных для регистрации, редактирования профиля, 
выбора параметров обучения и тестирования.

Основные формы:
- Пользовательские формы для аутентификации и профиля.
- Формы для выбора данных, параметров обучения и тестирования моделей.
"""

from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import UserProfile, UserAdditionalInfo, DataSet, TrainedModel
from django.core.exceptions import ValidationError
from django.utils import timezone
import re


class UserForm(forms.ModelForm):
    """
    Форма для редактирования данных пользователя (логин, имя, фамилия, email).
    """
    username = forms.CharField(
        label='Логин',
        max_length=150,
        required=True,
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )
    first_name = forms.CharField(
        label='Имя',
        max_length=30,
        required=True,
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )
    last_name = forms.CharField(
        label='Фамилия',
        max_length=30,
        required=True,
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )
    email = forms.EmailField(
        label='Email',
        required=True,
        widget=forms.EmailInput(attrs={'class': 'form-control'})
    )

    class Meta:
        model = User
        fields = ['username', 'first_name', 'last_name', 'email']

    def clean_username(self):
        """
        Валидация: проверяет, что пользователь с таким логином уже не существует.
        """
        username = self.cleaned_data['username']
        if User.objects.exclude(pk=self.instance.pk).filter(username=username).exists():
            raise forms.ValidationError('Пользователь с таким логином уже существует.')
        return username

    def clean_email(self):
        """
        Валидация: проверяет, что пользователь с таким email уже не существует.
        """
        email = self.cleaned_data['email']
        if User.objects.exclude(pk=self.instance.pk).filter(email=email).exists():
            raise forms.ValidationError('Пользователь с таким email уже существует.')
        return email


class CustomUserCreationForm(UserCreationForm):
    """
    Форма для регистрации нового пользователя с дополнительными полями профиля.
    """
    email = forms.EmailField(required=True, label="Email", help_text="Введите корректный email.")
    first_name = forms.CharField(required=True, label="Имя")
    last_name = forms.CharField(required=True, label="Фамилия")
    phone_number = forms.CharField(
        required=False,
        label="Номер телефона",
        help_text="Введите номер телефона в формате: '+999999999'. До 15 цифр."
    )
    birth_date = forms.DateField(required=True, label="Дата рождения",
                                 widget=forms.SelectDateWidget(years=range(1900, 2024)))
    gender = forms.ChoiceField(required=True, label="Пол", choices=[('M', 'Мужчина'), ('F', 'Женщина')])

    middle_name = forms.CharField(required=False, label="Отчество")
    nickname = forms.CharField(required=False, label="Никнейм")
    country = forms.CharField(required=False, label="Страна")
    city = forms.CharField(required=False, label="Город")
    telegram = forms.CharField(required=False, label="Telegram")

    class Meta(UserCreationForm.Meta):
        model = User
        fields = ['username', 'email', 'password1', 'password2', 'first_name', 'last_name']


class UserProfileForm(forms.ModelForm):
    """
    Форма для редактирования профиля пользователя (телефон, дата рождения, пол).
    """
    birth_date = forms.DateField(
        required=True,
        widget=forms.DateInput(attrs={'type': 'date'}, format='%Y-%m-%d'),
        label='Дата рождения',
        input_formats=['%Y-%m-%d']
    )

    class Meta:
        model = UserProfile
        fields = ['phone_number', 'birth_date', 'gender']

    def clean_birth_date(self):
        """
        Валидация: проверяет, что дата рождения не находится в будущем.
        """
        birth_date = self.cleaned_data.get('birth_date')
        if birth_date > timezone.now().date():
            raise ValidationError("Дата рождения не может быть в будущем.")
        return birth_date

    def clean_phone_number(self):
        """
        Валидация: проверяет, что номер телефона соответствует формату.
        """
        phone_number = self.cleaned_data.get('phone_number')
        if phone_number:
            pattern = re.compile(r'^\+\d{9,15}$')
            if not pattern.match(phone_number):
                raise forms.ValidationError("Введите корректный номер телефона в формате: '+999999999'.")
        return phone_number


class UserAdditionalInfoForm(forms.ModelForm):
    """
    Форма для редактирования дополнительных данных пользователя (никнейм, страна, город, Telegram).
    """
    class Meta:
        model = UserAdditionalInfo
        fields = ['middle_name', 'nickname', 'telegram', 'country', 'city']


class DataSelectionForm(forms.Form):
    """
    Форма для выбора набора данных для обучения.
    """
    dataset = forms.ModelChoiceField(
        queryset=None,
        label="Выберите набор данных для обучения",
        widget=forms.RadioSelect
    )

    def __init__(self, user, *args, **kwargs):
        super(DataSelectionForm, self).__init__(*args, **kwargs)
        self.fields['dataset'].queryset = DataSet.objects.filter(user=user)


class TradingStrategyForm(forms.Form):
    """
    Форма для выбора стратегии торговли (LONG/SHORT, stop-loss).
    """
    long = forms.BooleanField(label="Торговля в LONG", required=False)
    short = forms.BooleanField(label="Торговля в SHORT", required=False)
    stop_loss = forms.FloatField(
        label="Stop-Loss",
        min_value=0,
        max_value=100,
        required=True
    )


class IndicatorsForm(forms.Form):
    """
    Форма для выбора индикаторов, используемых в стратегии.
    """
    INDICATORS_CHOICES = [
        ('EMA', 'EMA'),
        ('RSI', 'RSI'),
        ('On-Balance Volume', 'On-Balance Volume'),
        ('Stochastic Oscillator', 'Stochastic Oscillator'),
        ('Bollinger Bands', 'Bollinger Bands'),
        ('MACD', 'MACD'),
        ('Average Directional Index', 'Average Directional Index'),
        ('Standard Deviation', 'Standard Deviation'),
    ]

    indicators = forms.MultipleChoiceField(
        choices=INDICATORS_CHOICES,
        widget=forms.CheckboxSelectMultiple,
        label="Выберите индикаторы",
        required=True
    )


class TrainingParametersForm(forms.Form):
    """
    Форма для задания параметров обучения (эпохи, размер батча, скорость обучения).
    """
    epochs = forms.IntegerField(label="Количество эпох", min_value=1, max_value=1000, initial=10)
    batch_size = forms.IntegerField(label="Размер батча", min_value=1, max_value=1024, initial=32)
    learning_rate = forms.FloatField(label="Скорость обучения", min_value=0.0001, max_value=1.0, initial=0.001)


class TrainingForm(forms.Form):
    """
    Общая форма для настройки параметров обучения: выбор набора данных, стратегии, индикаторов и параметров обучения.
    """
    dataset = forms.ModelChoiceField(
        queryset=None,
        label="Выберите набор данных для обучения",
        widget=forms.RadioSelect
    )
    long = forms.BooleanField(label="Торговля в LONG", required=False)
    short = forms.BooleanField(label="Торговля в SHORT", required=False)
    stop_loss = forms.FloatField(
        label="Stop-Loss",
        min_value=0,
        max_value=100,
        required=True
    )

    INDICATORS_CHOICES = [
        ('EMA', 'EMA'),
        ('RSI', 'RSI'),
        ('On-Balance Volume', 'On-Balance Volume'),
        ('Stochastic Oscillator', 'Stochastic Oscillator'),
        ('Bollinger Bands', 'Bollinger Bands'),
        ('MACD', 'MACD'),
        ('Average Directional Index', 'Average Directional Index'),
        ('Standard Deviation', 'Standard Deviation'),
    ]

    indicators = forms.MultipleChoiceField(
        choices=INDICATORS_CHOICES,
        widget=forms.CheckboxSelectMultiple,
        label="Выберите индикаторы",
        required=True
    )

    epochs = forms.IntegerField(label="Количество эпох", min_value=1, max_value=1000, initial=10)
    batch_size = forms.IntegerField(label="Размер батча", min_value=1, max_value=1024, initial=32)
    learning_rate = forms.FloatField(label="Скорость обучения", min_value=0.0001, max_value=1.0, initial=0.001)

    def __init__(self, user, *args, **kwargs):
        super(TrainingForm, self).__init__(*args, **kwargs)
        self.fields['dataset'].queryset = DataSet.objects.filter(user=user)

    def clean(self):
        """
        Валидация: проверяет, выбрана ли хотя бы одна стратегия (LONG или SHORT).
        """
        cleaned_data = super().clean()
        long = cleaned_data.get('long')
        short = cleaned_data.get('short')

        if not long and not short:
            raise forms.ValidationError("Вы должны выбрать хотя бы один вариант: Торговля в LONG или Торговля в SHORT.")

        return cleaned_data


class ModelTestingForm(forms.Form):
    """
    Форма для тестирования обученной модели на выбранном наборе данных и таймфрейме.
    """
    model = forms.ModelChoiceField(
        queryset=None,
        label="Выберите обученную модель",
        widget=forms.Select
    )
    dataset = forms.ModelChoiceField(
        queryset=None,
        label="Выберите набор данных для тестирования",
        widget=forms.Select
    )

    TIMEFRAME_CHOICES = [
        ('1s', '1 секунда'),
        ('5s', '5 секунд'),
        ('10s', '10 секунд'),
        ('30s', '30 секунд'),
        ('1m', '1 минута'),
        ('5m', '5 минут'),
        ('10m', '10 минут'),
    ]
    timeframe = forms.ChoiceField(
        choices=TIMEFRAME_CHOICES,
        label="Выберите таймфрейм для тестирования",
        widget=forms.Select
    )

    def __init__(self, user_profile, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['model'].queryset = TrainedModel.objects.filter(user=user_profile)
        self.fields['dataset'].queryset = DataSet.objects.filter(user=user_profile)
