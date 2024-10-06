from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import UserProfile, UserAdditionalInfo
from django.core.exceptions import ValidationError
from django.utils import timezone
from .models import DataSet, TrainedModel

class CustomUserCreationForm(UserCreationForm):
    # Поля из модели User
    email = forms.EmailField(required=True, label="Email", help_text="Введите корректный email.")

    # Поля из модели UserProfile (обязательные данные)
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

    # Поля из модели UserAdditionalInfo (необязательные данные)
    middle_name = forms.CharField(required=False, label="Отчество")
    nickname = forms.CharField(required=False, label="Никнейм")
    country = forms.CharField(required=False, label="Страна")
    city = forms.CharField(required=False, label="Город")
    telegram = forms.CharField(required=False, label="Telegram")

    class Meta(UserCreationForm.Meta):
        model = User
        fields = ['username', 'email', 'password1', 'password2', 'first_name', 'last_name', 'phone_number',
                  'birth_date', 'gender',
                  'middle_name', 'nickname', 'country', 'city', 'telegram']

class UserProfileForm(forms.ModelForm):
    birth_date = forms.DateField(
        required=True,
        widget=forms.DateInput(format='%d.%m.%Y', attrs={'type': 'date'}),
        label='Дата рождения'
    )

    class Meta:
        model = UserProfile
        fields = ['first_name', 'last_name', 'email', 'phone_number', 'birth_date', 'gender']

    # Валидация поля birth_date, чтобы оно не было в будущем
    def clean_birth_date(self):
        birth_date = self.cleaned_data.get('birth_date')
        if birth_date > timezone.now().date():
            raise ValidationError("Дата рождения не может быть в будущем.")
        return birth_date

class UserAdditionalInfoForm(forms.ModelForm):
    class Meta:
        model = UserAdditionalInfo
        fields = ['middle_name', 'nickname', 'telegram', 'country', 'city']


class DataSelectionForm(forms.Form):
    dataset = forms.ModelChoiceField(
        queryset=None,
        label="Выберите набор данных для обучения",
        widget=forms.RadioSelect
    )

    def __init__(self, user, *args, **kwargs):
        super(DataSelectionForm, self).__init__(*args, **kwargs)
        self.fields['dataset'].queryset = DataSet.objects.filter(user=user)

class TradingStrategyForm(forms.Form):
    long = forms.BooleanField(label="Торговля в LONG", required=False)
    short = forms.BooleanField(label="Торговля в SHORT", required=False)
    stop_loss = forms.FloatField(
        label="Stop-Loss",
        min_value=0,
        max_value=100,
        required=True
    )

class IndicatorsForm(forms.Form):
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
    epochs = forms.IntegerField(label="Количество эпох", min_value=1, max_value=1000, initial=10)
    batch_size = forms.IntegerField(label="Размер батча", min_value=1, max_value=1024, initial=32)
    learning_rate = forms.FloatField(label="Скорость обучения", min_value=0.0001, max_value=1.0, initial=0.001)


class TrainingForm(forms.Form):
    # Выбор набора данных
    dataset = forms.ModelChoiceField(
        queryset=None,
        label="Выберите набор данных для обучения",
        widget=forms.RadioSelect
    )

    # Стратегия торговли
    long = forms.BooleanField(label="Торговля в LONG", required=False)
    short = forms.BooleanField(label="Торговля в SHORT", required=False)
    stop_loss = forms.FloatField(
        label="Stop-Loss",
        min_value=0,
        max_value=100,
        required=True
    )

    # Выбор индикаторов
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

    # Параметры обучения
    epochs = forms.IntegerField(label="Количество эпох", min_value=1, max_value=1000, initial=10)
    batch_size = forms.IntegerField(label="Размер батча", min_value=1, max_value=1024, initial=32)
    learning_rate = forms.FloatField(label="Скорость обучения", min_value=0.0001, max_value=1.0, initial=0.001)

    def __init__(self, user, *args, **kwargs):
        super(TrainingForm, self).__init__(*args, **kwargs)
        self.fields['dataset'].queryset = DataSet.objects.filter(user=user)

    def clean(self):
        cleaned_data = super().clean()
        long = cleaned_data.get('long')
        short = cleaned_data.get('short')

        if not long and not short:
            raise forms.ValidationError("Вы должны выбрать хотя бы один вариант: Торговля в LONG или Торговля в SHORT.")

        return cleaned_data

class ModelSelectionForm(forms.Form):
    model = forms.ModelChoiceField(
        queryset=None,
        label="Выберите обученную модель для тестирования",
        widget=forms.RadioSelect
    )

    def __init__(self, user, *args, **kwargs):
        super(ModelSelectionForm, self).__init__(*args, **kwargs)
        self.fields['model'].queryset = TrainedModel.objects.filter(user=user)