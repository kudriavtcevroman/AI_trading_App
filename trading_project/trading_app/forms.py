from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import UserProfile, UserAdditionalInfo


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
    class Meta:
        model = UserProfile
        fields = ['first_name', 'last_name', 'email', 'phone_number', 'birth_date', 'gender']

class UserAdditionalInfoForm(forms.ModelForm):
    class Meta:
        model = UserAdditionalInfo
        fields = ['middle_name', 'nickname', 'country', 'city', 'telegram']