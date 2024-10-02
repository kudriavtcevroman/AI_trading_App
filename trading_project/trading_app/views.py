from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, authenticate, logout
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import AuthenticationForm
from .models import UserProfile, UserAdditionalInfo, AssetHistory
from .forms import CustomUserCreationForm, UserProfileForm, UserAdditionalInfoForm
from .binance_service import BinanceModel
from django.http import JsonResponse
import pandas as pd
from datetime import datetime

@login_required(login_url='/login/')
def home_view(request):
    return render(request, 'home.html')

# Регистрация пользователя
def register_view(request):
    if request.method == "POST":
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()  # Сохраняем пользователя

            # Сохраняем обязательные данные пользователя
            user_profile = UserProfile.objects.create(
                user=user,
                first_name=form.cleaned_data.get('first_name'),
                last_name=form.cleaned_data.get('last_name'),
                email=form.cleaned_data.get('email'),
                phone_number=form.cleaned_data.get('phone_number'),
                birth_date=form.cleaned_data.get('birth_date'),
                gender=form.cleaned_data.get('gender')
            )

            # Сохраняем необязательные данные пользователя
            UserAdditionalInfo.objects.create(
                user_profile=user_profile,
                middle_name=form.cleaned_data.get('middle_name'),
                nickname=form.cleaned_data.get('nickname'),
                country=form.cleaned_data.get('country'),
                city=form.cleaned_data.get('city'),
                telegram=form.cleaned_data.get('telegram')
            )

            messages.success(request, f"Аккаунт {user.username} был успешно создан.")
            login(request, user)
            return redirect("home")
    else:
        form = CustomUserCreationForm()
    return render(request, "register.html", {"form": form})

# Вход в систему
def login_view(request):
    if request.method == "POST":
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get("username")
            password = form.cleaned_data.get("password")
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect("home")  # Замените "home" на ваше целевое представление после входа
            else:
                messages.error(request, "Неправильное имя пользователя или пароль.")
        else:
            messages.error(request, "Неправильное имя пользователя или пароль.")
    else:
        form = AuthenticationForm()
    return render(request, "login.html", {"form": form})

# Выход из системы
def logout_view(request):
    logout(request)
    messages.info(request, "Вы вышли из системы.")
    return redirect("login")

# Личный кабинет - просмотр данных пользователя
@login_required(login_url='/login/')
def profile_view(request):
    user_profile = request.user.userprofile
    user_additional_info, created = UserAdditionalInfo.objects.get_or_create(user_profile=user_profile)

    if request.method == "POST":
        profile_form = UserProfileForm(request.POST, instance=user_profile)
        additional_info_form = UserAdditionalInfoForm(request.POST, instance=user_additional_info)

        if profile_form.is_valid() and additional_info_form.is_valid():
            profile_form.save()
            additional_info_form.save()
            messages.success(request, "Изменения успешно сохранены.")
            return redirect('profile')
        else:
            # В случае ошибки выводим уведомления
            for field, errors in profile_form.errors.items():
                for error in errors:
                    messages.error(request, f"Ошибка в поле {field}: {error}")
            for field, errors in additional_info_form.errors.items():
                for error in errors:
                    messages.error(request, f"Ошибка в поле {field}: {error}")

    else:
        profile_form = UserProfileForm(instance=user_profile)
        additional_info_form = UserAdditionalInfoForm(instance=user_additional_info)

    return render(request, 'profile.html', {
        'profile_form': profile_form,
        'additional_info_form': additional_info_form,
        'user_profile': user_profile,
        'user_additional_info': user_additional_info,
    })

# Удаление аккаунта
@login_required(login_url='/login/')
def delete_account_view(request):
    if request.method == "POST":
        request.user.delete()
        messages.success(request, "Ваш аккаунт был удален.")
        return redirect('login')
    return render(request, 'delete_account.html')

@login_required(login_url='/login/')
def edit_profile(request):
    user_profile = request.user.userprofile
    user_additional_info, created = UserAdditionalInfo.objects.get_or_create(user_profile=user_profile)

    if request.method == "POST":
        profile_form = UserProfileForm(request.POST, instance=user_profile)
        additional_info_form = UserAdditionalInfoForm(request.POST, instance=user_additional_info)

        if profile_form.is_valid() and additional_info_form.is_valid():
            profile_form.save()
            additional_info_form.save()
            return redirect('profile')
    else:
        profile_form = UserProfileForm(instance=user_profile)
        additional_info_form = UserAdditionalInfoForm(instance=user_additional_info)

    return render(request, 'edit_profile.html', {
        'profile_form': profile_form,
        'additional_info_form': additional_info_form
    })


binance = BinanceModel()  # Инициализация класса здесь, чтобы не повторять в функциях

@login_required(login_url='/login/')
def upload_training_data(request):
    user_profile = UserProfile.objects.get(user=request.user)

    # Отображаем список символов и таймфреймов
    symbols = binance.get_symbols()
    timeframes = {
        '1m': '1 минута',
        '5m': '5 минут',
        '15m': '15 минут',
        '1h': '1 час',
        '4h': '4 часа',
        '1d': '1 день',
        '1w': '1 неделя'
    }

    user_data = AssetHistory.objects.filter(user=user_profile)

    if request.method == 'POST':
        symbol = request.POST.get('pair')
        interval = request.POST.get('timeframe')
        start_date = request.POST.get('start_date')
        end_date = request.POST.get('end_date')

        # Проверяем правильность данных
        if not all([symbol, interval, start_date, end_date]):
            messages.error(request, 'Все поля обязательны для заполнения.')
            return render(request, 'upload_training_data.html', {
                'symbols': symbols,
                'timeframes': timeframes,
                'user_data': user_data
            })

        try:
            start_timestamp = pd.Timestamp(start_date)
            end_timestamp = pd.Timestamp(end_date)

            df = binance.get_historical_data(symbol, interval, start_timestamp, end_timestamp)

            if not df.empty:
                # Сохраняем данные в базу данных
                table_name = f"{symbol}_{interval}_{start_date}_{end_date}"
                binance.save_to_db(df, table_name)

                # Сохраняем данные для отображения пользователю
                for _, row in df.iterrows():
                    AssetHistory.objects.create(
                        user=user_profile,
                        asset_name=symbol,
                        timestamp=row['timestamp'],
                        open_price=row['open'],
                        high_price=row['high'],
                        low_price=row['low'],
                        close_price=row['close'],
                        volume=row['volume']
                    )
                messages.success(request, f"Данные для {symbol} успешно загружены.")
            else:
                messages.error(request, f"Не удалось загрузить данные для {symbol}.")

        except Exception as e:
            messages.error(request, f"Ошибка при загрузке данных: {str(e)}")

        return redirect('upload_training_data')

    return render(request, 'upload_training_data.html', {
        'symbols': symbols,
        'timeframes': timeframes,
        'user_data': user_data
    })

# Функция для удаления данных
@login_required(login_url='/login/')
def delete_table(request, table_id):
    data = get_object_or_404(AssetHistory, id=table_id, user=request.user.userprofile)
    data.delete()
    messages.success(request, 'Таблица успешно удалена.')
    return redirect('upload_training_data')

# Функция для изменения названия таблицы
@login_required(login_url='/login/')
def rename_table(request, table_id):
    if request.method == 'POST':
        new_name = request.POST.get('new_name')
        table = get_object_or_404(AssetHistory, id=table_id, user=request.user.userprofile)
        table.asset_name = new_name
        table.save()
        messages.success(request, 'Название таблицы успешно изменено.')
    return redirect('upload_training_data')

# Просмотр таблицы данных
@login_required(login_url='/login/')
def view_table(request, table_id):
    data = get_object_or_404(AssetHistory, id=table_id, user=request.user.userprofile)
    return render(request, 'view_table.html', {'data': data})