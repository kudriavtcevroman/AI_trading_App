from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, authenticate, logout
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import AuthenticationForm
from .models import UserProfile, UserAdditionalInfo, AssetHistory, DataSet
from .forms import CustomUserCreationForm, UserProfileForm, UserAdditionalInfoForm
from .binance_service import BinanceModel
import pandas as pd
from django.http import JsonResponse


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

@login_required(login_url='/login/')
def training_home(request):
    # Перенаправляем на подраздел «Загрузка данных для обучения» по умолчанию
    return redirect('upload_training_data')

binance = BinanceModel()  # Инициализация класса здесь, чтобы не повторять в функциях

@login_required(login_url='/login/')
def upload_training_data(request):
    user_profile = UserProfile.objects.get(user=request.user)

    if request.method == 'POST':
        symbol = request.POST.get('pair')
        interval = request.POST.get('timeframe')
        start_date = request.POST.get('start_date')
        end_date = request.POST.get('end_date')

        try:
            # Получение доступного диапазона дат для выбранного символа и интервала
            available_start_date, available_end_date = binance.get_available_date_range(symbol, interval)

            # Проверка выбранных дат
            if not (available_start_date <= pd.to_datetime(start_date) <= available_end_date and
                    available_start_date <= pd.to_datetime(end_date) <= available_end_date):
                messages.error(request, f"Выбранные даты выходят за пределы доступного диапазона: {available_start_date.date()} - {available_end_date.date()}")
                return redirect('upload_training_data')

            # Проверка, что начальная дата меньше конечной
            if pd.to_datetime(start_date) >= pd.to_datetime(end_date):
                messages.error(request, "Начальная дата должна быть раньше конечной даты.")
                return redirect('upload_training_data')

            # Получаем данные с Binance
            df = binance.get_historical_data(symbol, interval, start_date, end_date)

            if not df.empty:
                # Создаем запись DataSet
                dataset_name = f"{symbol}, {interval}, {start_date}, {end_date}"
                dataset = DataSet.objects.create(
                    user=user_profile,
                    name=dataset_name,
                    asset_name=symbol,
                    interval=interval,
                    start_date=start_date,
                    end_date=end_date
                )

                # Обновляем название набора данных с добавлением ID
                dataset.name = f"{dataset.id}, {dataset_name}"
                dataset.save()

                # Подготовка данных для сохранения
                data_entries = []
                for _, row in df.iterrows():
                    data_entries.append(AssetHistory(
                        dataset=dataset,
                        timestamp=row['open_time'],
                        open_price=row['open'],
                        high_price=row['high'],
                        low_price=row['low'],
                        close_price=row['close'],
                        volume=row['volume']
                    ))
                # Сохраняем данные пачкой
                AssetHistory.objects.bulk_create(data_entries)

                messages.success(request, f"Данные для {symbol} успешно загружены.")
            else:
                messages.error(request, f"Нет данных для выбранного периода.")

        except Exception as e:
            messages.error(request, f"Ошибка при загрузке данных: {str(e)}")

        return redirect('upload_training_data')

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

    user_datasets = DataSet.objects.filter(user=user_profile)

    return render(request, 'upload_training_data.html', {
        'symbols': symbols,
        'timeframes': timeframes,
        'user_datasets': user_datasets
    })


# Функция для удаления данных
@login_required(login_url='/login/')
def delete_dataset(request, dataset_id):
    dataset = get_object_or_404(DataSet, id=dataset_id, user=request.user.userprofile)
    dataset.delete()
    messages.success(request, 'Набор данных успешно удален.')
    return redirect('upload_training_data')

@login_required(login_url='/login/')
def rename_dataset(request, dataset_id):
    if request.method == 'POST':
        new_name = request.POST.get('new_name')
        dataset = get_object_or_404(DataSet, id=dataset_id, user=request.user.userprofile)
        dataset.name = new_name
        dataset.save()
        messages.success(request, 'Название набора данных успешно изменено.')
    return redirect('upload_training_data')

@login_required(login_url='/login/')
def view_dataset(request, dataset_id):
    dataset = get_object_or_404(DataSet, id=dataset_id, user=request.user.userprofile)
    data = dataset.candles.all()
    return render(request, 'view_dataset.html', {'data': data, 'dataset': dataset})

@login_required(login_url='/login/')
def get_date_range(request):
    symbol = request.GET.get('pair')
    interval = request.GET.get('timeframe')
    if symbol and interval:
        start_date, end_date = binance.get_available_date_range(symbol, interval)
        if start_date and end_date:
            return JsonResponse({
                'success': True,
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d')
            })
    return JsonResponse({'success': False})


@login_required(login_url='/login/')
def training_model(request):
    # Представление для подраздела «Обучение торговой модели»
    return render(request, 'training_model.html')