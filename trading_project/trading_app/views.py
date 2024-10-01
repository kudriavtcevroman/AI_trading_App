from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, authenticate, logout, update_session_auth_hash
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm, PasswordChangeForm
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from .models import UserProfile, UserAdditionalInfo
from .forms import CustomUserCreationForm, UserProfileForm, UserAdditionalInfoForm  # Форма для редактирования профиля
from django.http import JsonResponse
from .binance_service import BinanceModel
from .models import AssetHistory

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


binance = BinanceModel()

def upload_training_data(request):
    # Отображаем страницу с загрузкой данных
    if request.method == 'POST':
        symbol = request.POST.get('symbol')
        interval = request.POST.get('interval')
        start_date = request.POST.get('start_date')
        end_date = request.POST.get('end_date')

        if not all([symbol, interval, start_date, end_date]):
            return render(request, 'upload_training_data.html', {'error': 'Все поля обязательны'})

        data = binance.get_historical_data(symbol, interval, start_date, end_date)
        if data.empty:
            return render(request, 'upload_training_data.html', {'error': 'Нет данных для выбранного диапазона'})

        # Сохраняем данные в базу данных
        table_name = f"{symbol}_{interval}_{start_date}_{end_date}"
        binance.save_to_db(data, table_name)

        # Добавляем данные в модель AssetHistory для пользователя
        AssetHistory.objects.create(
            user=request.user.userprofile,
            asset_name=symbol,
            timestamp=start_date,
            open_price=data['open'][0],
            high_price=data['high'][0],
            low_price=data['low'][0],
            close_price=data['close'][0],
            volume=data['volume'][0]
        )

        return redirect('training_data_list')

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

    return render(request, 'upload_training_data.html', {'symbols': symbols, 'timeframes': timeframes})


def get_available_dates(request):
    symbol = request.GET.get('symbol')
    interval = request.GET.get('interval')

    if symbol and interval:
        start_date, end_date = binance.get_available_dates(symbol, interval)
        return JsonResponse({'start_date': start_date, 'end_date': end_date})

    return JsonResponse({'error': 'Invalid parameters'}, status=400)


# Функция для загрузки данных и отображения списка сохраненных таблиц
# Функция для загрузки данных и отображения списка сохраненных таблиц
def upload_training_data(request):
    # Проверяем наличие профиля пользователя
    user_profile = UserProfile.objects.get(user=request.user)

    # Получаем список загруженных данных для отображения
    user_data = AssetHistory.objects.filter(user=user_profile)

    # Если метод POST (пользователь отправил форму), проверяем действие
    if request.method == 'POST':
        # Если нажали на кнопку "Изменить название"
        if 'new_name' in request.POST:
            table_id = request.POST.get('table_id')
            new_name = request.POST.get('new_name')
            data = get_object_or_404(AssetHistory, id=table_id, user=user_profile)
            data.asset_name = new_name
            data.save()
            messages.success(request, 'Название таблицы успешно изменено.')
            return redirect('upload_training_data')

        # Если пользователь загружает новые данные
        if 'pair' in request.POST and 'timeframe' in request.POST:
            symbol = request.POST.get('pair')
            interval = request.POST.get('timeframe')
            start_date = request.POST.get('start_date')
            end_date = request.POST.get('end_date')

            # Здесь идет код для получения данных с Binance
            binance_model = BinanceModel()
            df = binance_model.get_historical_data(symbol, interval, start_date, end_date)

            if not df.empty:
                # Сохранение данных в базу
                table_name = f"{symbol}_{interval}_{start_date}_{end_date}"
                binance_model.save_to_db(df, table_name)
                messages.success(request, f"Данные для {symbol} успешно загружены.")

            return redirect('upload_training_data')

    return render(request, 'upload_training_data.html', {
        'user_data': user_data,
        'available_pairs': BinanceModel().get_symbols(),  # Получаем доступные торговые пары
    })

# Удаление таблицы
def delete_table(request, table_id):
    data = get_object_or_404(AssetHistory, id=table_id, user=request.user.userprofile)
    data.delete()
    messages.success(request, 'Таблица успешно удалена.')
    return redirect('upload_training_data')

# Просмотр таблицы
def view_table(request, table_id):
    data = get_object_or_404(AssetHistory, id=table_id, user=request.user.userprofile)
    return render(request, 'view_table.html', {'data': data})

# Изменение названия таблицы
def rename_table(request, table_id):
    if request.method == 'POST':
        new_name = request.POST.get('new_name')
        table = get_object_or_404(AssetHistory, id=table_id, user=request.user.userprofile)
        table.asset_name = new_name
        table.save()
        messages.success(request, 'Название таблицы успешно изменено.')
    return redirect('upload_training_data')