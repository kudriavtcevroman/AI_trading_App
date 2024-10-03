from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, authenticate, logout
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import AuthenticationForm
from .models import UserProfile, UserAdditionalInfo, AssetHistory, DataSet, TrainingSession, TrainedModel
from .forms import CustomUserCreationForm, UserProfileForm, UserAdditionalInfoForm, DataSelectionForm, TradingStrategyForm, IndicatorsForm, TrainingParametersForm, TrainingForm
from .binance_service import BinanceModel
from .tasks import train_model
import pandas as pd
from django.http import JsonResponse
from django.utils import timezone


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
    if request.method == 'POST':
        form = TrainingForm(request.user.userprofile, request.POST)
        if form.is_valid():
            # Получаем данные из формы
            dataset = form.cleaned_data['dataset']
            long = form.cleaned_data['long']
            short = form.cleaned_data['short']
            stop_loss = form.cleaned_data['stop_loss']
            indicators = form.cleaned_data['indicators']
            epochs = form.cleaned_data['epochs']
            batch_size = form.cleaned_data['batch_size']
            learning_rate = form.cleaned_data['learning_rate']

            # Создаем тренировочную сессию
            training_session = TrainingSession.objects.create(
                user=request.user.userprofile,
                dataset=dataset,
                long=long,
                short=short,
                stop_loss=stop_loss,
                indicators=indicators,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                status='pending'
            )

            # Запускаем обучение в фоне
            train_model.delay(training_session.id)

            if request.headers.get('x-requested-with') == 'XMLHttpRequest':
                # Если это AJAX-запрос, возвращаем JSON
                return JsonResponse({'success': True, 'training_session_id': training_session.id})
            else:
                # Перенаправляем на страницу обучения
                return redirect('training_progress', training_session_id=training_session.id)
        else:
            if request.headers.get('x-requested-with') == 'XMLHttpRequest':
                # Возвращаем ошибки формы в JSON
                return JsonResponse({'success': False, 'errors': form.errors})
    else:
        form = TrainingForm(request.user.userprofile)

    return render(request, 'training_model.html', {'form': form})

@login_required(login_url='/login/')
def training_data_selection(request):
    if request.method == 'POST':
        form = DataSelectionForm(request.user.userprofile, request.POST)
        if form.is_valid():
            dataset = form.cleaned_data['dataset']
            request.session['training_dataset_id'] = dataset.id
            return redirect('trading_strategy')
    else:
        form = DataSelectionForm(request.user.userprofile)

    return render(request, 'training_data_selection.html', {'form': form})

@login_required(login_url='/login/')
def trading_strategy(request):
    if request.method == 'POST':
        form = TradingStrategyForm(request.POST)
        if form.is_valid():
            long = form.cleaned_data['long']
            short = form.cleaned_data['short']
            stop_loss = form.cleaned_data['stop_loss']

            if not long and not short:
                form.add_error(None, "Вы должны выбрать хотя бы один вариант: Торговля в LONG или Торговля в SHORT.")
            else:
                request.session['trading_strategy'] = {
                    'long': long,
                    'short': short,
                    'stop_loss': stop_loss
                }
                return redirect('indicators_selection')
    else:
        form = TradingStrategyForm()

    return render(request, 'trading_strategy.html', {'form': form})

@login_required(login_url='/login/')
def indicators_selection(request):
    if request.method == 'POST':
        form = IndicatorsForm(request.POST)
        if form.is_valid():
            indicators = form.cleaned_data['indicators']
            request.session['selected_indicators'] = indicators
            return redirect('training_parameters')
    else:
        form = IndicatorsForm()

    return render(request, 'indicators_selection.html', {'form': form})

@login_required(login_url='/login/')
def training_parameters(request):
    if request.method == 'POST':
        form = TrainingParametersForm(request.POST)
        if form.is_valid():
            # Создаем TrainingSession и запускаем обучение
            user_profile = request.user.userprofile
            dataset_id = request.session.get('training_dataset_id')
            trading_strategy = request.session.get('trading_strategy')
            indicators = request.session.get('selected_indicators')

            if not dataset_id or not trading_strategy or not indicators:
                messages.error(request, "Необходимые данные для обучения отсутствуют.")
                return redirect('training_data_selection')

            dataset = get_object_or_404(DataSet, id=dataset_id, user=user_profile)

            training_session = TrainingSession.objects.create(
                user=user_profile,
                dataset=dataset,
                long=trading_strategy['long'],
                short=trading_strategy['short'],
                stop_loss=trading_strategy['stop_loss'],
                indicators=indicators,
                epochs=form.cleaned_data['epochs'],
                batch_size=form.cleaned_data['batch_size'],
                learning_rate=form.cleaned_data['learning_rate'],
                status='pending'
            )

            # Запуск фоновой задачи обучения
            # Здесь можно использовать Celery или Django Q
            # Для примера просто вызываем функцию (не рекомендуется для долгих задач)
            train_model(training_session.id)

            return redirect('training_progress', training_session_id=training_session.id)
    else:
        form = TrainingParametersForm()

    return render(request, 'training_parameters.html', {'form': form})

@login_required(login_url='/login/')
def training_progress(request, training_session_id):
    training_session = get_object_or_404(TrainingSession, id=training_session_id, user=request.user.userprofile)

    return render(request, 'training_progress.html', {'training_session': training_session})

@login_required(login_url='/login/')
def training_status(request):
    training_session_id = request.GET.get('id')
    training_session = get_object_or_404(TrainingSession, id=training_session_id, user=request.user.userprofile)

    data = {
        'status': training_session.status,
        'status_display': training_session.get_status_display(),
        'progress': training_session.progress,
        'accuracy': training_session.accuracy or 0,
        'epoch': training_session.current_epoch or 0
    }
    return JsonResponse(data)

@login_required(login_url='/login/')
def saved_models(request):
    models = TrainedModel.objects.filter(user=request.user.userprofile)
    return render(request, 'saved_models.html', {'models': models})

@login_required(login_url='/login/')
def delete_model(request, model_id):
    model = get_object_or_404(TrainedModel, id=model_id, user=request.user.userprofile)
    model.delete()
    messages.success(request, 'Модель успешно удалена.')
    return redirect('saved_models')
