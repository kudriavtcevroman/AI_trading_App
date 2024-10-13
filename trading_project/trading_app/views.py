"""
views.py

Модуль, содержащий функции для обработки запросов в веб-приложении, работающем на Django.
Функции включают аутентификацию пользователей, работу с профилями, загрузку данных,
обучение моделей и тестирование.

Основные функции:
- Аутентификация и управление пользователями (регистрация, вход, выход, профиль).
- Загрузка и управление наборами данных.
- Обучение и тестирование моделей машинного обучения.
- Взаимодействие с API Binance для получения данных.

Используемые библиотеки:
- Django: для рендеринга шаблонов, аутентификации и обработки запросов.
- requests, pandas: для работы с внешним API Binance и манипуляции данными.
"""

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, authenticate, logout
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import AuthenticationForm
from .models import UserProfile, UserAdditionalInfo, AssetHistory, DataSet, TrainingSession, TrainedModel, TradeSignal
from .forms import CustomUserCreationForm, UserProfileForm, UserAdditionalInfoForm, TrainingForm, ModelTestingForm, UserForm
from .binance_service import BinanceModel
from .tasks import train_model, test_model
import pandas as pd
from django.http import JsonResponse
from django.utils.dateformat import format as django_format
from django.utils import timezone
import json
import datetime
import logging

logger = logging.getLogger(__name__)

@login_required(login_url='/login/')
def home_view(request):
    """
    Обрабатывает запрос на домашнюю страницу.
    """
    return render(request, 'home.html')

def register_view(request):
    """
    Регистрация нового пользователя. Создание учетной записи и сопутствующей информации о профиле.
    """
    if request.method == "POST":
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.first_name = form.cleaned_data.get('first_name')
            user.last_name = form.cleaned_data.get('last_name')
            user.email = form.cleaned_data.get('email')
            user.save()

            user_profile = UserProfile.objects.create(
                user=user,
                phone_number=form.cleaned_data.get('phone_number'),
                birth_date=form.cleaned_data.get('birth_date'),
                gender=form.cleaned_data.get('gender')
            )

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

def login_view(request):
    """
    Вход пользователя в систему с помощью формы аутентификации.
    """
    if request.method == "POST":
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get("username")
            password = form.cleaned_data.get("password")
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect("home")
            else:
                messages.error(request, "Неправильное имя пользователя или пароль.")
        else:
            messages.error(request, "Неправильное имя пользователя или пароль.")
    else:
        form = AuthenticationForm()
    return render(request, "login.html", {"form": form})

def logout_view(request):
    """
    Выход пользователя из системы.
    """
    logout(request)
    messages.info(request, "Вы вышли из системы.")
    return redirect("login")

@login_required(login_url='/login/')
def profile_view(request):
    """
    Просмотр и редактирование профиля пользователя.
    """
    user = request.user
    user_profile = user.userprofile
    user_additional_info, created = UserAdditionalInfo.objects.get_or_create(user_profile=user_profile)

    if request.method == "POST":
        user_form = UserForm(request.POST, instance=user)
        profile_form = UserProfileForm(request.POST, instance=user_profile)
        additional_info_form = UserAdditionalInfoForm(request.POST, instance=user_additional_info)

        if user_form.is_valid() and profile_form.is_valid() and additional_info_form.is_valid():
            user_form.save()
            profile_form.save()
            additional_info_form.save()
            messages.success(request, "Изменения успешно сохранены.")
            return redirect('profile')
        else:
            messages.error(request, "Пожалуйста, исправьте ошибки в форме.")
    else:
        user_form = UserForm(instance=user)
        profile_form = UserProfileForm(instance=user_profile)
        additional_info_form = UserAdditionalInfoForm(instance=user_additional_info)

    return render(request, 'profile/profile.html', {
        'user_form': user_form,
        'profile_form': profile_form,
        'additional_info_form': additional_info_form,
        'user_profile': user_profile,
        'user_additional_info': user_additional_info,
    })

@login_required(login_url='/login/')
def delete_account_view(request):
    """
    Удаление учетной записи пользователя.
    """
    if request.method == "POST":
        request.user.delete()
        messages.success(request, "Ваш аккаунт был удален.")
        return redirect('login')
    return render(request, 'profile/delete_account.html')


@login_required(login_url='/login/')
def training_home(request):
    """
    Перенаправление на страницу модели обучения.
    """
    return redirect('training_model')

binance = BinanceModel()

@login_required(login_url='/login/')
def upload_dataset(request):
    """
    Загрузка данных с Binance и сохранение их в базу данных.
    """
    user_profile = UserProfile.objects.get(user=request.user)

    if request.method == 'POST':
        symbol = request.POST.get('pair')
        interval = request.POST.get('timeframe')
        start_date = request.POST.get('start_date')
        end_date = request.POST.get('end_date')

        try:
            available_start_date, available_end_date = binance.get_available_date_range(symbol, interval)

            if not (available_start_date <= pd.to_datetime(start_date) <= available_end_date and
                    available_start_date <= pd.to_datetime(end_date) <= available_end_date):
                messages.error(request, f"Выбранные даты выходят за пределы доступного диапазона: {available_start_date.date()} - {available_end_date.date()}")
                return redirect('upload_data')

            if pd.to_datetime(start_date) >= pd.to_datetime(end_date):
                messages.error(request, "Начальная дата должна быть раньше конечной даты.")
                return redirect('upload_data')

            df = binance.get_historical_data(symbol, interval, start_date, end_date)

            if not df.empty:
                dataset_name = f"{symbol}, {interval}, {start_date}, {end_date}"
                dataset = DataSet.objects.create(
                    user=user_profile,
                    name=dataset_name,
                    asset_name=symbol,
                    interval=interval,
                    start_date=start_date,
                    end_date=end_date
                )

                dataset.name = f"{dataset.id}, {dataset_name}"
                dataset.save()

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

                AssetHistory.objects.bulk_create(data_entries)

                messages.success(request, f"Данные для {symbol} успешно загружены.")
            else:
                messages.error(request, f"Нет данных для выбранного периода.")
        except Exception as e:
            messages.error(request, f"Ошибка при загрузке данных: {str(e)}")

        return redirect('upload_data')

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

    return render(request, 'upload_dataset/upload_dataset.html', {
        'symbols': symbols,
        'timeframes': timeframes,
        'user_datasets': user_datasets
    })

@login_required(login_url='/login/')
def delete_dataset(request, dataset_id):
    """
    Удаление набора данных.
    """
    dataset = get_object_or_404(DataSet, id=dataset_id, user=request.user.userprofile)
    dataset.delete()
    messages.success(request, 'Набор данных успешно удален.')
    return redirect('upload_data')

@login_required(login_url='/login/')
def rename_dataset(request, dataset_id):
    """
    Переименование набора данных.
    """
    if request.method == 'POST':
        new_name = request.POST.get('new_name')
        dataset = get_object_or_404(DataSet, id=dataset_id, user=request.user.userprofile)
        dataset.name = new_name
        dataset.save()
        messages.success(request, 'Название набора данных успешно изменено.')
    return redirect('upload_data')

@login_required(login_url='/login/')
def view_dataset(request, dataset_id):
    """
    Просмотр данных набора.
    """
    dataset = get_object_or_404(DataSet, id=dataset_id, user=request.user.userprofile)
    data = dataset.candles.all()
    return render(request, 'upload_dataset/view_dataset.html', {'data': data, 'dataset': dataset})

@login_required(login_url='/login/')
def get_date_range(request):
    """
    Возвращает диапазон доступных дат для выбранного символа и интервала.
    """
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
    """
    Обучение модели на выбранном наборе данных.
    """
    if request.method == 'POST':
        form = TrainingForm(request.user.userprofile, request.POST)
        if form.is_valid():
            dataset = form.cleaned_data['dataset']
            long = form.cleaned_data['long']
            short = form.cleaned_data['short']
            stop_loss = form.cleaned_data['stop_loss']
            indicators = form.cleaned_data['indicators']
            epochs = form.cleaned_data['epochs']
            batch_size = form.cleaned_data['batch_size']
            learning_rate = form.cleaned_data['learning_rate']

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

            train_model.delay(training_session.id)

            if request.headers.get('x-requested-with') == 'XMLHttpRequest':
                return JsonResponse({'success': True, 'training_session_id': training_session.id})
            else:
                return redirect('training_progress', training_session_id=training_session.id)
        else:
            if request.headers.get('x-requested-with') == 'XMLHttpRequest':
                return JsonResponse({'success': False, 'errors': form.errors})
    else:
        form = TrainingForm(request.user.userprofile)

    return render(request, 'training/training_model.html', {'form': form})

@login_required(login_url='/login/')
def training_progress(request, training_session_id):
    """
    Страница прогресса обучения модели.
    """
    training_session = get_object_or_404(TrainingSession, id=training_session_id, user=request.user.userprofile)
    return render(request, 'training/training_progress.html', {'training_session': training_session})

@login_required(login_url='/login/')
def training_status(request):
    """
    Возвращает статус текущей сессии обучения модели.
    """
    training_session_id = request.GET.get('id')
    training_session = get_object_or_404(TrainingSession, id=training_session_id, user=request.user.userprofile)

    data = {
        'status': training_session.status,
        'status_display': training_session.get_status_display(),
        'progress': training_session.progress,
        'mse': training_session.mse or 0,
        'mae': training_session.mae or 0,
        'rmse': training_session.rmse or 0,
        'history': training_session.history
    }
    return JsonResponse(data)

@login_required(login_url='/login/')
def saved_models(request):
    """
    Страница сохраненных моделей.
    """
    models = TrainedModel.objects.filter(user=request.user.userprofile)
    return render(request, 'training/saved_models.html', {'models': models})

@login_required(login_url='/login/')
def delete_model(request, model_id):
    """
    Удаление обученной модели.
    """
    model = get_object_or_404(TrainedModel, id=model_id, user=request.user.userprofile)
    model.delete()
    messages.success(request, 'Модель успешно удалена.')
    return redirect('saved_models')

@login_required(login_url='/login/')
def testing_model_view(request):
    """
    Тестирование обученной модели.
    """
    user_profile = request.user.userprofile
    if request.method == 'POST':
        form = ModelTestingForm(user_profile, request.POST)
        if form.is_valid():
            model = form.cleaned_data['model']
            dataset = form.cleaned_data['dataset']
            timeframe = form.cleaned_data['timeframe']

            request.session['testing_model_id'] = model.id
            request.session['testing_dataset_id'] = dataset.id
            request.session['testing_timeframe'] = timeframe

            return redirect('testing_chart')
    else:
        form = ModelTestingForm(user_profile)

    return render(request, 'testing_model/testing_model.html', {'form': form})

@login_required(login_url='/login/')
def start_model_testing(request):
    """
    Запуск тестирования модели.
    """
    if request.method == 'POST':
        model_id = request.session.get('testing_model_id')
        dataset_id = request.session.get('testing_dataset_id')
        if model_id and dataset_id:
            test_model.delay(model_id, dataset_id)
            return JsonResponse({'success': True, 'message': 'Тестирование началось'})
        else:
            return JsonResponse({'success': False, 'message': 'Не выбрана модель или набор данных'}, status=400)
    else:
        return JsonResponse({'success': False, 'message': 'Некорректный запрос'}, status=400)

@login_required(login_url='/login/')
def get_latest_bars(request):
    """
    Получение последних баров (свечей) для отображения на графике.
    """
    dataset_id = request.session.get('testing_dataset_id')
    if not dataset_id:
        return JsonResponse({'error': 'No dataset selected for testing'}, status=400)

    dataset = DataSet.objects.get(id=dataset_id, user=request.user.userprofile)
    bars = AssetHistory.objects.filter(dataset=dataset).order_by('-timestamp')[:5]

    latest_data = {
        'bars': [
            {
                'timestamp': django_format(bar.timestamp, 'U') * 1000,
                'open_price': float(bar.open_price),
                'high_price': float(bar.high_price),
                'low_price': float(bar.low_price),
                'close_price': float(bar.close_price)
            } for bar in bars
        ]
    }

    return JsonResponse(latest_data)

@login_required(login_url='/login/')
def get_initial_chart_data(request):
    """
    Получение данных для начального отображения графика.
    """
    dataset_id = request.session.get('testing_dataset_id')
    if not dataset_id:
        return JsonResponse({'error': 'No dataset selected for testing'}, status=400)

    dataset = DataSet.objects.get(id=dataset_id, user=request.user.userprofile)
    bars = AssetHistory.objects.filter(dataset=dataset).order_by('timestamp')

    chart_data = {
        'symbol': dataset.asset_name,
        'bars': [
            {
                'timestamp': django_format(bar.timestamp, 'U') * 1000,
                'open_price': float(bar.open_price),
                'high_price': float(bar.high_price),
                'low_price': float(bar.low_price),
                'close_price': float(bar.close_price)
            } for bar in bars
        ]
    }

    return JsonResponse(chart_data)

@login_required(login_url='/login/')
def test_results(request, model_id):
    """
    Страница результатов тестирования модели с графиком и сигналами.
    """
    trained_model = get_object_or_404(TrainedModel, id=model_id, user=request.user.userprofile)
    signals = trained_model.signals.all().order_by('date')

    dataset = trained_model.training_session.dataset
    candles = AssetHistory.objects.filter(dataset=dataset).order_by('timestamp')

    chart_data = [
        {
            'time': int(candle.timestamp.timestamp()) * 1000,
            'open': float(candle.open_price),
            'high': float(candle.high_price),
            'low': float(candle.low_price),
            'close': float(candle.close_price),
        }
        for candle in candles
    ]

    signals_data = [
        {
            'time': int(signal.date.timestamp()) * 1000,
            'price': float(signal.open_price),
            'trade_type': signal.trade_type,
            'color': get_signal_color(signal.trade_type),
        }
        for signal in signals
    ]

    return render(request, 'testing_model/test_results.html', {
        'trained_model': trained_model,
        'chart_data': json.dumps(chart_data),
        'signals_data': json.dumps(signals_data),
        'signals': signals,
    })

def get_signal_color(trade_type):
    """
    Определяет цвет сигнала в зависимости от его типа.
    """
    if 'LONG' in trade_type:
        return 'green'
    elif 'SHORT' in trade_type:
        return 'red'
    elif 'HOLD' in trade_type:
        return 'blue'
    elif 'EXIT' in trade_type:
        return 'orange'
    elif 'STOP LOSS' in trade_type:
        return 'yellow'
    else:
        return 'grey'

@login_required(login_url='/login/')
def testing_chart(request):
    """
    Отображение графика для тестирования модели.
    """
    model_id = request.session.get('testing_model_id')
    dataset_id = request.session.get('testing_dataset_id')
    timeframe = request.session.get('testing_timeframe', '1m')

    if not model_id or not dataset_id:
        messages.error(request, "Выберите модель и набор данных для тестирования.")
        return redirect('testing_model')

    trained_model = get_object_or_404(TrainedModel, id=model_id, user=request.user.userprofile)
    dataset = get_object_or_404(DataSet, id=dataset_id, user=request.user.userprofile)

    candles = AssetHistory.objects.filter(dataset=dataset).order_by('timestamp')
    chart_data = [
        {
            'time': int(candle.timestamp.timestamp()) * 1000,
            'open': float(candle.open_price),
            'high': float(candle.high_price),
            'low': float(candle.low_price),
            'close': float(candle.close_price),
        }
        for candle in candles
    ]

    timeframe_mapping = {
        '1s': 1000,
        '5s': 5000,
        '10s': 10000,
        '30s': 30000,
        '1m': 60000,
        '5m': 300000,
        '10m': 600000,
    }
    interval_ms = timeframe_mapping.get(timeframe, 60000)

    return render(request, 'testing_model/testing_chart.html', {
        'trained_model': trained_model,
        'dataset': dataset,
        'chart_data': json.dumps(chart_data),
        'timeframe': timeframe,
        'interval_ms': interval_ms,
    })

@login_required(login_url='/login/')
def get_signal(request):
    """
    Получение торговых сигналов для модели на основе временных меток.
    """
    model_id = request.GET.get('model_id')
    time = request.GET.get('time')

    try:
        trained_model = get_object_or_404(TrainedModel, id=model_id, user=request.user.userprofile)
        timestamp = datetime.datetime.fromtimestamp(int(time) / 1000, tz=timezone.utc)

        signals = TradeSignal.objects.filter(
            model=trained_model,
            date=timestamp
        ).order_by('date')

        if signals.exists():
            data = {
                'signals': [
                    {
                        'time': int(signal.date.timestamp()) * 1000,
                        'price': float(signal.open_price),
                        'trade_type': signal.trade_type,
                        'color': get_signal_color(signal.trade_type),
                    } for signal in signals
                ]
            }
        else:
            data = {'signals': []}

        return JsonResponse(data)

    except Exception as e:
        logger.error(f"Error in get_signal view: {e}", exc_info=True)
        return JsonResponse({'signals': []})
