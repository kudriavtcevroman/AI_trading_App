"""
urls.py

Этот модуль содержит URL-шаблоны для приложения. Он связывает URL-запросы с соответствующими представлениями (views), 
используя функции из views.py и стандартные представления Django для аутентификации.

Основные функции:
- Определение URL-путей для различных страниц приложения (главная страница, аутентификация, профили пользователей).
- Управление загрузкой данных, обучением и тестированием моделей машинного обучения.
- Использование встроенных представлений Django для управления паролем.
"""

from django.urls import path
from . import views
from django.contrib.auth import views as auth_views

urlpatterns = [
    # Главная страница
    path('', views.home_view, name='home'),

    # Аутентификация
    path("register/", views.register_view, name="register"),
    path("login/", views.login_view, name="login"),
    path("logout/", views.logout_view, name="logout"),

    # Профиль пользователя
    path('profile/', views.profile_view, name='profile'),
    path('profile/delete/', views.delete_account_view, name='delete_account'),
    path('profile/password_change/',
         auth_views.PasswordChangeView.as_view(template_name='profile/password_change.html'), name='password_change'),
    path('profile/password_change/password_change_done/',
         auth_views.PasswordChangeDoneView.as_view(template_name='profile/password_change_done.html'),
         name='password_change_done'),

    # Управление данными
    path('upload_data/', views.upload_dataset, name='upload_data'),
    path('delete_dataset/<int:dataset_id>/', views.delete_dataset, name='delete_dataset'),
    path('rename_dataset/<int:dataset_id>/', views.rename_dataset, name='rename_dataset'),
    path('view_dataset/<int:dataset_id>/', views.view_dataset, name='view_dataset'),
    path('get_date_range/', views.get_date_range, name='get_date_range'),

    # Обучение моделей
    path('training/', views.training_home, name='training_home'),
    path('training/training_model/', views.training_model, name='training_model'),
    path('training/progress/<int:training_session_id>/', views.training_progress, name='training_progress'),
    path('training/status/', views.training_status, name='training_status'),
    path('training/saved_models/', views.saved_models, name='saved_models'),
    path('training/delete_model/<int:model_id>/', views.delete_model, name='delete_model'),

    # Тестирование моделей
    path('testing_model/', views.testing_model_view, name='testing_model'),
    path('testing_model/get_signal/', views.get_signal, name='get_signal'),
    path('test_results/<int:model_id>/', views.test_results, name='test_results'),
    path('testing_model/chart/', views.testing_chart, name='testing_chart'),
    path('testing_model/start_test/', views.start_model_testing, name='start_model_testing'),

    # API для получения данных для графиков
    path('api/get_initial_chart_data/', views.get_initial_chart_data, name='get_initial_chart_data'),
    path('api/get_latest_bars/', views.get_latest_bars, name='get_latest_bars'),
]
