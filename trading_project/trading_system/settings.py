"""
settings.py

Конфигурационный файл для проекта Django "trading_system". Этот файл содержит основные настройки проекта,
включая базовые пути, параметры базы данных, настройки приложений, промежуточного ПО, и другие настройки.

Подробнее о настройках Django: https://docs.djangoproject.com/en/4.2/ref/settings/
"""

from pathlib import Path
import os

# Определяем базовую директорию проекта
BASE_DIR = Path(__file__).resolve().parent.parent

# Важные настройки безопасности
SECRET_KEY = 'django-insecure-$@)3ab&v0ul)pusk26s@k*_y5ck$^+x3^4tzsb_!5$b61+pw$4'
DEBUG = True
ALLOWED_HOSTS = []

# Приложения проекта
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'trading_app',
]

# Промежуточное ПО
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

# URL-конфигурация
ROOT_URLCONF = 'trading_system.urls'

# Настройки шаблонов
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [os.path.join(BASE_DIR, 'templates')],  # Директория с шаблонами
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

# WSGI-приложение
WSGI_APPLICATION = 'trading_system.wsgi.application'

# Настройки базы данных (PostgreSQL)
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'trading_app_db',
        'USER': 'postgres',
        'PASSWORD': 'postgres',
        'HOST': '127.0.0.1',
        'PORT': '5432',
    }
}

# Логирование
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
        },
    },
    'root': {
        'handlers': ['console'],
        'level': 'INFO',
    },
    'loggers': {
        'django': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': True,
        },
        'trading_app': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': False,
        },
    },
}

# Валидаторы паролей
AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

# Локализация и интернационализация
LANGUAGE_CODE = 'ru'
TIME_ZONE = 'UTC'
USE_I18N = True  # Включение интернационализации
USE_L10N = True  # Локализация
USE_TZ = True  # Включение поддержки временных зон

# Формат даты
DATE_INPUT_FORMATS = ['%d.%m.%Y', '%Y-%m-%d']

# Статические файлы
STATIC_URL = 'static/'

# Настройки для первичных ключей по умолчанию
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# Настройки для аутентификации и перенаправлений
LOGIN_URL = 'login'
LOGOUT_REDIRECT_URL = '/login/'  # После выхода перенаправлять на страницу авторизации

# Настройки Celery (использование Redis в качестве брокера сообщений)
CELERY_BROKER_URL = 'redis://localhost:6379/0'
CELERY_RESULT_BACKEND = 'redis://localhost:6379/0'
