"""
celery.py

Этот файл конфигурации Celery используется для инициализации и настройки Celery в проекте Django.
Celery — это асинхронный планировщик задач, который используется для обработки фоновых задач в проектах Django.

Основные функции:
- Настройка окружения Django для Celery.
- Инициализация объекта Celery с настройками из Django.
- Автоматическое обнаружение задач в зарегистрированных приложениях.
"""

import os
from celery import Celery

# Устанавливаем настройки Django для Celery
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'trading_system.settings')

# Инициализация объекта Celery
app = Celery('trading_system')

# Загружаем конфигурации из файла настроек Django
app.config_from_object('django.conf:settings', namespace='CELERY')

# Автоматическое обнаружение задач в зарегистрированных приложениях
app.autodiscover_tasks()

# Дополнительные конфигурации для работы с потоками
app.conf.worker_pool = 'threads'
