import os
from celery import Celery

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'trading_project.settings')

app = Celery('trading_project')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()

app.conf.worker_pool = 'threads'