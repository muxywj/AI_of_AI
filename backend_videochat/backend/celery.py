import os
from celery import Celery

# Django 설정 모듈을 환경변수로 설정
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')

app = Celery('backend')

# Django 설정에서 Celery 설정을 로드
app.config_from_object('django.conf:settings', namespace='CELERY')

# 등록된 Django 앱에서 태스크를 자동으로 발견
app.autodiscover_tasks()

@app.task(bind=True)
def debug_task(self):
    print(f'Request: {self.request!r}')
