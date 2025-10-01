from django.urls import path
from . import clip_views

urlpatterns = [
    path('<int:video_id>/<str:timestamp>/', clip_views.ClipPreviewView.as_view(), name='clip_preview'),
]
