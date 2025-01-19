from django.urls import path
from .views import ChatView

urlpatterns = [
    path('chat/<str:bot_name>/', ChatView.as_view(), name='chat'),
]