from django.urls import path, include
from .views import ChatView

urlpatterns = [
    path('chat/<str:bot_name>/', ChatView.as_view(), name='chat'),
    path('api/auth/', include('dj_rest_auth.urls')),
    path('api/auth/registration/', include('dj_rest_auth.registration.urls')),
]