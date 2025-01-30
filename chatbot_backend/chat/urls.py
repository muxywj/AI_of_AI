from django import views
from django.urls import path
from .views import ChatView, google_callback
from django.urls import path, include

from django.urls import path
from . import views




urlpatterns = [
    path('chat/<str:bot_name>/', ChatView.as_view(), name='chat'),

    

    path('api/auth/', include('dj_rest_auth.urls')),
    
    path('api/auth/registration/', include('dj_rest_auth.registration.urls')),
    


    path('accounts/google/callback/', views.google_callback, name='google_callback'),
    path('api/auth/google/callback', views.google_callback, name='google_callback'),


    path('auth/google/callback/', views.google_callback, name='google_callback'),
    path('api/auth/google/callback/', google_callback, name='google_callback'),


    # 사용자 정보 조회 엔드포인트

    # dj-rest-auth의 URL 연결
    path('auth/', include('dj_rest_auth.urls')),
    path('auth/registration/', include('dj_rest_auth.registration.urls')),
    
]