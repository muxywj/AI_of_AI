# chatbot_backend/urls.py

from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('chat.urls')),  # 'chatbot'으로 변경
    path('api/auth/', include('allauth.urls')),
    
]
