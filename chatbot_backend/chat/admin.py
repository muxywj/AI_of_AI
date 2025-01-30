# chat/admin.py
from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import User, SocialAccount

# 이미 등록된 모델 해제
try:
    admin.site.unregister(User)
except admin.sites.NotRegistered:
    pass

try:
    admin.site.unregister(SocialAccount)
except admin.sites.NotRegistered:
    pass

class SocialAccountInline(admin.TabularInline):
    model = SocialAccount
    extra = 0
    readonly_fields = ('provider', 'email', 'created_at', 'updated_at')

@admin.register(User)
class CustomUserAdmin(UserAdmin):
    list_display = ('username', 'email', 'is_staff', 'is_active', 'date_joined')
    list_filter = ('is_staff', 'is_superuser', 'is_active', 'groups')
    search_fields = ('username', 'email')
    ordering = ('-date_joined',)
    inlines = [SocialAccountInline]

@admin.register(SocialAccount)
class SocialAccountAdmin(admin.ModelAdmin):
    list_display = ('user', 'provider', 'email', 'created_at')
    list_filter = ('provider',)
    search_fields = ('user__email', 'email')
    readonly_fields = ('created_at', 'updated_at')