# chat/serializers.py
from rest_framework import serializers
from .models import User, SocialAccount

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ('id', 'username', 'email')
        read_only_fields = ('id', 'email')

class SocialAccountSerializer(serializers.ModelSerializer):
    class Meta:
        model = SocialAccount
        fields = ('id', 'provider', 'email', 'created_at')
        read_only_fields = ('id', 'created_at')