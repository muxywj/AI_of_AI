# chat/serializers.py
from rest_framework import serializers
from .models import User, SocialAccount, VideoChatSession, VideoChatMessage, VideoAnalysisCache

class UserSerializer(serializers.ModelSerializer):
    # 전체 이름을 하나의 필드로 제공
    full_name = serializers.SerializerMethodField()
    
    class Meta:
        model = User
        fields = ('id', 'username', 'email', 'first_name', 'last_name', 'full_name')
        read_only_fields = ('id', 'email')
    
    def get_full_name(self, obj):
        """first_name과 last_name을 합쳐서 전체 이름 반환"""
        if obj.first_name and obj.last_name:
            return f"{obj.first_name} {obj.last_name}"
        elif obj.first_name:
            return obj.first_name
        elif obj.last_name:
            return obj.last_name
        else:
            return obj.username  # 이름이 없으면 username 반환

class SocialAccountSerializer(serializers.ModelSerializer):
    class Meta:
        model = SocialAccount
        fields = ('id', 'provider', 'email', 'created_at')
        read_only_fields = ('id', 'created_at')

class VideoChatMessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = VideoChatMessage
        fields = ('id', 'message_type', 'content', 'ai_model', 'created_at')
        read_only_fields = ('id', 'created_at')

class VideoChatSessionSerializer(serializers.ModelSerializer):
    messages = VideoChatMessageSerializer(many=True, read_only=True)
    message_count = serializers.SerializerMethodField()
    
    class Meta:
        model = VideoChatSession
        fields = ('id', 'video_id', 'video_title', 'video_analysis_data', 'created_at', 'updated_at', 'is_active', 'messages', 'message_count')
        read_only_fields = ('id', 'created_at', 'updated_at')
    
    def get_message_count(self, obj):
        return obj.messages.count()

class VideoAnalysisCacheSerializer(serializers.ModelSerializer):
    class Meta:
        model = VideoAnalysisCache
        fields = ('video_id', 'analysis_data', 'analysis_summary', 'created_at', 'updated_at')
        read_only_fields = ('created_at', 'updated_at')