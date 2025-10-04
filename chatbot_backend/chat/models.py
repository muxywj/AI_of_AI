from django.db import models
from django.contrib.auth.models import AbstractUser
from django.utils.translation import gettext_lazy as _
from django.conf import settings
import uuid

class User(AbstractUser):
    email = models.EmailField(_('이메일 주소'), unique=True)
    
    class Meta:
        verbose_name = _('사용자')
        verbose_name_plural = _('사용자들')

    def __str__(self):
        return self.email

class SocialAccount(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='social_accounts'
    )
    provider = models.CharField(max_length=30)
    email = models.EmailField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ('provider', 'email')
        verbose_name = _('소셜 계정')
        verbose_name_plural = _('소셜 계정들')
        app_label = 'chat'  # 앱 레이블을 chat으로 명시

    def __str__(self):
        return f"{self.user.email} - {self.provider}"

class VideoChatSession(models.Model):
    """영상 분석 후 채팅 세션 모델"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='video_chat_sessions')
    video_id = models.IntegerField(help_text="영상 ID (backend_videochat의 Video 모델과 연결)")
    video_title = models.CharField(max_length=200, blank=True, null=True)
    video_analysis_data = models.JSONField(default=dict, blank=True, help_text="영상 분석 결과 데이터")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)
    
    class Meta:
        verbose_name = _('영상 채팅 세션')
        verbose_name_plural = _('영상 채팅 세션들')
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Video Chat Session - {self.video_title} ({self.user.email})"

class VideoChatMessage(models.Model):
    """영상 채팅 메시지 모델"""
    MESSAGE_TYPES = [
        ('user', '사용자'),
        ('ai_individual', 'AI 개별 응답'),
        ('ai_optimal', 'AI 통합 응답'),
        ('system', '시스템'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.ForeignKey(VideoChatSession, on_delete=models.CASCADE, related_name='messages')
    message_type = models.CharField(max_length=20, choices=MESSAGE_TYPES)
    content = models.TextField()
    ai_model = models.CharField(max_length=50, blank=True, null=True, help_text="AI 모델명 (gpt, claude, mixtral 등)")
    parent_message = models.ForeignKey('self', on_delete=models.CASCADE, null=True, blank=True, related_name='child_messages', help_text="통합 응답의 경우 원본 개별 응답들을 참조")
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        verbose_name = _('영상 채팅 메시지')
        verbose_name_plural = _('영상 채팅 메시지들')
        ordering = ['created_at']
    
    def __str__(self):
        return f"Video Chat Message - {self.message_type} ({self.session.video_title})"

class Video(models.Model):
    """영상 모델 - backend_videochat 방식"""
    title = models.CharField(max_length=200, blank=True, null=True)
    filename = models.CharField(max_length=255, default='')
    original_name = models.CharField(max_length=255, default='')
    file_path = models.CharField(max_length=500, default='')
    file_size = models.BigIntegerField(default=0)
    file = models.FileField(upload_to='uploads/', blank=True, null=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    analysis_status = models.CharField(max_length=50, default='pending')
    is_analyzed = models.BooleanField(default=False)
    duration = models.FloatField(default=0.0)
    enhanced_analysis = models.BooleanField(default=False)
    success_rate = models.FloatField(default=0.0)
    processing_time = models.IntegerField(default=0)
    analysis_type = models.CharField(max_length=50, default='basic')
    advanced_features_used = models.JSONField(default=dict, blank=True)
    scene_types = models.JSONField(default=list, blank=True)
    unique_objects = models.IntegerField(default=0)
    analysis_json_path = models.CharField(max_length=500, blank=True, null=True)
    analysis_progress = models.IntegerField(default=0, help_text="분석 진행률 (0-100)")
    analysis_message = models.CharField(max_length=200, blank=True, null=True, help_text="분석 진행 메시지")
    frame_images_path = models.CharField(max_length=500, blank=True, null=True, help_text="프레임 이미지 저장 경로")
    
    # 추가된 필드들 (마이그레이션 0006에서 추가됨)
    activity_patterns = models.JSONField(default=dict, blank=True, help_text='활동 패턴')
    age_distribution = models.JSONField(default=dict, blank=True, help_text='나이대 분포')
    detected_objects = models.JSONField(default=list, blank=True, help_text='감지된 객체들 (car, person, sign, etc.)')
    gender_distribution = models.JSONField(default=dict, blank=True, help_text='성별 분포')
    highlights = models.JSONField(default=list, blank=True, help_text='주요 이벤트 하이라이트')
    hourly_statistics = models.JSONField(default=dict, blank=True, help_text='시간대별 통계')
    key_events = models.JSONField(default=list, blank=True, help_text='주요 이벤트 타임스탬프')
    lighting_conditions = models.JSONField(default=dict, blank=True, help_text='조명 조건 (bright, normal, dark, artificial)')
    location_type = models.CharField(max_length=50, blank=True, null=True, help_text='위치 유형')
    object_statistics = models.JSONField(default=dict, blank=True, help_text='객체별 통계')
    person_demographics = models.JSONField(default=dict, blank=True, help_text='사람 인구통계 (gender, age, clothing)')
    scene_context = models.JSONField(default=dict, blank=True, help_text='장면 맥락 (indoor, outdoor, street, building)')
    search_keywords = models.JSONField(default=list, blank=True, help_text='검색 키워드')
    search_tags = models.JSONField(default=list, blank=True, help_text='검색 태그')
    time_of_day = models.CharField(max_length=20, blank=True, null=True, help_text='시간대 (morning, afternoon, evening, night)')
    video_summary = models.TextField(blank=True, null=True, help_text='영상 요약')
    weather_conditions = models.JSONField(default=dict, blank=True, help_text='날씨 조건 (rain, snow, sunny, cloudy)')
    
    class Meta:
        verbose_name = _('영상')
        verbose_name_plural = _('영상들')
        ordering = ['-uploaded_at']
    
    def __str__(self):
        return self.title or self.original_name or f"Video {self.id}"

class VideoAnalysisCache(models.Model):
    """영상 분석 결과 캐시 모델"""
    video_id = models.IntegerField(unique=True, help_text="영상 ID")
    analysis_data = models.JSONField(default=dict, help_text="영상 분석 결과")
    analysis_summary = models.TextField(blank=True, help_text="분석 결과 요약")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = _('영상 분석 캐시')
        verbose_name_plural = _('영상 분석 캐시들')
    
    def __str__(self):
        return f"Video Analysis Cache - Video {self.video_id}"