from django.db import models
from django.contrib.auth.models import AbstractUser
from django.utils.translation import gettext_lazy as _
from django.conf import settings

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