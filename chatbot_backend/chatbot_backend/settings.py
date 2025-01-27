import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

DEBUG = True
APPEND_SLASH = True  # (기본값: True)


ALLOWED_HOSTS = ['*']  # 필요한 도메인 추가

ROOT_URLCONF = 'chatbot_backend.urls'
STATIC_URL = '/static/'
AUTHENTICATION_BACKENDS = (
    'django.contrib.auth.backends.ModelBackend',
    'allauth.account.auth_backends.AuthenticationBackend',
)
# settings.py
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',  # SQLite 사용
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}
AUTHENTICATION_BACKENDS = (
    'allauth.account.auth_backends.AuthenticationBackend',
    'django.contrib.auth.backends.ModelBackend',
    
)
SITE_ID = 1
LOGIN_REDIRECT_URL = '/'

# Google Login 설정
SOCIALACCOUNT_PROVIDERS = {
    'google': {
        'APP': {
            'client_id': '94821981810-ftg2njljaurf7p50vpgs24bimkc7mfcg.apps.googleusercontent.com',
            'secret': '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',
            'key': ''
        }
    }
}
# 소셜 로그인 설정

GOOGLE_CLIENT_ID = "94821981810-ftg2njljaurf7p50vpgs24bimkc7mfcg.apps.googleusercontent.com"
GOOGLE_SECRET_KEY = "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"
GOOGLE_REDIRECT_URI = "http://localhost:3000"

# CORS 설정
CORS_ALLOW_CREDENTIALS = True
CORS_ALLOWED_ORIGINS = [
    "http://localhost:3000",
]



REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework.authentication.SessionAuthentication',
        'rest_framework.authentication.TokenAuthentication',
    ],
}

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': os.path.join(BASE_DIR, 'db.sqlite3'),
    }
}


# REST API 기본 설정
REST_USE_JWT = True
ACCOUNT_EMAIL_VERIFICATION = 'mandatory'
ACCOUNT_EMAIL_REQUIRED = True

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'rest_framework.authtoken',
    'corsheaders',
    'django.contrib.sites',
    

    'dj_rest_auth',
    'allauth',
    'allauth.account',
    'allauth.socialaccount',
    'allauth.socialaccount.providers.google',
    'allauth.socialaccount.providers.kakao',

    'chat.apps.ChatConfig',
]
AUTH_USER_MODEL = 'chat.User'
MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',  # 최상단에 위치
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',

    'allauth.account.middleware.AccountMiddleware', 
]
CORS_ALLOW_ALL_ORIGINS = True

# CORS 설정
CORS_ALLOWED_ORIGINS = [
    "http://localhost:3000",
]

CORS_ALLOW_CREDENTIALS = True

# 추가 CORS 설정
CORS_ALLOW_METHODS = [
    'DELETE',
    'GET',
    'OPTIONS',
    'PATCH',
    'POST',
    'PUT',
]

CORS_ALLOW_HEADERS = [
    'accept',
    'accept-encoding',
    'authorization',
    'content-type',
    'dnt',
    'origin',
    'user-agent',
    'x-csrftoken',
    'x-requested-with',
]

# Session 설정
SESSION_COOKIE_SAMESITE = 'Lax'  # 또는 'None'
SESSION_COOKIE_SECURE = False  # 개발 환경에서는 False, 프로덕션에서는 True
CSRF_COOKIE_SECURE = False    # 개발 환경에서는 False, 프로덕션에서는 True
CSRF_COOKIE_SAMESITE = 'Lax'  # 또는 'None'
CSRF_TRUSTED_ORIGINS = ['http://localhost:3000']

# CORS 설정
CORS_ALLOWED_ORIGINS = [
    "http://localhost:3000",
]
SECRET_KEY = 'fl)a4kismb2m2=vhr+g2u!yn#q#z51=!4t1ftu)^-6lvm!_%bg'



TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],  # 템플릿 디렉터리를 지정할 수 있습니다.
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]
