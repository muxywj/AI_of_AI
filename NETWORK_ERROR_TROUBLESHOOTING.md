# 🔧 네트워크 오류 해결 가이드

## 🚨 업로드 오류: Network Error 해결 방법

### 1. 서버 상태 확인

#### 백엔드 서버 확인
```bash
# Django 서버 실행 확인
cd /Users/seon/AIOFAI_F/AI_of_AI/chatbot_backend
python3 manage.py runserver 0.0.0.0:8000
```

#### 프론트엔드 서버 확인
```bash
# React 서버 실행 확인
cd /Users/seon/AIOFAI_F/AI_of_AI/frontend
npm start
```

### 2. 포트 충돌 확인
```bash
# 포트 사용 중인 프로세스 확인
lsof -i :8000  # Django 서버
lsof -i :3000  # React 서버
```

### 3. 방화벽 설정 확인
```bash
# macOS 방화벽 상태 확인
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --getglobalstate
```

### 4. API 엔드포인트 테스트

#### 직접 API 테스트
```bash
# 서버 상태 확인
curl http://localhost:8000/api/videos/

# 업로드 엔드포인트 테스트
curl -X POST http://localhost:8000/api/video/upload/ \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_video.mp4"
```

### 5. 브라우저 개발자 도구 확인

#### Network 탭에서 확인할 사항:
1. **요청 URL**: 올바른 엔드포인트로 요청이 가는지
2. **요청 상태**: 200, 400, 500 등 상태 코드
3. **요청 헤더**: Content-Type, Authorization 등
4. **응답 내용**: 서버에서 반환하는 에러 메시지

### 6. 일반적인 해결 방법

#### 방법 1: 서버 재시작
```bash
# 백엔드 서버 재시작
pkill -f "python3 manage.py runserver"
cd /Users/seon/AIOFAI_F/AI_of_AI/chatbot_backend
python3 manage.py runserver 0.0.0.0:8000

# 프론트엔드 서버 재시작
pkill -f "npm start"
cd /Users/seon/AIOFAI_F/AI_of_AI/frontend
npm start
```

#### 방법 2: 브라우저 캐시 클리어
- Chrome: Ctrl+Shift+R (강력 새로고침)
- 개발자 도구 → Network 탭 → "Disable cache" 체크

#### 방법 3: CORS 설정 확인
```python
# chatbot_backend/chatbot_backend/settings.py
CORS_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

CORS_ALLOW_CREDENTIALS = True
```

### 7. 파일 업로드 관련 설정 확인

#### Django 설정 확인
```python
# settings.py
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')
MEDIA_URL = '/media/'

# 파일 업로드 크기 제한
FILE_UPLOAD_MAX_MEMORY_SIZE = 50 * 1024 * 1024  # 50MB
DATA_UPLOAD_MAX_MEMORY_SIZE = 50 * 1024 * 1024  # 50MB
```

#### Nginx 설정 (프로덕션 환경)
```nginx
client_max_body_size 50M;
```

### 8. 네트워크 연결 확인

#### 로컬 네트워크 테스트
```bash
# localhost 연결 테스트
ping localhost

# 포트 연결 테스트
telnet localhost 8000
telnet localhost 3000
```

### 9. 에러 로그 확인

#### Django 로그 확인
```bash
# Django 서버 로그에서 에러 메시지 확인
tail -f /path/to/django/logs/error.log
```

#### 브라우저 콘솔 확인
- F12 → Console 탭에서 JavaScript 에러 확인
- Network 탭에서 실패한 요청 확인

### 10. 임시 해결책

#### 대용량 파일 업로드 시
```javascript
// 프론트엔드에서 파일 크기 체크
const handleFileUpload = (file) => {
  const maxSize = 50 * 1024 * 1024; // 50MB
  if (file.size > maxSize) {
    alert('파일 크기가 너무 큽니다. 50MB 이하로 업로드해주세요.');
    return;
  }
  // 업로드 진행
};
```

#### 타임아웃 설정 증가
```javascript
// api.js에서 타임아웃 증가
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000, // 60초로 증가
});
```

## 🔍 디버깅 체크리스트

- [ ] 백엔드 서버가 실행 중인가?
- [ ] 프론트엔드 서버가 실행 중인가?
- [ ] 포트 8000, 3000이 사용 가능한가?
- [ ] 브라우저에서 localhost:8000에 접근 가능한가?
- [ ] 파일 크기가 제한을 초과하지 않는가?
- [ ] 네트워크 연결이 안정적인가?
- [ ] 방화벽이 차단하지 않는가?
- [ ] CORS 설정이 올바른가?

## 📞 추가 도움이 필요한 경우

위 방법들을 시도한 후에도 문제가 지속되면:
1. 브라우저 개발자 도구의 Network 탭 스크린샷
2. Django 서버 로그의 에러 메시지
3. 업로드하려는 파일의 크기와 형식
4. 사용 중인 브라우저와 버전

이 정보들을 함께 제공해주시면 더 구체적인 해결책을 제안드릴 수 있습니다.
