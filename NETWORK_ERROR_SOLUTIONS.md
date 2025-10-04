# 🔧 네트워크 오류 해결 방법

## 🚨 "업로드 오류: Network Error" 해결 가이드

### ✅ 현재 상태 확인
- ✅ Django 서버: `http://localhost:8000` 정상 실행 중
- ✅ React 서버: `http://localhost:3000` 정상 실행 중  
- ✅ 업로드 API: `POST /api/video/upload/` 정상 작동
- ✅ CORS 설정: 올바르게 구성됨

### 🔍 가능한 원인 및 해결 방법

#### 1. 브라우저 캐시 문제
```bash
# 브라우저에서 강력 새로고침
Ctrl + Shift + R (Windows/Linux)
Cmd + Shift + R (Mac)

# 또는 개발자 도구에서
F12 → Network 탭 → "Disable cache" 체크
```

#### 2. 파일 크기 제한
```javascript
// 업로드 전 파일 크기 체크 (50MB 제한)
const maxSize = 50 * 1024 * 1024; // 50MB
if (file.size > maxSize) {
  alert('파일 크기가 너무 큽니다. 50MB 이하로 업로드해주세요.');
  return;
}
```

#### 3. 네트워크 타임아웃
```javascript
// api.js에서 타임아웃 증가
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000, // 2분으로 증가
});
```

#### 4. 서버 재시작
```bash
# Django 서버 재시작
pkill -f "python3 manage.py runserver"
cd /Users/seon/AIOFAI_F/AI_of_AI/chatbot_backend
python3 manage.py runserver 0.0.0.0:8000

# React 서버 재시작  
pkill -f "npm start"
cd /Users/seon/AIOFAI_F/AI_of_AI/frontend
npm start
```

#### 5. 포트 충돌 확인
```bash
# 포트 사용 중인 프로세스 확인
lsof -i :8000
lsof -i :3000

# 프로세스 종료
kill -9 [PID]
```

### 🛠️ 디버깅 단계

#### 단계 1: 브라우저 개발자 도구 확인
1. F12 키를 눌러 개발자 도구 열기
2. Network 탭으로 이동
3. 파일 업로드 시도
4. 실패한 요청을 클릭하여 상세 정보 확인

#### 단계 2: 콘솔 로그 확인
업데이트된 코드에서 다음 정보들을 확인:
- 에러 타입
- 에러 코드  
- 에러 상태
- 에러 데이터
- 네트워크 에러 메시지

#### 단계 3: 직접 API 테스트
```bash
# 터미널에서 직접 업로드 테스트
curl -X POST http://localhost:8000/api/video/upload/ \
  -F "video=@/path/to/your/video.mp4" \
  -F "title=test"
```

### 🎯 즉시 시도할 수 있는 해결책

#### 방법 1: 브라우저 캐시 클리어
- Chrome: Ctrl+Shift+R
- Safari: Cmd+Shift+R
- Firefox: Ctrl+F5

#### 방법 2: 다른 브라우저에서 테스트
- Chrome, Safari, Firefox 등 다른 브라우저에서 시도

#### 방법 3: 파일 크기 확인
- 업로드하려는 비디오 파일이 50MB 이하인지 확인
- 더 작은 테스트 파일로 시도

#### 방법 4: 네트워크 연결 확인
```bash
# 인터넷 연결 테스트
ping google.com

# 로컬 서버 연결 테스트
ping localhost
```

### 📱 모바일/태블릿에서 테스트
- 모바일 브라우저에서도 동일한 오류가 발생하는지 확인
- 네트워크 환경이 다른 경우 문제 원인 파악 가능

### 🔄 단계별 문제 해결

1. **브라우저 새로고침** → 문제 지속 시
2. **서버 재시작** → 문제 지속 시  
3. **다른 브라우저 테스트** → 문제 지속 시
4. **파일 크기 확인** → 문제 지속 시
5. **네트워크 연결 확인** → 문제 지속 시
6. **개발자 도구에서 상세 에러 확인**

### 📞 추가 도움이 필요한 경우

위 방법들을 시도한 후에도 문제가 지속되면 다음 정보를 제공해주세요:

1. **브라우저 개발자 도구의 Network 탭 스크린샷**
2. **콘솔의 에러 메시지 전체**
3. **업로드하려는 파일의 크기와 형식**
4. **사용 중인 브라우저와 버전**
5. **운영체제 정보**

이 정보들을 바탕으로 더 구체적인 해결책을 제안드릴 수 있습니다.

### 🎉 성공 확인 방법

업로드가 성공하면:
- "영상이 성공적으로 업로드되었습니다! 분석이 시작됩니다." 메시지 표시
- 비디오 목록에 새 파일이 나타남
- 콘솔에 성공 로그 출력

이 모든 것이 정상적으로 작동하면 네트워크 오류가 해결된 것입니다!
