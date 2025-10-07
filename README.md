# 🧠 AI OF AI: 인공지능 답변 통합 플랫폼

> **AI OF AI**는 여러 인공지능 모델을 하나의 플랫폼에서 동시에 활용할 수 있도록 개발된 **AI 통합 응답 시스템**입니다.  
> 사용자 질문에 대해 ChatGPT, Gemini, Claude 등 다양한 AI의 답변을 한눈에 비교할 수 있으며,  
> LangChain 기반 RAG 및 YOLOv8, BLIP2 등의 AI 기능을 통합하여 실질적이고 효율적인 결과를 제공합니다.

---

## 🎨 프로젝트 대표 이미지
<img width="2556" height="1666" alt="image" src="https://github.com/user-attachments/assets/805b6e68-cb78-44bc-a52d-7c7e641684fa" />


---

## 📘 개요

AI OF AI는 **다중 AI 통합**, **시각화된 비교 인터페이스**, **유연한 설정 시스템**을 통해  
사용자에게 가장 폭넓고 정확한 인공지능 경험을 제공합니다.  
Python(Django) 백엔드와 React.js 프론트엔드로 구성되어 있으며,  
Docker 기반 컨테이너 환경에서 안정적으로 실행됩니다.

---

## 🚀 주요 기능

| 기능 | 설명 |
|------|------|
| **다중 AI 응답 비교** | ChatGPT, Gemini, Claude 등 다양한 AI 모델의 답변을 동시에 표시 |
| **LangChain RAG 기반 검색** | 지식 기반 검색을 통한 컨텍스트 강화형 답변 |
| **AI 영상 분석 기능** | YOLOv8을 활용한 객체 인식, BLIP/BLIP2를 통한 이미지 캡셔닝 |
| **자동 태스크 처리** | Celery 기반 비동기 작업 큐를 활용한 AI 프로세스 분리 |
| **사용자 설정 지원** | JSON 설정 파일로 모델 수, 표시 개수, 프롬프트 형태 등을 커스터마이징 |
| **직관적인 UI/UX** | Tailwind CSS 기반 반응형 레이아웃, 다중 보기 및 전체 화면 모드 지원 |

---

## ⚙️ 시스템 아키텍처
<img width="512" height="512" alt="image" src="https://github.com/user-attachments/assets/9e68e9b8-2328-49d0-adab-9967f62a1208" />



**구성 요약**
- **Frontend (React.js + Tailwind CSS):** 사용자 인터페이스 및 멀티 AI 응답 비교 뷰
- **Backend (Django REST API):** 요청 처리, DB 관리, AI 모델 연결
- **Core AI (LangChain RAG, PyTorch):** AI 응답 생성 및 멀티모델 관리
- **Containerization (Docker):** 배포 환경 일원화
- **Async Task (Celery):** 비동기 AI 처리 및 백그라운드 연산 관리

---

## 🧩 저장소 구조

```

AI-OF-AI/
├── frontend/             # React.js 기반 프론트엔드
├── backend/              # Django REST Framework API 서버
├── core_ai/              # LangChain RAG, YOLOv8, BLIP2 등 핵심 AI 모듈
├── docs/                 # 시스템 정의서, 명세서, 요구사항 문서 등
│   └── images/           # (프로젝트 이미지 폴더)
├── docker-compose.yml    # 컨테이너 실행 환경 설정
└── README.md

````

---

## 🧠 지원 중인 AI 모델

| AI 이름 | 주요 특징 |
|----------|-----------|
| ChatGPT | OpenAI의 대표 언어 모델 |
| Gemini / Gemini Pro | 구글의 멀티모달 대화형 모델 |
| Claude | 자연스러운 대화 중심 AI 어시스턴트 |
| Copilot | 코드 작성 및 정보 탐색 지원 도우미 |
| Mistral AI | 경량 고성능 LLM |
| HuggingChat | Hugging Face 오픈소스 챗봇 |
| YouAI | 개인 맞춤형 응답 최적화 |
| Devin AI | 개발 보조 AI |
| Tongyi Qianwen | 데이터 통합형 AI |
| DouBao AI | 금융 분석 중심 AI |
| Character AI | 스토리텔링 기반 캐릭터 생성 AI |
| Suno AI | 음악 생성 AI |
| ...외 10여 종 이상 지원 |

---

## 💡 주요 화면 예시
<img width="3024" height="1646" alt="image" src="https://github.com/user-attachments/assets/2de65649-15a2-49cd-8601-01d6bda1f1a9" />
<img width="1986" height="1694" alt="image" src="https://github.com/user-attachments/assets/6a22718a-15ad-4931-b62f-dcb4ff1ef69c" />
<img width="320" height="494" alt="image" src="https://github.com/user-attachments/assets/f774185b-165b-4673-9294-7fc1e0835fb7" />



---

## 🧠 AI 응답 처리 흐름

1️⃣ 사용자가 질문 입력  
2️⃣ Django 서버 → Core AI로 전달  
3️⃣ LangChain RAG가 질의 기반 컨텍스트 검색  
4️⃣ 다중 AI 모델에 병렬 질의  
5️⃣ 응답 수집 및 비교 결과를 UI에 표시  

---

## 🧰 개발 환경 및 프레임워크

| 구분 | 기술 스택 |
|------|-------------|
| **Frontend** | React.js, Tailwind CSS |
| **Backend** | Django REST Framework, Celery, Redis |
| **AI Core** | LangChain, PyTorch, YOLOv8, BLIP/BLIP2 |
| **Deployment** | Docker, docker-compose |
| **Database** | SQLite3 / PostgreSQL |


---

## 🧪 실행 방법

### 🖥️ 로컬 실행

#### Backend (Django)
```bash
cd backend
python manage.py runserver
````

#### Frontend (React)

```bash
cd frontend
npm install
npm run start
```

### 🐳 Docker 환경 실행

```bash
docker-compose up -d
```

> 컨테이너 구성: frontend + backend + redis + worker (Celery)

---

## 🧩 설정 및 확장

* **AI 모델 추가:**
  `/core_ai/config/ai_models.json` 파일을 수정해 신규 AI 모델 등록 가능
* **프롬프트 템플릿:**
  `/core_ai/prompts/` 내 JSON 파일을 수정하여 맞춤형 RAG 프롬프트 설정 가능
* **환경 변수 설정:**
  `.env` 파일 내 API Key 및 모델 환경 정보 입력

---

## 🧠 기술 스택 시각화

<img width="657" height="448" alt="image" src="https://github.com/user-attachments/assets/d2813772-1dab-44ed-96e3-89db875d1bd6" />


---

## 🧪 소프트웨어 시험 결과 요약

| 항목                      | 결과                                |
| ----------------------- | --------------------------------- |
| **빌드 테스트**              | Windows 11, macOS Ventura 환경에서 성공 |
| **다중 AI 응답 기능**         | 정상 작동                             |
| **비동기 태스크 처리 (Celery)** | 대기열 기반 정상 처리                      |
| **Docker 실행**           | 모든 컨테이너 정상 빌드 및 통신 확인             |
| **성능 평가**               | 평균 응답 시간 2.1초, 안정성 99.2%          |

---

## 👥 팀 정보

| 이름      | 역할                             |
| ------- | ------------------------------ |
| **심민아** | 프론트엔드 UI/UX, React 구성          |
| **구경선** | 백엔드 API, Django REST Framework |
| **양원진** | AI Core 설계 및 시스템 통합            |
| **팀명**  | AI OF AI : 인공지능 통합 플랫폼         |

<img width="1232" height="638" alt="image" src="https://github.com/user-attachments/assets/35cb4129-739c-40f5-bc3b-9e72205e47f6" />

---

## 💬 참고 및 기여

* 본 프로젝트는 **대학교 졸업작품 연구용**으로 개발되었습니다.
* 오픈소스 기여나 피드백은 환영하지만, **상업적 이용은 제한**됩니다.
* 개선 제안 및 버그 리포트는 Issue 탭을 통해 제출해주세요.

---

## 📜 라이선스

> © 2025 AI OF AI Project Team
> 본 프로젝트는 **교육 및 연구 목적**으로만 사용 가능합니다.
> License: *Unlicensed (Private Repository)*

---

## 🎥 시연 영상 (추가 예정)

> 추후 공개 예정입니다.

