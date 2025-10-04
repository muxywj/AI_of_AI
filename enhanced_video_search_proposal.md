# 🎬 LLM 기반 비디오 검색 시스템 발전 계획

## 📊 현재 시스템 분석

### ✅ 이미 구현된 강력한 기능들
- **실시간 비디오 분석**: YOLO 기반 객체 감지, 프레임별 분석
- **Multi-modal LLM 통합**: GPT, Claude, Mixtral, Ollama 지원
- **고급 검색 기능**: 크로스 비디오 검색, 인트라 비디오 추적, 시간 기반 분석
- **풍부한 메타데이터**: 날씨, 시간대, 색상, 성별, 나이 등 상세 분석
- **프레임 캡션**: 각 프레임에 대한 의미적 설명 생성

## 🚀 핵심 발전 방향

### 1. Scene-based Video Understanding (장면 기반 비디오 이해)

현재 프레임별 분석을 장면 단위로 확장하여 더 의미있는 검색이 가능하도록 개선:

```python
# 새로운 모델 추가
class VideoScene(models.Model):
    video = models.ForeignKey(Video, on_delete=models.CASCADE)
    scene_id = models.IntegerField()
    start_timestamp = models.FloatField()
    end_timestamp = models.FloatField()
    scene_description = models.TextField()  # LLM이 생성한 장면 설명
    scene_type = models.CharField(max_length=50)  # indoor/outdoor/street 등
    dominant_objects = models.JSONField()  # 주요 객체들
    activity_context = models.TextField()  # 활동 맥락
    semantic_embedding = models.JSONField()  # 벡터 임베딩
    weather_condition = models.CharField(max_length=20)
    time_of_day = models.CharField(max_length=20)
    lighting_condition = models.CharField(max_length=20)
    created_at = models.DateTimeField(auto_now_add=True)
```

### 2. Enhanced LLM Integration (향상된 LLM 통합)

```python
class VideoLLMAnalyzer:
    def __init__(self):
        self.scene_analyzer = SceneAnalyzer()
        self.semantic_search = SemanticSearchEngine()
        self.query_processor = QueryProcessor()
    
    def analyze_video_scenes(self, video_path):
        """비디오를 장면별로 분석하고 LLM으로 설명 생성"""
        scenes = self.scene_analyzer.detect_scenes(video_path)
        enhanced_scenes = []
        
        for scene in scenes:
            # 프레임 샘플링
            key_frames = self.extract_key_frames(scene)
            
            # LLM으로 장면 설명 생성
            scene_description = self.generate_scene_description(key_frames)
            
            # 의미적 임베딩 생성
            embedding = self.create_semantic_embedding(scene_description)
            
            enhanced_scenes.append({
                'scene': scene,
                'description': scene_description,
                'embedding': embedding,
                'metadata': self.extract_scene_metadata(scene)
            })
        
        return enhanced_scenes
    
    def semantic_video_search(self, query, video_id=None):
        """의미적 비디오 검색"""
        # 쿼리 분석 및 임베딩 생성
        query_embedding = self.create_query_embedding(query)
        
        # 유사도 검색
        if video_id:
            # 특정 비디오 내 검색
            results = self.search_within_video(video_id, query_embedding)
        else:
            # 전체 비디오 검색
            results = self.search_across_videos(query_embedding)
        
        return results
```

### 3. Advanced Query Processing (고급 쿼리 처리)

```python
class QueryProcessor:
    def __init__(self):
        self.llm_client = LLMClient()
    
    def parse_natural_query(self, query):
        """자연어 쿼리를 구조화된 검색 조건으로 변환"""
        prompt = f"""
        다음 비디오 검색 쿼리를 분석하여 구조화된 검색 조건을 생성해주세요:
        
        쿼리: "{query}"
        
        다음 형식으로 응답해주세요:
        {{
            "search_type": "cross_video|intra_video|time_analysis",
            "target_video_id": null,
            "conditions": {{
                "weather": ["rain", "snow", "sunny"],
                "time_of_day": ["morning", "afternoon", "evening", "night"],
                "objects": ["person", "car", "building"],
                "colors": ["red", "blue", "green"],
                "activities": ["walking", "running", "standing"],
                "scene_context": ["indoor", "outdoor", "street"]
            }},
            "temporal_constraints": {{
                "start_time": null,
                "end_time": null,
                "duration_range": null
            }},
            "semantic_intent": "사용자의 의도 설명"
        }}
        """
        
        response = self.llm_client.chat(prompt)
        return self.parse_structured_response(response)
```

## 📋 단계별 구현 계획

### Phase 1: 데이터 구조 확장 (2-3주)
1. **새로운 모델 추가**:
   - `VideoScene` 모델
   - `SceneAnalysis` 모델  
   - `SemanticEmbedding` 모델

2. **기존 분석 결과 확장**:
   - 프레임별 분석을 장면별로 그룹화
   - 의미적 임베딩 생성 및 저장

### Phase 2: LLM 통합 강화 (3-4주)
1. **장면 설명 생성**:
   - 프레임 시퀀스를 LLM에 입력하여 장면 설명 생성
   - 활동, 맥락, 감정 등 고차원 정보 추출

2. **쿼리 처리 개선**:
   - 자연어 쿼리를 구조화된 검색 조건으로 변환
   - 의도 분석 및 검색 전략 선택

### Phase 3: 검색 엔진 구현 (4-5주)
1. **의미적 검색**:
   - 벡터 임베딩 기반 유사도 검색
   - 하이브리드 검색 (키워드 + 의미적)

2. **고급 검색 기능**:
   - 시간 기반 필터링
   - 객체 추적 및 시퀀스 분석
   - 감정/분위기 기반 검색

### Phase 4: 사용자 인터페이스 개선 (2-3주)
1. **검색 인터페이스**:
   - 자연어 검색 입력
   - 검색 결과 시각화
   - 필터링 옵션

2. **결과 표시**:
   - 장면별 하이라이트
   - 타임라인 뷰
   - 관련 프레임 갤러리

## 🎯 구체적인 사용 사례

### 예시 1: "비가오는 밤에 촬영된 영상을 찾아줘"
```python
# 쿼리 분석 결과
{
    "search_type": "cross_video",
    "conditions": {
        "weather": ["rain"],
        "time_of_day": ["night"],
        "lighting_condition": ["dark"]
    },
    "semantic_intent": "비가 오는 밤의 분위기나 상황을 담은 영상 검색"
}

# 검색 결과
- 비디오 A: 매칭 점수 0.95 (비 오는 밤 거리)
- 비디오 B: 매칭 점수 0.87 (야간 비 오는 실내)
- 비디오 C: 매칭 점수 0.72 (어두운 밤, 약간의 비)
```

### 예시 2: "이 영상에서 주황색 상의를 입은 남성이 지나간 장면을 추적해줘"
```python
# 쿼리 분석 결과
{
    "search_type": "intra_video",
    "target_video_id": 123,
    "conditions": {
        "objects": ["person"],
        "colors": ["orange"],
        "gender": ["male"],
        "clothing": ["shirt", "top"]
    },
    "semantic_intent": "특정 비디오 내에서 주황색 상의를 입은 남성의 등장 장면들 추적"
}

# 검색 결과
- 장면 1 (0:15-0:25): 주황색 상의 남성 등장, 신뢰도 0.92
- 장면 3 (1:30-1:45): 같은 인물 재등장, 신뢰도 0.88
- 장면 5 (2:10-2:20): 배경에서 지나가는 모습, 신뢰도 0.75
```

### 예시 3: "이 영상에서 3:00~5:00분 사이에 지나간 사람들의 성비 분포는 어떻게 돼?"
```python
# 쿼리 분석 결과
{
    "search_type": "time_analysis",
    "target_video_id": 123,
    "temporal_constraints": {
        "start_time": 180,  # 3분 = 180초
        "end_time": 300     # 5분 = 300초
    },
    "analysis_type": "gender_distribution",
    "semantic_intent": "특정 시간 구간 내 사람들의 성별 분포 분석"
}

# 분석 결과
{
    "time_range": "3:00-5:00",
    "total_persons": 15,
    "gender_distribution": {
        "male": 8,
        "female": 7
    },
    "gender_ratio": {
        "male_percentage": 53.3,
        "female_percentage": 46.7
    },
    "detailed_breakdown": [
        {"timestamp": "3:15", "gender": "male", "confidence": 0.89},
        {"timestamp": "3:22", "gender": "female", "confidence": 0.92},
        # ... 더 많은 데이터
    ]
}
```

## 🔧 기술적 구현 세부사항

### 1. 장면 감지 알고리즘
```python
class SceneDetector:
    def detect_scenes(self, video_path):
        """비디오에서 장면 변화 감지"""
        cap = cv2.VideoCapture(video_path)
        scenes = []
        prev_frame = None
        scene_start = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if prev_frame is not None:
                # 히스토그램 기반 장면 변화 감지
                hist_diff = cv2.compareHist(
                    cv2.calcHist([prev_frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]),
                    cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]),
                    cv2.HISTCMP_CORREL
                )
                
                if hist_diff < 0.3:  # 장면 변화 임계값
                    scenes.append({
                        'start': scene_start,
                        'end': cap.get(cv2.CAP_PROP_POS_MSEC) / 1000,
                        'frame_count': cap.get(cv2.CAP_PROP_POS_FRAMES) - scene_start
                    })
                    scene_start = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            
            prev_frame = frame
        
        cap.release()
        return scenes
```

### 2. 의미적 임베딩 생성
```python
class SemanticEmbedder:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def create_scene_embedding(self, scene_description):
        """장면 설명을 벡터 임베딩으로 변환"""
        embedding = self.embedding_model.encode(scene_description)
        return embedding.tolist()
    
    def create_query_embedding(self, query):
        """검색 쿼리를 벡터 임베딩으로 변환"""
        embedding = self.embedding_model.encode(query)
        return embedding.tolist()
    
    def calculate_similarity(self, embedding1, embedding2):
        """두 임베딩 간의 코사인 유사도 계산"""
        return cosine_similarity([embedding1], [embedding2])[0][0]
```

### 3. 하이브리드 검색 엔진
```python
class HybridSearchEngine:
    def __init__(self):
        self.semantic_searcher = SemanticSearcher()
        self.keyword_searcher = KeywordSearcher()
        self.metadata_searcher = MetadataSearcher()
    
    def search(self, query, search_type='hybrid'):
        """하이브리드 검색 실행"""
        results = []
        
        if search_type in ['semantic', 'hybrid']:
            semantic_results = self.semantic_searcher.search(query)
            results.extend(semantic_results)
        
        if search_type in ['keyword', 'hybrid']:
            keyword_results = self.keyword_searcher.search(query)
            results.extend(keyword_results)
        
        if search_type in ['metadata', 'hybrid']:
            metadata_results = self.metadata_searcher.search(query)
            results.extend(metadata_results)
        
        # 결과 통합 및 중복 제거
        return self.merge_and_rank_results(results)
```

## 📈 성능 최적화 방안

### 1. 캐싱 전략
- 장면별 임베딩 캐싱
- 검색 결과 캐싱
- LLM 응답 캐싱

### 2. 인덱싱 최적화
- 벡터 데이터베이스 활용 (Pinecone, Weaviate)
- 메타데이터 인덱싱
- 시간 기반 인덱싱

### 3. 배치 처리
- 비디오 분석 배치 처리
- 임베딩 생성 배치 처리
- 검색 결과 사전 계산

## 🎯 성공 지표

### 정량적 지표
- 검색 정확도: 90% 이상
- 검색 응답 시간: 2초 이내
- 사용자 만족도: 4.5/5.0 이상

### 정성적 지표
- 자연어 쿼리 이해도
- 검색 결과 관련성
- 사용자 경험 개선도

이 발전 계획을 통해 현재의 실시간 관제 시스템을 LLM 기반의 지능적인 비디오 검색 및 분석 시스템으로 발전시킬 수 있습니다. 기존의 강력한 기반 위에 의미적 이해와 자연어 처리를 추가하여 사용자가 원하는 정보를 더욱 정확하고 빠르게 찾을 수 있게 될 것입니다.
