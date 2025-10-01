# chat/db_builder.py - 고도화된 비디오 RAG 시스템
from typing import Optional
import os
import json
import time
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import torch
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from django.conf import settings
from django.core.cache import cache
from langchain_core.documents import Document
import os, logging
from konlpy.tag import Mecab, Okt

logger = logging.getLogger(__name__)

MECAB_DIC = os.getenv("MECAB_DIC", "/opt/homebrew/lib/mecab/dic/mecab-ko-dic")

def make_korean_analyzer(preferred: Optional[str] = None):
    if preferred == "okt":
        return Okt()
    try:
        return Mecab(dicpath=MECAB_DIC)
    except Exception:
        return Okt()

# LangChain 관련 import
try:
    from langchain_community.document_loaders import JSONLoader
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.documents import Document
    from langchain.retrievers import EnsembleRetriever
    from langchain_community.retrievers import BM25Retriever
    from langchain_openai import ChatOpenAI
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    print("⚠️ LangChain 라이브러리 미설치 - RAG 기능 비활성화")
    LANGCHAIN_AVAILABLE = False

# 한국어 NLP 처리
try:
    from konlpy.tag import Mecab, Hannanum, Kkma
    KONLPY_AVAILABLE = True
except ImportError:
    print("⚠️ KoNLPy 미설치 - 한국어 처리 기능 제한")
    KONLPY_AVAILABLE = False

load_dotenv()

# 고급 설정
@dataclass
class AdvancedVideoRAGConfig:
    # FAISS 설정
    use_gpu: bool = torch.cuda.is_available()
    nlist: int = 100  # sqrt(N) 기반으로 동적 조정
    nprobe: int = 10  # nlist/10
    embedding_model: str = "intfloat/multilingual-e5-large"
    embedding_dim: int = 1024
    
    # 임베딩 및 검색 설정
    chunk_size: int = 512
    chunk_overlap: int = 128
    top_k: int = 8  # 5에서 8로 증가
    similarity_threshold: float = 0.75  # 0.8에서 0.75로 낮춤
    
    # 계층적 검색 설정
    frame_level_weight: float = 0.4
    segment_level_weight: float = 0.35
    video_level_weight: float = 0.25
    
    # 캐싱 설정
    cache_ttl_embedding: int = 7200  # 2시간으로 증가
    cache_ttl_analysis: int = 3600   # 1시간으로 증가
    cache_ttl_response: int = 14400  # 4시간으로 증가
    
    # 한국어 처리 설정
    use_korean_morphology: bool = KONLPY_AVAILABLE
    korean_analyzer: str = "mecab"
    
    # 품질 평가 설정
    min_confidence_threshold: float = 0.3
    quality_boost_threshold: float = 0.7
    
    # 모델 설정
    llm_model: str = "gemma2-9b-it"
    max_tokens: int = 1024
    temperature: float = 0.2

class HierarchicalTemporalIndex:
    """계층적 시간축 인덱싱 시스템"""
    
    def __init__(self):
        self.timeline = defaultdict(list)
        self.segments = []
        self.events = []
        self.semantic_clusters = {}  # 의미적 클러스터
        self.quality_index = {}  # 품질별 인덱스
        
    def add_frame_data_advanced(self, timestamp: float, frame_id: int, 
                              caption: str, objects: List[str], scene_data: Dict,
                              quality_score: float = 0.5, attributes: Dict = None):
        """고도화된 프레임 데이터 추가"""
        event = {
            'timestamp': timestamp,
            'frame_id': frame_id,
            'caption': caption,
            'objects': objects,
            'scene_type': scene_data.get('scene_type', ''),
            'lighting': scene_data.get('lighting', 'normal'),
            'activity_level': scene_data.get('activity_level', 'low'),
            'person_count': scene_data.get('person_count', 0),
            'quality_score': quality_score,
            'attributes': attributes or {},
            'semantic_keywords': self._extract_semantic_keywords(caption, objects)
        }
        
        self.timeline[timestamp].append(event)
        self.events.append(event)
        
        # 품질별 인덱싱
        quality_tier = self._get_quality_tier(quality_score)
        if quality_tier not in self.quality_index:
            self.quality_index[quality_tier] = []
        self.quality_index[quality_tier].append(event)
    
    def _extract_semantic_keywords(self, caption: str, objects: List[str]) -> List[str]:
        """의미적 키워드 추출"""
        keywords = []
        
        # 객체에서 키워드
        keywords.extend(objects)
        
        # 캡션에서 중요 단어 추출
        if caption:
            # 간단한 키워드 추출 (실제로는 더 정교한 NLP 사용)
            important_words = ['사람', '남자', '여자', '아이', '차', '건물', '길', '옷', '가방']
            for word in important_words:
                if word in caption:
                    keywords.append(word)
        
        return list(set(keywords))
    
    def _get_quality_tier(self, quality_score: float) -> str:
        """품질 등급 결정"""
        if quality_score >= 0.8:
            return 'high'
        elif quality_score >= 0.5:
            return 'medium'
        else:
            return 'low'
    
    def create_hierarchical_segments(self, segment_duration: float = 30.0):
        """계층적 세그먼트 생성"""
        if not self.events:
            return
        
        max_time = max(event['timestamp'] for event in self.events)
        current_time = 0
        
        while current_time < max_time:
            end_time = min(current_time + segment_duration, max_time)
            
            segment_events = [
                event for event in self.events 
                if current_time <= event['timestamp'] < end_time
            ]
            
            if segment_events:
                segment = self._create_detailed_segment(segment_events, current_time, end_time)
                self.segments.append(segment)
            
            current_time = end_time
        
        # 의미적 클러스터링
        self._create_semantic_clusters()
    
    def _create_detailed_segment(self, events: List[Dict], start_time: float, end_time: float) -> Dict:
        """상세 세그먼트 생성"""
        # 품질 기반 이벤트 가중치
        high_quality_events = [e for e in events if e['quality_score'] >= 0.7]
        representative_events = high_quality_events if high_quality_events else events
        
        # 주요 객체 추출 (품질 가중치 적용)
        object_weights = defaultdict(float)
        for event in events:
            weight = event['quality_score']
            for obj in event['objects']:
                object_weights[obj] += weight
        
        dominant_objects = sorted(object_weights.keys(), key=object_weights.get, reverse=True)[:5]
        
        # 활동 수준 분석
        activity_levels = [e['activity_level'] for e in events if e['activity_level']]
        dominant_activity = max(set(activity_levels), key=activity_levels.count) if activity_levels else 'unknown'
        
        # 인구 통계
        person_counts = [e['person_count'] for e in events]
        avg_person_count = np.mean(person_counts) if person_counts else 0
        
        # 씬 요약 생성
        scene_summary = self._generate_advanced_scene_summary(representative_events, dominant_objects, dominant_activity)
        
        return {
            'start_time': start_time,
            'end_time': end_time,
            'duration': end_time - start_time,
            'events': events,
            'high_quality_events': high_quality_events,
            'dominant_objects': dominant_objects,
            'object_weights': dict(object_weights),
            'scene_summary': scene_summary,
            'dominant_activity': dominant_activity,
            'average_person_count': avg_person_count,
            'quality_distribution': self._calculate_quality_distribution(events),
            'semantic_keywords': self._extract_segment_keywords(events)
        }
    
    def _generate_advanced_scene_summary(self, events: List[Dict], objects: List[str], activity: str) -> str:
        """고도화된 씬 요약 생성"""
        if not events:
            return "빈 구간"
        
        # 대표 이벤트 선택
        best_event = max(events, key=lambda e: e['quality_score'])
        
        # 기본 정보
        scene_type = best_event.get('scene_type', '일반')
        lighting = best_event.get('lighting', 'normal')
        
        # 요약 생성
        summary_parts = []
        
        if scene_type and scene_type != '일반':
            summary_parts.append(f"{scene_type} 환경")
        
        if lighting != 'normal':
            summary_parts.append(f"{lighting} 조명")
        
        if objects:
            if len(objects) == 1:
                summary_parts.append(f"{objects[0]} 중심")
            else:
                summary_parts.append(f"{objects[0]}, {objects[1]} 등 다양한 객체")
        
        if activity != 'unknown':
            summary_parts.append(f"{activity} 활동")
        
        return "에서 ".join(summary_parts) if summary_parts else "일반적인 장면"
    
    def _calculate_quality_distribution(self, events: List[Dict]) -> Dict:
        """품질 분포 계산"""
        if not events:
            return {}
        
        quality_scores = [e['quality_score'] for e in events]
        
        return {
            'average': np.mean(quality_scores),
            'max': np.max(quality_scores),
            'min': np.min(quality_scores),
            'std': np.std(quality_scores),
            'high_quality_ratio': sum(1 for q in quality_scores if q >= 0.7) / len(quality_scores)
        }
    
    def _extract_segment_keywords(self, events: List[Dict]) -> List[str]:
        """세그먼트 키워드 추출"""
        all_keywords = []
        for event in events:
            all_keywords.extend(event.get('semantic_keywords', []))
        
        # 빈도 기반 상위 키워드
        keyword_counts = Counter(all_keywords)
        return [keyword for keyword, count in keyword_counts.most_common(10)]
    
    def _create_semantic_clusters(self):
        """의미적 클러스터 생성"""
        # 키워드 기반 클러스터링
        keyword_to_events = defaultdict(list)
        
        for event in self.events:
            for keyword in event.get('semantic_keywords', []):
                keyword_to_events[keyword].append(event)
        
        # 클러스터 생성 (최소 2개 이상의 이벤트)
        for keyword, events in keyword_to_events.items():
            if len(events) >= 2:
                self.semantic_clusters[keyword] = {
                    'events': events,
                    'count': len(events),
                    'quality_score': np.mean([e['quality_score'] for e in events]),
                    'time_span': max(e['timestamp'] for e in events) - min(e['timestamp'] for e in events)
                }

class AdvancedKoreanTextProcessor:
    """고도화된 한국어 텍스트 전처리"""
    
    def __init__(self, analyzer: str = "mecab"):
        self.analyzer_type = analyzer
        self.analyzer = make_korean_analyzer(analyzer)
        
        # 확장된 패턴 정의
        self.temporal_patterns = [
            r'(\d+)초', r'(\d+)분', r'(\d+)시간',
            r'처음에?', r'마지막에?', r'중간에?', r'시작할?\s?때?', r'끝날?\s?때?',
            r'먼저', r'나중에?', r'그\s?다음에?', r'이후에?', r'전에?',
            r'언제', r'몇\s?분', r'몇\s?초', r'얼마나', r'동안',
            r'(\d+)시\s?(\d+)분?', r'오전', r'오후', r'새벽', r'저녁'
        ]
        
        self.person_patterns = [
            r'사람', r'남자', r'여자', r'아이', r'어린이', r'청소년', r'성인', r'노인',
            r'소년', r'소녀', r'남성', r'여성', r'인물', r'보행자'
        ]
        
        self.object_patterns = [
            r'차', r'자동차', r'버스', r'트럭', r'오토바이', r'자전거',
            r'가방', r'핸드백', r'백팩', r'모자', r'안경', r'옷', r'신발'
        ]
        
        self.action_patterns = [
            r'걷[는다가기]', r'뛰[는다가기]', r'달리[는다가기]', r'서[있는다가]',
            r'앉[아있는다가]', r'움직[이인다가임]', r'지나[가는다간]'
        ]
    
    def extract_temporal_markers_advanced(self, text: str) -> Dict[str, List[str]]:
        """고도화된 시간 표현 추출"""
        import re
        
        markers = {
            'time_expressions': [],
            'sequence_markers': [],
            'duration_markers': [],
            'specific_times': []
        }
        
        for pattern in self.temporal_patterns:
            matches = re.findall(pattern, text)
            if matches:
                if any(word in pattern for word in ['초', '분', '시간']):
                    markers['duration_markers'].extend(matches)
                elif any(word in pattern for word in ['처음', '마지막', '먼저', '나중']):
                    markers['sequence_markers'].extend(matches)
                elif any(word in pattern for word in ['시', '오전', '오후']):
                    markers['specific_times'].extend(matches)
                else:
                    markers['time_expressions'].extend(matches)
        
        return markers
    
    def analyze_question_intent_advanced(self, question: str) -> Dict[str, Any]:
        """고도화된 질문 의도 분석"""
        intent = {
            'primary_type': 'general',
            'secondary_types': [],
            'confidence': 0.5,
            'entities': {
                'temporal': [],
                'persons': [],
                'objects': [],
                'actions': [],
                'attributes': []
            },
            'complexity': 'simple',
            'required_analysis_level': 'frame'
        }
        
        q = question.lower()
        
        # 시간 관련 분석
        temporal_markers = self.extract_temporal_markers_advanced(question)
        if any(temporal_markers.values()):
            intent['primary_type'] = 'temporal'
            intent['confidence'] += 0.3
            intent['entities']['temporal'] = temporal_markers
            intent['required_analysis_level'] = 'segment'
        
        # 사람 관련 분석
        import re
        person_matches = []
        for pattern in self.person_patterns:
            matches = re.findall(pattern, q)
            person_matches.extend(matches)
        
        if person_matches:
            if intent['primary_type'] == 'general':
                intent['primary_type'] = 'person_analysis'
            else:
                intent['secondary_types'].append('person_analysis')
            intent['confidence'] += 0.2
            intent['entities']['persons'] = person_matches
        
        # 객체 관련 분석
        object_matches = []
        for pattern in self.object_patterns:
            matches = re.findall(pattern, q)
            object_matches.extend(matches)
        
        if object_matches:
            if intent['primary_type'] == 'general':
                intent['primary_type'] = 'object_detection'
            else:
                intent['secondary_types'].append('object_detection')
            intent['confidence'] += 0.2
            intent['entities']['objects'] = object_matches
        
        # 행동 관련 분석
        action_matches = []
        for pattern in self.action_patterns:
            matches = re.findall(pattern, q)
            action_matches.extend(matches)
        
        if action_matches:
            if intent['primary_type'] == 'general':
                intent['primary_type'] = 'action_recognition'
            else:
                intent['secondary_types'].append('action_recognition')
            intent['confidence'] += 0.2
            intent['entities']['actions'] = action_matches
            intent['required_analysis_level'] = 'segment'
        
        # 복잡도 판단
        complexity_indicators = len(intent['secondary_types']) + len([v for v in intent['entities'].values() if v])
        
        if complexity_indicators >= 3:
            intent['complexity'] = 'complex'
            intent['required_analysis_level'] = 'video'
        elif complexity_indicators >= 2:
            intent['complexity'] = 'moderate'
        
        # 속성 관련 키워드 확인
        attribute_keywords = ['색깔', '옷', '모자', '가방', '안경', '키', '몸무게', '나이']
        found_attributes = [attr for attr in attribute_keywords if attr in q]
        if found_attributes:
            intent['entities']['attributes'] = found_attributes
            if intent['primary_type'] == 'general':
                intent['primary_type'] = 'attribute_analysis'
            else:
                intent['secondary_types'].append('attribute_analysis')
        
        return intent

class MultiLevelCacheManager:
    """다층 캐싱 시스템"""
    
    def __init__(self, config: AdvancedVideoRAGConfig):
        self.config = config
        self.memory_cache = {}  # 메모리 캐시
        self.quality_cache = {}  # 품질별 캐시
        
    def get_cache_key(self, video_id: str, query: str, cache_type: str, quality_level: str = 'all') -> str:
        """계층적 캐시 키 생성"""
        content = f"{video_id}:{query}:{cache_type}:{quality_level}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_hierarchical_cache(self, video_id: str, query: str, analysis_level: str) -> Optional[Dict]:
        """계층적 캐시 조회"""
        cache_levels = ['frame', 'segment', 'video']
        
        # 요청된 레벨부터 상위 레벨까지 검색
        for level in cache_levels[cache_levels.index(analysis_level):]:
            key = self.get_cache_key(video_id, query, f"search_{level}")
            
            # Django 캐시 조회
            cached = cache.get(key)
            if cached:
                return {
                    'results': cached['results'],
                    'cached_level': level,
                    'timestamp': cached['timestamp']
                }
            
            # 메모리 캐시 조회
            if key in self.memory_cache:
                return self.memory_cache[key]
        
        return None
    
    def set_hierarchical_cache(self, video_id: str, query: str, analysis_level: str, results: List[Dict]):
        """계층적 캐시 저장"""
        key = self.get_cache_key(video_id, query, f"search_{analysis_level}")
        
        cache_data = {
            'results': results,
            'analysis_level': analysis_level,
            'timestamp': time.time()
        }
        
        # Django 캐시 저장
        cache.set(key, cache_data, timeout=self.config.cache_ttl_analysis)
        
        # 메모리 캐시 저장 (제한된 크기)
        if len(self.memory_cache) < 100:
            self.memory_cache[key] = cache_data
    
    def get_quality_aware_cache(self, video_id: str, query: str, min_quality: float = 0.5) -> Optional[List[Dict]]:
        """품질 인식 캐시 조회"""
        quality_levels = ['high', 'medium', 'low']
        quality_thresholds = {'high': 0.7, 'medium': 0.5, 'low': 0.0}
        
        for quality_level in quality_levels:
            if quality_thresholds[quality_level] >= min_quality:
                key = self.get_cache_key(video_id, query, "quality_search", quality_level)
                cached = cache.get(key)
                if cached:
                    return cached
        
        return None

class SuperiorVideoRAGSystem:
    """최고급 비디오 분석 RAG 시스템"""
    
    def __init__(self, config: Optional[AdvancedVideoRAGConfig] = None):
        self.config = config or AdvancedVideoRAGConfig()
        self.device = "cuda" if self.config.use_gpu and torch.cuda.is_available() else "cpu"
        
        # 고도화된 컴포넌트 초기화
        self.cache_manager = MultiLevelCacheManager(self.config)
        self.korean_processor = AdvancedKoreanTextProcessor(self.config.korean_analyzer)
        
        # 시스템 상태
        self._embeddings_initialized = False
        self._llm_initialized = False
        
        print(f"🚀 Superior VideoRAG 시스템 초기화 (디바이스: {self.device})")
        
        if not LANGCHAIN_AVAILABLE:
            print("⚠️ LangChain 미설치 - 기본 RAG 기능만 사용")
            return
        
        try:
            self._init_advanced_embeddings()
            self._init_llm()
        except Exception as e:
            print(f"⚠️ RAG 시스템 초기화 부분 실패: {e}")
        
        # 비디오 데이터베이스 저장소
        self.video_databases = {}
        self.temporal_indexes = {}
        self.quality_indexes = {}
        
        print("✅ Superior VideoRAG 시스템 초기화 완료")
    
    def _init_advanced_embeddings(self):
        """고도화된 임베딩 모델 초기화"""
        try:
            model_kwargs = {
                "device": self.device,
                "trust_remote_code": True
            }
            encode_kwargs = {
                'normalize_embeddings': True,
                'batch_size': 64,  # 배치 크기 증가
                'show_progress_bar': False
            }
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.config.embedding_model,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
            self._embeddings_initialized = True
            print(f"✅ 고도화된 다국어 임베딩 모델 로드 완료: {self.config.embedding_model}")
        except Exception as e:
            print(f"⚠️ 임베딩 모델 초기화 실패: {e}")
            self._embeddings_initialized = False
    
    def _init_llm(self):
        """LLM 초기화"""
        try:
            self.llm = ChatOpenAI(
                model=self.config.llm_model,
                openai_api_key=os.environ["GROQ_API_KEY"],
                openai_api_base="https://api.groq.com/openai/v1",
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            self._llm_initialized = True
            print(f"✅ 고도화된 LLM 초기화 완료: {self.config.llm_model}")
        except Exception as e:
            print(f"⚠️ LLM 초기화 실패: {e}")
            self._llm_initialized = False
    
    def process_video_analysis_json_advanced(self, json_file_path: str, video_id: str) -> bool:
        """최고급 비디오 분석 JSON 처리"""
        try:
            if not os.path.exists(json_file_path):
                print(f"⚠️ JSON 파일을 찾을 수 없음: {json_file_path}")
                return False
            
            print(f"📄 고도화된 JSON 분석 파일 처리 중: {json_file_path}")
            
            with open(json_file_path, 'r', encoding='utf-8') as f:
                analysis_data = json.load(f)
            
            # 계층적 시간축 인덱스 생성
            temporal_index = HierarchicalTemporalIndex()
            
            # 다층 문서 생성
            frame_documents = []
            segment_documents = []
            semantic_documents = []
            
            frame_results = analysis_data.get('frame_results', [])
            video_metadata = analysis_data.get('metadata', {})
            
            print(f"📊 처리할 프레임 수: {len(frame_results)}")
            
            # 프레임별 고도화된 문서 생성
            for i, frame_result in enumerate(frame_results):
                frame_id = frame_result.get('image_id', i)
                timestamp = frame_result.get('timestamp', 0)
                
                # 다양한 캡션 소스 통합
                caption = (frame_result.get('final_caption') or 
                          frame_result.get('enhanced_caption') or 
                          frame_result.get('caption') or '')
                
                objects = frame_result.get('objects', [])
                scene_analysis = frame_result.get('scene_analysis', {})
                
                # 품질 점수 계산
                quality_score = self._calculate_frame_quality_score(frame_result)
                
                # 계층적 시간축 인덱스에 추가
                temporal_index.add_frame_data_advanced(
                    timestamp, frame_id, caption, 
                    [obj.get('class', '') for obj in objects],
                    {
                        'scene_type': scene_analysis.get('scene_classification', {}).get('location', {}).get('label', ''),
                        'lighting': scene_analysis.get('lighting', 'normal'),
                        'activity_level': scene_analysis.get('activity_level', 'low'),
                        'person_count': len([obj for obj in objects if obj.get('class') == 'person'])
                    },
                    quality_score,
                    frame_result.get('attributes', {})
                )
                
                # 고도화된 프레임 문서 생성
                content_parts = self._build_advanced_frame_content(
                    frame_id, timestamp, caption, objects, scene_analysis, quality_score
                )
                
                if content_parts:
                    metadata = {
                        'video_id': video_id,
                        'frame_id': frame_id,
                        'timestamp': timestamp,
                        'objects': [obj.get('class', '') for obj in objects],
                        'scene_type': scene_analysis.get('scene_classification', {}).get('location', {}).get('label', ''),
                        'quality_score': quality_score,
                        'quality_tier': self._get_quality_tier(quality_score),
                        'level': 'frame'
                    }
                    
                    frame_documents.append(Document(
                        page_content='. '.join(content_parts), 
                        metadata=metadata
                    ))
            
            # 계층적 세그먼트 생성
            temporal_index.create_hierarchical_segments()
            
            print(f"📊 생성된 세그먼트 수: {len(temporal_index.segments)}")
            
            # 고도화된 세그먼트별 문서 생성
            for segment in temporal_index.segments:
                segment_content = self._build_advanced_segment_content(segment)
                
                metadata = {
                    'video_id': video_id,
                    'start_time': segment['start_time'],
                    'end_time': segment['end_time'],
                    'duration': segment['duration'],
                    'dominant_objects': segment['dominant_objects'],
                    'scene_summary': segment['scene_summary'],
                    'quality_distribution': segment['quality_distribution'],
                    'average_person_count': segment['average_person_count'],
                    'level': 'segment'
                }
                
                segment_documents.append(Document(
                    page_content=segment_content,
                    metadata=metadata
                ))
            
            # 의미적 클러스터 문서 생성
            for keyword, cluster in temporal_index.semantic_clusters.items():
                cluster_content = self._build_semantic_cluster_content(keyword, cluster)
                
                metadata = {
                    'video_id': video_id,
                    'semantic_keyword': keyword,
                    'event_count': cluster['count'],
                    'quality_score': cluster['quality_score'],
                    'time_span': cluster['time_span'],
                    'level': 'semantic'
                }
                
                semantic_documents.append(Document(
                    page_content=cluster_content,
                    metadata=metadata
                ))
            # 전체 비디오 문서 생성
            video_document = self._build_comprehensive_video_document(analysis_data, temporal_index)
            video_metadata_doc = {
                'video_id': video_id,
                'level': 'video',
                'total_frames': len(frame_results),
                'duration': video_metadata.get('duration', 0),
                'analysis_type': video_metadata.get('analysis_type', 'unknown'),
                'quality_summary': self._calculate_video_quality_summary(temporal_index)
            }
            
            all_documents = frame_documents + segment_documents + semantic_documents + [Document(
                page_content=video_document,
                metadata=video_metadata_doc
            )]
            
            print(f"📊 총 생성된 문서 수: {len(all_documents)}")
            
            # 최고급 계층적 벡터 DB 생성
            success = self._create_superior_hierarchical_vector_db(video_id, all_documents)
            
            if success:
                # 시간축 인덱스 및 품질 인덱스 저장
                self.temporal_indexes[video_id] = temporal_index
                self.quality_indexes[video_id] = self._create_quality_index(frame_results)
                
                print(f"✅ 비디오 {video_id} 최고급 RAG DB 생성 완료: {len(all_documents)}개 문서")
                return True
            
            return False
            
        except Exception as e:
            print(f"❌ 최고급 RAG DB 생성 실패: {e}")
            import traceback
            print(f"상세 오류: {traceback.format_exc()}")
            return False
    
    def _calculate_frame_quality_score(self, frame_result: Dict) -> float:
        """프레임 품질 점수 계산"""
        quality_factors = []
        
        # 캡션 품질
        caption = frame_result.get('final_caption') or frame_result.get('enhanced_caption') or frame_result.get('caption', '')
        if caption:
            caption_quality = min(1.0, len(caption.split()) / 10)  # 단어 수 기반
            quality_factors.append(caption_quality * 0.3)
        
        # 객체 감지 품질
        objects = frame_result.get('objects', [])
        if objects:
            avg_confidence = np.mean([obj.get('confidence', 0) for obj in objects])
            quality_factors.append(avg_confidence * 0.4)
        
        # 속성 분석 품질
        persons = frame_result.get('persons', [])
        if persons:
            attr_confidences = []
            for person in persons:
                attrs = person.get('attributes', {})
                for attr_name, attr_data in attrs.items():
                    if isinstance(attr_data, dict) and 'confidence' in attr_data:
                        attr_confidences.append(attr_data['confidence'])
            
            if attr_confidences:
                avg_attr_confidence = np.mean(attr_confidences)
                quality_factors.append(avg_attr_confidence * 0.3)
        
        return np.mean(quality_factors) if quality_factors else 0.5
    
    def _get_quality_tier(self, quality_score: float) -> str:
        """품질 등급 결정"""
        if quality_score >= self.config.quality_boost_threshold:
            return 'high'
        elif quality_score >= self.config.min_confidence_threshold:
            return 'medium'
        else:
            return 'low'
    
    def _build_advanced_frame_content(self, frame_id: int, timestamp: float, 
                                    caption: str, objects: List, scene_analysis: Dict, 
                                    quality_score: float) -> List[str]:
        """고도화된 프레임 내용 구성"""
        content_parts = []
        
        # 품질 정보 포함
        quality_indicator = "고품질" if quality_score >= 0.7 else "중품질" if quality_score >= 0.4 else "저품질"
        
        if caption:
            content_parts.append(f"프레임 {frame_id} ({timestamp:.1f}초, {quality_indicator}): {caption}")
        
        # 객체 정보 (신뢰도 포함)
        if objects:
            high_conf_objects = [obj for obj in objects if obj.get('confidence', 0) > 0.5]
            if high_conf_objects:
                object_list = [f"{obj.get('class', '')}({obj.get('confidence', 0):.2f})" 
                             for obj in high_conf_objects]
                content_parts.append(f"고신뢰도 객체: {', '.join(object_list)}")
            
            # 모든 객체도 포함
            all_objects = [obj.get('class', '') for obj in objects if obj.get('class')]
            if all_objects:
                content_parts.append(f"전체 감지 객체: {', '.join(set(all_objects))}")
        
        # 상세 장면 분석
        if scene_analysis:
            scene_details = []
            
            scene_class = scene_analysis.get('scene_classification', {})
            if scene_class:
                location = scene_class.get('location', {}).get('label', '')
                time_of_day = scene_class.get('time', {}).get('label', '')
                if location or time_of_day:
                    scene_details.append(f"장면: {location} {time_of_day}".strip())
            
            # 추가 씬 정보
            if scene_analysis.get('lighting'):
                scene_details.append(f"조명: {scene_analysis['lighting']}")
            
            if scene_analysis.get('activity_level'):
                scene_details.append(f"활동도: {scene_analysis['activity_level']}")
            
            if scene_details:
                content_parts.append(", ".join(scene_details))
            
            # OCR 텍스트
            ocr_text = scene_analysis.get('ocr_text', '')
            if ocr_text:
                content_parts.append(f"추출된 텍스트: {ocr_text}")
        
        return content_parts
    
    def _build_advanced_segment_content(self, segment: Dict) -> str:
        """고도화된 세그먼트 내용 구성"""
        start_time = segment['start_time']
        end_time = segment['end_time']
        duration = segment['duration']
        scene_summary = segment['scene_summary']
        dominant_objects = segment['dominant_objects']
        avg_person_count = segment['average_person_count']
        quality_dist = segment['quality_distribution']
        
        content_parts = []
        
        # 기본 시간 정보
        content_parts.append(f"{start_time:.1f}초-{end_time:.1f}초 구간 ({duration:.1f}초 지속)")
        
        # 씬 요약
        content_parts.append(f"장면 요약: {scene_summary}")
        
        # 주요 객체 (가중치 포함)
        if dominant_objects:
            content_parts.append(f"주요 객체: {', '.join(dominant_objects[:3])}")
        
        # 인원 정보
        if avg_person_count > 0:
            content_parts.append(f"평균 인원: {avg_person_count:.1f}명")
        
        # 품질 정보
        if quality_dist and 'average' in quality_dist:
            quality_desc = "고품질" if quality_dist['average'] >= 0.7 else "중품질" if quality_dist['average'] >= 0.4 else "저품질"
            content_parts.append(f"분석 품질: {quality_desc} (평균 {quality_dist['average']:.2f})")
        
        # 활동 정보
        if 'dominant_activity' in segment:
            content_parts.append(f"주요 활동: {segment['dominant_activity']}")
        
        # 고품질 이벤트 수
        high_quality_events = len(segment.get('high_quality_events', []))
        total_events = len(segment.get('events', []))
        if total_events > 0:
            content_parts.append(f"고품질 이벤트: {high_quality_events}/{total_events}")
        
        return ". ".join(content_parts)
    
    def _build_semantic_cluster_content(self, keyword: str, cluster: Dict) -> str:
        """의미적 클러스터 내용 구성"""
        event_count = cluster['count']
        quality_score = cluster['quality_score']
        time_span = cluster['time_span']
        
        content = f"'{keyword}' 관련 장면들: {event_count}개 이벤트에서 발생"
        
        if time_span > 0:
            content += f", {time_span:.1f}초에 걸쳐 나타남"
        
        quality_desc = "고품질" if quality_score >= 0.7 else "중품질" if quality_score >= 0.4 else "저품질"
        content += f", 분석 품질: {quality_desc}"
        
        # 이벤트 세부 정보
        events = cluster.get('events', [])
        if events:
            timestamps = [e['timestamp'] for e in events]
            content += f", 주요 출현 시점: {min(timestamps):.1f}초-{max(timestamps):.1f}초"
        
        return content
    
    def _build_comprehensive_video_document(self, analysis_data: Dict, temporal_index: HierarchicalTemporalIndex) -> str:
        """종합적인 비디오 문서 구성"""
        metadata = analysis_data.get('metadata', {})
        
        content_parts = [
            f"비디오 종합 분석 결과:",
            f"- 총 길이: {metadata.get('duration', 0)}초",
            f"- 분석된 프레임: {len(analysis_data.get('frame_results', []))}개",
            f"- 생성된 구간: {len(temporal_index.segments)}개",
            f"- 의미적 클러스터: {len(temporal_index.semantic_clusters)}개"
        ]
        
        # 전체 비디오의 주요 객체 (품질 가중치 적용)
        all_objects_weighted = defaultdict(float)
        for event in temporal_index.events:
            weight = event['quality_score']
            for obj in event.get('objects', []):
                all_objects_weighted[obj] += weight
        
        if all_objects_weighted:
            top_objects = sorted(all_objects_weighted.items(), key=lambda x: x[1], reverse=True)[:5]
            content_parts.append(f"- 주요 객체 (가중치): {', '.join([f'{obj}({weight:.1f})' for obj, weight in top_objects])}")
        
        # 품질별 분포
        quality_tiers = defaultdict(int)
        for event in temporal_index.events:
            tier = temporal_index._get_quality_tier(event['quality_score'])
            quality_tiers[tier] += 1
        
        if quality_tiers:
            quality_summary = ', '.join([f"{tier}품질: {count}개" for tier, count in quality_tiers.items()])
            content_parts.append(f"- 품질 분포: {quality_summary}")
        
        # 시간적 패턴
        if temporal_index.events:
            time_span = max(e['timestamp'] for e in temporal_index.events) - min(e['timestamp'] for e in temporal_index.events)
            content_parts.append(f"- 분석 시간 범위: {time_span:.1f}초")
        
        # 의미적 키워드 요약
        if temporal_index.semantic_clusters:
            top_keywords = sorted(temporal_index.semantic_clusters.items(), 
                                key=lambda x: x[1]['count'], reverse=True)[:5]
            keyword_summary = ', '.join([f"{keyword}({data['count']}회)" for keyword, data in top_keywords])
            content_parts.append(f"- 주요 의미 키워드: {keyword_summary}")
        
        return '\n'.join(content_parts)
    
    def _calculate_video_quality_summary(self, temporal_index: HierarchicalTemporalIndex) -> Dict:
        """비디오 전체 품질 요약 계산"""
        if not temporal_index.events:
            return {}
        
        quality_scores = [e['quality_score'] for e in temporal_index.events]
        
        return {
            'average_quality': np.mean(quality_scores),
            'max_quality': np.max(quality_scores),
            'min_quality': np.min(quality_scores),
            'quality_std': np.std(quality_scores),
            'high_quality_ratio': sum(1 for q in quality_scores if q >= 0.7) / len(quality_scores),
            'total_events': len(temporal_index.events)
        }
    
    def _create_quality_index(self, frame_results: List[Dict]) -> Dict:
        """품질별 인덱스 생성"""
        quality_index = {'high': [], 'medium': [], 'low': []}
        
        for frame_result in frame_results:
            quality_score = self._calculate_frame_quality_score(frame_result)
            tier = self._get_quality_tier(quality_score)
            quality_index[tier].append({
                'frame_id': frame_result.get('image_id', 0),
                'timestamp': frame_result.get('timestamp', 0),
                'quality_score': quality_score
            })
        
        return quality_index
    
    def _create_superior_hierarchical_vector_db(self, video_id: str, documents: List[Document]) -> bool:
        """최고급 계층적 벡터 DB 생성"""
        if not self._embeddings_initialized or not documents:
            return False
        
        try:
            # 문서 레벨별 분리
            frame_docs = [doc for doc in documents if doc.metadata.get('level') == 'frame']
            segment_docs = [doc for doc in documents if doc.metadata.get('level') == 'segment']
            semantic_docs = [doc for doc in documents if doc.metadata.get('level') == 'semantic']
            video_docs = [doc for doc in documents if doc.metadata.get('level') == 'video']
            
            print(f"📊 문서 분리: 프레임 {len(frame_docs)}, 세그먼트 {len(segment_docs)}, 의미 {len(semantic_docs)}, 비디오 {len(video_docs)}")
            
            # 전체 통합 FAISS 인덱스 생성
            db = FAISS.from_documents(documents, embedding=self.embeddings)
            
            # 레벨별 전용 검색기 생성
            retrievers = {}
            
            if frame_docs:
                frame_db = FAISS.from_documents(frame_docs, embedding=self.embeddings)
                retrievers['frame'] = frame_db.as_retriever(
                    search_type="similarity",
                    search_kwargs={'k': self.config.top_k}
                )
            
            if segment_docs:
                segment_db = FAISS.from_documents(segment_docs, embedding=self.embeddings)
                retrievers['segment'] = segment_db.as_retriever(
                    search_type="similarity",
                    search_kwargs={'k': max(1, self.config.top_k // 2)}
                )
            
            if semantic_docs:
                semantic_db = FAISS.from_documents(semantic_docs, embedding=self.embeddings)
                retrievers['semantic'] = semantic_db.as_retriever(
                    search_type="similarity", 
                    search_kwargs={'k': max(1, self.config.top_k // 3)}
                )
            
            if video_docs:
                video_db = FAISS.from_documents(video_docs, embedding=self.embeddings)
                retrievers['video'] = video_db.as_retriever(
                    search_type="similarity",
                    search_kwargs={'k': 1}
                )
            
            # 품질별 검색기 생성
            quality_retrievers = {}
            for quality_tier in ['high', 'medium', 'low']:
                quality_docs = [doc for doc in documents 
                              if doc.metadata.get('quality_tier') == quality_tier]
                
                if quality_docs:
                    quality_db = FAISS.from_documents(quality_docs, embedding=self.embeddings)
                    quality_retrievers[quality_tier] = quality_db.as_retriever(
                        search_type="similarity",
                        search_kwargs={'k': self.config.top_k}
                    )
            
            # 가중치 기반 앙상블 검색기 구성
            try:
                # BM25 검색기 추가
                bm25_retriever = BM25Retriever.from_documents(documents)
                bm25_retriever.k = self.config.top_k
                
                # 주요 검색기들
                main_retrievers = []
                weights = []
                
                if 'frame' in retrievers:
                    main_retrievers.append(retrievers['frame'])
                    weights.append(self.config.frame_level_weight)
                
                if 'segment' in retrievers:
                    main_retrievers.append(retrievers['segment'])
                    weights.append(self.config.segment_level_weight)
                
                if 'video' in retrievers:
                    main_retrievers.append(retrievers['video'])
                    weights.append(self.config.video_level_weight)
                
                main_retrievers.append(bm25_retriever)
                weights.append(0.15)  # BM25 가중치
                
                # 가중치 정규화
                total_weight = sum(weights)
                normalized_weights = [w/total_weight for w in weights]
                
                ensemble_retriever = EnsembleRetriever(
                    retrievers=main_retrievers,
                    weights=normalized_weights
                )
                
            except Exception as e:
                print(f"⚠️ 앙상블 검색기 생성 실패: {e}")
                ensemble_retriever = db.as_retriever(
                    search_type="similarity",
                    search_kwargs={'k': self.config.top_k}
                )
            
            # 데이터베이스 정보 저장
            self.video_databases[video_id] = {
                'db': db,
                'retriever': ensemble_retriever,
                'level_retrievers': retrievers,
                'quality_retrievers': quality_retrievers,
                'documents': documents,
                'created_at': datetime.now(),
                'config': self.config,
                'document_stats': {
                    'total': len(documents),
                    'frame': len(frame_docs),
                    'segment': len(segment_docs),
                    'semantic': len(semantic_docs),
                    'video': len(video_docs)
                }
            }
            
            print(f"✅ 최고급 계층적 벡터 DB 생성 완료")
            return True
            
        except Exception as e:
            print(f"❌ 최고급 벡터 DB 생성 실패: {e}")
            import traceback
            print(f"상세 오류: {traceback.format_exc()}")
            return False
    
    def intelligent_search_video_content(self, video_id: str, query: str, 
                                       context: Optional[Dict] = None) -> List[Dict]:
        """지능형 비디오 내용 검색"""
        if video_id not in self.video_databases:
            print(f"⚠️ 비디오 {video_id}의 RAG DB가 없음")
            return []
        
        try:
            # 질문 의도 분석
            intent = self.korean_processor.analyze_question_intent_advanced(query)
            
            print(f"🧠 질문 의도 분석:")
            print(f"   - 주 타입: {intent['primary_type']}")
            print(f"   - 부 타입: {intent['secondary_types']}")
            print(f"   - 신뢰도: {intent['confidence']:.2f}")
            print(f"   - 복잡도: {intent['complexity']}")
            print(f"   - 필요 분석 레벨: {intent['required_analysis_level']}")
            
            # 계층적 캐시 확인
            cached_result = self.cache_manager.get_hierarchical_cache(
                video_id, query, intent['required_analysis_level']
            )
            if cached_result:
                print(f"🎯 계층적 캐시에서 결과 반환 (레벨: {cached_result['cached_level']})")
                return cached_result['results']
            
            # 품질 인식 캐시 확인
            min_quality = 0.7 if intent['confidence'] > 0.8 else 0.5
            quality_cached = self.cache_manager.get_quality_aware_cache(video_id, query, min_quality)
            if quality_cached:
                print(f"🎯 품질 인식 캐시에서 결과 반환")
                return quality_cached
            
            # 지능형 검색 전략 실행
            results = self._execute_intelligent_search_strategy(video_id, query, intent)
            
            # 시간적 컨텍스트 추가
            if intent['primary_type'] == 'temporal' and video_id in self.temporal_indexes:
                results = self._add_advanced_temporal_context(video_id, query, results, intent)
            
            # 품질 기반 후처리
            results = self._apply_quality_boost(results, intent)
            
            # 결과 캐싱
            self.cache_manager.set_hierarchical_cache(
                video_id, query, intent['required_analysis_level'], results
            )
            
            print(f"🔍 지능형 검색 완료: {len(results)}개 결과")
            print(f"   - 사용된 전략: {intent['primary_type']}")
            print(f"   - 품질 부스트: {'적용' if intent['confidence'] > 0.7 else '미적용'}")
            
            return results
            
        except Exception as e:
            print(f"❌ 지능형 검색 실패: {e}")
            import traceback
            print(f"상세 오류: {traceback.format_exc()}")
            return []
    
    def _execute_intelligent_search_strategy(self, video_id: str, query: str, intent: Dict) -> List[Dict]:
        """지능형 검색 전략 실행"""
        db_info = self.video_databases[video_id]
        
        # 의도별 맞춤 검색
        if intent['primary_type'] == 'temporal':
            return self._temporal_focused_search(db_info, query, intent)
        elif intent['primary_type'] == 'person_analysis':
            return self._person_focused_search(db_info, query, intent)
        elif intent['primary_type'] == 'object_detection':
            return self._object_focused_search(db_info, query, intent)
        elif intent['primary_type'] == 'attribute_analysis':
            return self._attribute_focused_search(db_info, query, intent)
        elif intent['primary_type'] == 'semantic':
            return self._semantic_focused_search(db_info, query, intent)
        else:
            return self._comprehensive_search(db_info, query, intent)
    
    def _temporal_focused_search(self, db_info: Dict, query: str, intent: Dict) -> List[Dict]:
        """시간 중심 검색"""
        # 세그먼트 레벨 우선 검색
        if 'segment' in db_info['level_retrievers']:
            docs = db_info['level_retrievers']['segment'].get_relevant_documents(query)
        else:
            docs = db_info['retriever'].get_relevant_documents(query)
        
        return self._format_search_results_advanced(docs, search_type='temporal')
    
    def _person_focused_search(self, db_info: Dict, query: str, intent: Dict) -> List[Dict]:
        """사람 중심 검색"""
        # 프레임 레벨에서 사람 관련 정보 검색
        if 'frame' in db_info['level_retrievers']:
            docs = db_info['level_retrievers']['frame'].get_relevant_documents(query)
        else:
            docs = db_info['retriever'].get_relevant_documents(query)
        
        # 사람 관련 문서만 필터링
        person_docs = [doc for doc in docs 
                      if any(obj.lower() in ['person', 'man', 'woman', 'people'] 
                            for obj in doc.metadata.get('objects', []))]
        
        return self._format_search_results_advanced(person_docs or docs, search_type='person')
    
    def _object_focused_search(self, db_info: Dict, query: str, intent: Dict) -> List[Dict]:
        """객체 중심 검색"""
        # 프레임 레벨 검색
        if 'frame' in db_info['level_retrievers']:
            docs = db_info['level_retrievers']['frame'].get_relevant_documents(query)
        else:
            docs = db_info['retriever'].get_relevant_documents(query)
        
        return self._format_search_results_advanced(docs, search_type='object')
    
    def _attribute_focused_search(self, db_info: Dict, query: str, intent: Dict) -> List[Dict]:
        """속성 중심 검색"""
        # 고품질 문서 우선 검색
        if 'high' in db_info['quality_retrievers']:
            docs = db_info['quality_retrievers']['high'].get_relevant_documents(query)
            if not docs and 'medium' in db_info['quality_retrievers']:
                docs = db_info['quality_retrievers']['medium'].get_relevant_documents(query)
        else:
            docs = db_info['retriever'].get_relevant_documents(query)
        
        return self._format_search_results_advanced(docs, search_type='attribute')
    
    def _semantic_focused_search(self, db_info: Dict, query: str, intent: Dict) -> List[Dict]:
        """의미 중심 검색"""
        # 의미적 클러스터 우선 검색
        if 'semantic' in db_info['level_retrievers']:
            semantic_docs = db_info['level_retrievers']['semantic'].get_relevant_documents(query)
            if semantic_docs:
                return self._format_search_results_advanced(semantic_docs, search_type='semantic')
        
        # 세그먼트 레벨 보조 검색
        if 'segment' in db_info['level_retrievers']:
            docs = db_info['level_retrievers']['segment'].get_relevant_documents(query)
        else:
            docs = db_info['retriever'].get_relevant_documents(query)
        
        return self._format_search_results_advanced(docs, search_type='semantic')
    
    def _comprehensive_search(self, db_info: Dict, query: str, intent: Dict) -> List[Dict]:
        """종합 검색"""
        # 모든 레벨에서 검색하여 통합
        all_results = []
        
        # 각 레벨별 검색
        for level, retriever in db_info['level_retrievers'].items():
            try:
                docs = retriever.get_relevant_documents(query)
                level_results = self._format_search_results_advanced(docs, search_type=level)
                all_results.extend(level_results)
            except Exception as e:
                print(f"⚠️ {level} 레벨 검색 실패: {e}")
        
        # 중복 제거 및 점수 기반 정렬
        unique_results = self._deduplicate_and_rank_results(all_results)
        
        return unique_results[:self.config.top_k]
    
    def _format_search_results_advanced(self, docs: List[Document], search_type: str = 'general') -> List[Dict]:
        """고도화된 검색 결과 포맷팅"""
        results = []
        for doc in docs:
            result = {
                'content': doc.page_content,
                'metadata': doc.metadata,
                'level': doc.metadata.get('level', 'unknown'),
                'search_type': search_type,
                'frame_id': doc.metadata.get('frame_id'),
                'timestamp': doc.metadata.get('timestamp'),
                'objects': doc.metadata.get('objects', []),
                'quality_score': doc.metadata.get('quality_score', 0.5),
                'quality_tier': doc.metadata.get('quality_tier', 'medium')
            }
            
            # 레벨별 추가 정보
            if result['level'] == 'segment':
                result.update({
                    'start_time': doc.metadata.get('start_time'),
                    'end_time': doc.metadata.get('end_time'),
                    'duration': doc.metadata.get('duration'),
                    'dominant_objects': doc.metadata.get('dominant_objects', [])
                })
            elif result['level'] == 'semantic':
                result.update({
                    'semantic_keyword': doc.metadata.get('semantic_keyword'),
                    'event_count': doc.metadata.get('event_count', 0)
                })
            
            results.append(result)
        
        return results
    
    def _deduplicate_and_rank_results(self, results: List[Dict]) -> List[Dict]:
        """중복 제거 및 순위 매기기"""
        if not results:
            return []
        
        # 내용 기반 중복 제거
        seen_content = set()
        unique_results = []
        
        for result in results:
            content_hash = hashlib.md5(result['content'].encode()).hexdigest()
            
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_results.append(result)
        
        # 종합 점수 기반 정렬
        def calculate_ranking_score(result):
            base_score = result.get('quality_score', 0.5)
            
            # 레벨별 가중치
            level_weights = {'frame': 0.8, 'segment': 1.0, 'semantic': 0.9, 'video': 0.6}
            level_weight = level_weights.get(result.get('level', 'frame'), 0.5)
            
            # 품질 등급 보너스
            quality_bonuses = {'high': 0.3, 'medium': 0.1, 'low': 0.0}
            quality_bonus = quality_bonuses.get(result.get('quality_tier', 'medium'), 0.0)
            
            return base_score * level_weight + quality_bonus
        
        unique_results.sort(key=calculate_ranking_score, reverse=True)
        return unique_results
    
    def _add_advanced_temporal_context(self, video_id: str, query: str, results: List[Dict], intent: Dict) -> List[Dict]:
        """고도화된 시간적 컨텍스트 추가"""
        temporal_index = self.temporal_indexes.get(video_id)
        if not temporal_index:
            return results
        
        enhanced_results = []
        for result in results:
            timestamp = result.get('timestamp')
            if timestamp is not None:
                # 시간적 컨텍스트 분석
                temporal_context = self._analyze_temporal_context(temporal_index, timestamp, intent)
                result['advanced_temporal_context'] = temporal_context
            
            enhanced_results.append(result)
        
        return enhanced_results
    
    def _analyze_temporal_context(self, temporal_index: HierarchicalTemporalIndex, 
                                timestamp: float, intent: Dict) -> Dict:
        """시간적 컨텍스트 분석"""
        context = {
            'nearby_events': [],
            'sequence_analysis': {},
            'temporal_patterns': {}
        }
        
        # 주변 이벤트 찾기 (확장된 윈도우)
        window_size = 10.0  # 10초 윈도우
        nearby_events = []
        
        for event in temporal_index.events:
            time_diff = abs(event['timestamp'] - timestamp)
            if time_diff <= window_size:
                nearby_events.append({
                    'timestamp': event['timestamp'],
                    'caption': event['caption'],
                    'objects': event['objects'],
                    'quality_score': event['quality_score'],
                    'time_diff': time_diff
                })
        
        # 시간순 정렬
        nearby_events.sort(key=lambda x: x['timestamp'])
        context['nearby_events'] = nearby_events[:5]
        
        # 시퀀스 분석
        if len(nearby_events) >= 2:
            context['sequence_analysis'] = {
                'sequence_length': len(nearby_events),
                'time_span': max(e['timestamp'] for e in nearby_events) - min(e['timestamp'] for e in nearby_events),
                'activity_progression': self._analyze_activity_progression(nearby_events)
            }
        
        return context
    
    def _analyze_activity_progression(self, events: List[Dict]) -> str:
        """활동 진행 패턴 분석"""
        if len(events) < 2:
            return "단일 이벤트"
        
        # 간단한 패턴 분석
        object_changes = []
        for i in range(1, len(events)):
            prev_objects = set(events[i-1]['objects'])
            curr_objects = set(events[i]['objects'])
            
            if prev_objects != curr_objects:
                object_changes.append("변화")
            else:
                object_changes.append("지속")
        
        if all(change == "지속" for change in object_changes):
            return "안정적 장면"
        elif object_changes.count("변화") > len(object_changes) // 2:
            return "역동적 변화"
        else:
            return "점진적 변화"
    
    def _apply_quality_boost(self, results: List[Dict], intent: Dict) -> List[Dict]:
        """품질 기반 결과 향상"""
        if intent['confidence'] < 0.7:
            return results
        
        boosted_results = []
        for result in results:
            quality_score = result.get('quality_score', 0.5)
            
            # 고품질 결과에 가중치 부여
            if quality_score >= self.config.quality_boost_threshold:
                result['boosted'] = True
                result['boost_factor'] = 1.2
            
            boosted_results.append(result)
        
        # 품질 부스트된 결과 우선 정렬
        boosted_results.sort(key=lambda x: (
            x.get('boosted', False),
            x.get('quality_score', 0.5)
        ), reverse=True)
        
        return boosted_results
    
    def generate_contextual_korean_answer(self, video_id: str, question: str, 
                                        context: Optional[Dict] = None) -> str:
        """상황 인식 한국어 답변 생성"""
        if not self._llm_initialized:
            return "LLM이 초기화되지 않아 답변을 생성할 수 없습니다."
        
        # 지능형 검색 수행
        search_results = self.intelligent_search_video_content(video_id, question, context)
        
        if not search_results:
            return "관련된 비디오 내용을 찾을 수 없습니다."
        
        # 질문 의도 분석
        intent = self.korean_processor.analyze_question_intent_advanced(question)
        
        # 상황별 맞춤 프롬프트 구성
        prompt = self._build_contextual_korean_prompt(question, search_results, intent, context)
        
        try:
            response = self.llm.invoke(prompt)
            answer = response.content
            
            # 응답 품질 평가 및 개선
            enhanced_answer = self._enhance_answer_quality(answer, search_results, intent)
            
            # 응답 캐싱
            cache_key = self.cache_manager.get_cache_key(video_id, question, "enhanced_response")
            cache.set(cache_key, enhanced_answer, timeout=self.config.cache_ttl_response)
            
            return enhanced_answer
            
        except Exception as e:
            print(f"❌ 상황 인식 답변 생성 실패: {e}")
            return "답변 생성 중 오류가 발생했습니다."
    
    def _build_contextual_korean_prompt(self, question: str, search_results: List[Dict], 
                                      intent: Dict, context: Optional[Dict] = None) -> str:
        """상황별 맞춤 한국어 프롬프트 구성"""
        
        # 의도별 전문 시스템 메시지
        system_messages = {
            'temporal': """당신은 비디오의 시간적 흐름과 순서를 정확히 분석하는 전문가입니다. 
                         시간 순서, 이벤트 발생 시점, 지속 시간, 변화 패턴을 정확히 파악하여 답변해주세요.""",
            'person_analysis': """당신은 비디오 속 인물과 보행자를 세밀히 분석하는 전문가입니다. 
                                사람의 외모, 행동, 속성, 위치 변화를 정확히 파악하여 답변해주세요.""",
            'object_detection': """당신은 비디오 속 객체와 사물을 정확히 식별하고 분석하는 전문가입니다. 
                                 객체의 종류, 위치, 상태, 상호작용을 상세히 분석하여 답변해주세요.""",
            'attribute_analysis': """당신은 비디오 속 사람들의 속성과 특징을 분석하는 전문가입니다. 
                                   의복, 액세서리, 외모, 자세 등을 자세히 분석하여 답변해주세요.""",
            'action_recognition': """당신은 비디오 속 행동과 활동을 분석하는 전문가입니다. 
                                   동작의 종류, 진행 과정, 상호작용을 정확히 파악하여 답변해주세요."""
        }
        
        system_msg = system_messages.get(intent['primary_type'], 
                                       "당신은 비디오 내용을 종합적으로 분석하는 전문가입니다.")
        
        # 검색 결과를 품질과 레벨별로 구성
        context_sections = self._organize_context_by_quality_and_level(search_results)
        
        # 고품질 정보 강조
        high_quality_info = context_sections.get('high_quality', [])
        medium_quality_info = context_sections.get('medium_quality', [])
        
        context_text = ""
        
        if high_quality_info:
            context_text += "=== 고품질 분석 결과 ===\n"
            for info in high_quality_info[:3]:  # 상위 3개
                context_text += f"- {info}\n"
            context_text += "\n"
        
        if medium_quality_info:
            context_text += "=== 보조 분석 결과 ===\n"
            for info in medium_quality_info[:2]:  # 상위 2개
                context_text += f"- {info}\n"
            context_text += "\n"
        
        # 시간적 컨텍스트 추가
        temporal_info = self._extract_temporal_information(search_results)
        if temporal_info:
            context_text += "=== 시간적 정보 ===\n"
            context_text += temporal_info + "\n\n"
        
        # 비디오 메타정보
        video_info = ""
        if context:
            video_info = f"""
비디오 정보:
- 파일명: {context.get('filename', 'unknown')}
- 길이: {context.get('duration', 0)}초
- 분석 품질: {context.get('analysis_quality', '중간')}
- 주요 특징: {', '.join(context.get('key_features', []))}
"""
        
        # 의도별 특별 지침
        intent_specific_instructions = self._get_intent_specific_instructions(intent)
        
        # 최종 프롬프트 구성
        prompt = f"""{system_msg}

{video_info}

비디오 분석 결과:
{context_text}

{intent_specific_instructions}

사용자 질문: {question}

답변 요구사항:
1. 제공된 분석 결과만을 바탕으로 정확히 답변하세요
2. 시간 정보(초, 분)와 구체적 위치를 포함하세요
3. 고품질 분석 결과를 우선적으로 활용하세요
4. 확실하지 않은 내용은 추측하지 말고 "분석 결과에서 확인되지 않음"이라고 명시하세요
5. 한국어로 자연스럽고 정확하게 답변하세요
6. 관련 객체나 인물이 있다면 구체적인 특징과 함께 언급하세요

답변:"""
        
        return prompt
    
    def _organize_context_by_quality_and_level(self, search_results: List[Dict]) -> Dict[str, List[str]]:
        """품질과 레벨별 컨텍스트 구성"""
        organized = {
            'high_quality': [],
            'medium_quality': [],
            'low_quality': []
        }
        
        for result in search_results:
            quality_tier = result.get('quality_tier', 'medium')
            content = result['content']
            level = result.get('level', 'frame')
            
            # 레벨 정보 추가
            if level == 'segment':
                start_time = result.get('start_time', 0)
                end_time = result.get('end_time', 0)
                content = f"[{start_time:.1f}s-{end_time:.1f}s 구간] {content}"
            elif level == 'frame':
                timestamp = result.get('timestamp', 0)
                frame_id = result.get('frame_id', 0)
                content = f"[프레임 {frame_id}, {timestamp:.1f}s] {content}"
            elif level == 'semantic':
                keyword = result.get('semantic_keyword', '')
                content = f"[의미: {keyword}] {content}"
            
            # 품질별 분류
            if quality_tier == 'high':
                organized['high_quality'].append(content)
            elif quality_tier == 'medium':
                organized['medium_quality'].append(content)
            else:
                organized['low_quality'].append(content)
        
        return organized
    
    def _extract_temporal_information(self, search_results: List[Dict]) -> str:
        """시간적 정보 추출"""
        temporal_info = []
        
        # 시간순 정렬
        timed_results = [r for r in search_results if r.get('timestamp') is not None]
        timed_results.sort(key=lambda x: x['timestamp'])
        
        if len(timed_results) >= 2:
            time_span = timed_results[-1]['timestamp'] - timed_results[0]['timestamp']
            temporal_info.append(f"분석 시간 범위: {timed_results[0]['timestamp']:.1f}s ~ {timed_results[-1]['timestamp']:.1f}s ({time_span:.1f}초 동안)")
        
        # 고급 시간적 컨텍스트
        for result in search_results[:2]:  # 상위 2개만
            temporal_context = result.get('advanced_temporal_context')
            if temporal_context:
                nearby_events = temporal_context.get('nearby_events', [])
                if nearby_events:
                    sequence_info = f"주변 이벤트: {len(nearby_events)}개 관련 장면"
                    temporal_info.append(sequence_info)
                    break
        
        return '\n'.join(temporal_info) if temporal_info else ""
    
    def _get_intent_specific_instructions(self, intent: Dict) -> str:
        """의도별 특별 지침"""
        instructions = {
            'temporal': """
시간 관련 답변 시 특별 주의사항:
- 정확한 시간(초) 명시
- 이벤트 순서와 지속 시간 포함
- 시간적 변화 패턴 설명""",
            'person_analysis': """
인물 분석 시 특별 주의사항:
- 구체적인 외모와 의복 특징 설명
- 위치 및 행동 변화 추적
- 다른 인물과의 구분점 명시""",
            'object_detection': """
객체 분석 시 특별 주의사항:
- 객체의 정확한 명칭과 특징
- 위치 및 상태 변화
- 다른 객체와의 관계""",
            'attribute_analysis': """
속성 분석 시 특별 주의사항:
- 신뢰도가 높은 속성 정보 우선 사용
- 불확실한 속성은 명시적으로 표시
- 여러 관찰 결과가 있다면 종합적 판단"""
        }
        
        return instructions.get(intent['primary_type'], "")
    
    def _enhance_answer_quality(self, answer: str, search_results: List[Dict], intent: Dict) -> str:
        """답변 품질 향상"""
        # 기본 검증
        if not answer or len(answer.strip()) < 10:
            return "제공된 정보가 불충분하여 정확한 답변을 생성할 수 없습니다."
        
        # 시간 정보 검증 및 보강
        if intent['primary_type'] == 'temporal':
            answer = self._enhance_temporal_answer(answer, search_results)
        
        # 품질 정보 추가
        high_quality_count = sum(1 for r in search_results if r.get('quality_tier') == 'high')
        if high_quality_count > 0:
            answer += f"\n\n(이 답변은 {high_quality_count}개의 고품질 분석 결과를 바탕으로 작성되었습니다.)"
        
        return answer
    
    def _enhance_temporal_answer(self, answer: str, search_results: List[Dict]) -> str:
        """시간 관련 답변 향상"""
        # 시간 정보가 누락된 경우 보강
        import re
        
        time_pattern = r'\d+\.?\d*초'
        if not re.search(time_pattern, answer):
            # 검색 결과에서 시간 정보 추출
            timestamps = [r.get('timestamp') for r in search_results if r.get('timestamp') is not None]
            if timestamps:
                min_time, max_time = min(timestamps), max(timestamps)
                if min_time == max_time:
                    answer += f" (관련 시점: {min_time:.1f}초)"
                else:
                    answer += f" (관련 시간대: {min_time:.1f}초~{max_time:.1f}초)"
        
        return answer
    
    # 기존 호환성 메서드들
    def process_video_analysis_json(self, json_file_path: str, video_id: str) -> bool:
        """기존 호환성 유지"""
        return self.process_video_analysis_json_advanced(json_file_path, video_id)
    
    def search_video_content(self, video_id: str, query: str, top_k: int = 5):
        """기존 호환성 유지"""
        results = self.intelligent_search_video_content(video_id, query)
        return results[:top_k]
    
    def answer_question(self, video_id: str, question: str):
        """기존 호환성 유지"""
        return self.generate_contextual_korean_answer(video_id, question)
    
    def get_database_info(self, video_id: str = None):
        """데이터베이스 정보 조회 (향상됨)"""
        if video_id:
            if video_id in self.video_databases:
                db_info = self.video_databases[video_id]
                temporal_index = self.temporal_indexes.get(video_id)
                quality_index = self.quality_indexes.get(video_id)
                
                return {
                    'video_id': video_id,
                    'document_count': len(db_info['documents']),
                    'document_stats': db_info.get('document_stats', {}),
                    'created_at': db_info['created_at'].isoformat(),
                    'config': {
                        'embedding_model': self.config.embedding_model,
                        'top_k': self.config.top_k,
                        'similarity_threshold': self.config.similarity_threshold
                    },
                    'temporal_index': {
                        'total_events': len(temporal_index.events) if temporal_index else 0,
                        'segments': len(temporal_index.segments) if temporal_index else 0,
                        'semantic_clusters': len(temporal_index.semantic_clusters) if temporal_index else 0
                    } if temporal_index else None,
                    'quality_distribution': {
                        tier: len(frames) for tier, frames in quality_index.items()
                    } if quality_index else None
                }
            else:
                return None
        else:
            return {
                'total_videos': len(self.video_databases),
                'videos': list(self.video_databases.keys()),
                'system_status': {
                    'embeddings_initialized': self._embeddings_initialized,
                    'llm_initialized': self._llm_initialized,
                    'device': self.device
                },
                'config_summary': {
                    'embedding_model': self.config.embedding_model,
                    'korean_analysis': self.config.use_korean_morphology,
                    'quality_aware': True,
                    'hierarchical_search': True
                }
            }

# 전역 시스템 인스턴스
_superior_rag_system = None

def get_enhanced_video_rag_system(config: Optional[AdvancedVideoRAGConfig] = None):
    """최고급 RAG 시스템 인스턴스 반환 (싱글톤)"""
    global _superior_rag_system
    if _superior_rag_system is None:
        _superior_rag_system = SuperiorVideoRAGSystem(config)
    return _superior_rag_system

# 하위 호환성을 위한 래퍼 함수
def get_video_rag_system():
    """기존 호환성 유지"""
    return get_enhanced_video_rag_system()
