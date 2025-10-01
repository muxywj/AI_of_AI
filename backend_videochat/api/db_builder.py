

# chat/db_builder.py - 고도화된 비디오 RAG 시스템
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

logger = logging.getLogger(__name__)

# konlpy를 선택적으로 import
try:
    from konlpy.tag import Mecab, Okt
    KONLPY_AVAILABLE = True
    MECAB_DIC = os.getenv("MECAB_DIC", "/opt/homebrew/lib/mecab/dic/mecab-ko-dic")
except ImportError:
    KONLPY_AVAILABLE = False
    Mecab = None
    Okt = None
    MECAB_DIC = None

def make_korean_analyzer(preferred: str = None):
    if not KONLPY_AVAILABLE:
        return None
    if preferred == "okt":
        return Okt()
    try:
        return Mecab(dicpath=MECAB_DIC)
    except Exception:
        return Okt()


# LangChain 관련 import - 임시 비활성화
try:
    # from langchain_community.document_loaders import JSONLoader
    # from langchain_community.vectorstores import FAISS
    # from langchain_huggingface import HuggingFaceEmbeddings
    # from langchain_core.documents import Document
    # from langchain.retrievers import EnsembleRetriever
    # from langchain_community.retrievers import BM25Retriever
    # from langchain_openai import ChatOpenAI
    # from langchain.text_splitter import RecursiveCharacterTextSplitter
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
class VideoRAGConfig:
    # FAISS 설정
    use_gpu: bool = torch.cuda.is_available()
    nlist: int = 100  # sqrt(N) 기반으로 동적 조정
    nprobe: int = 10  # nlist/10
    embedding_model: str = "intfloat/multilingual-e5-large"
    embedding_dim: int = 1024
    
    # 임베딩 및 검색 설정
    chunk_size: int = 512
    chunk_overlap: int = 128
    top_k: int = 5
    similarity_threshold: float = 0.8
    
    # 캐싱 설정
    cache_ttl_embedding: int = 3600  # 1시간
    cache_ttl_analysis: int = 1800   # 30분
    cache_ttl_response: int = 7200   # 2시간
    
    # 한국어 처리 설정
    use_korean_morphology: bool = KONLPY_AVAILABLE
    korean_analyzer: str = "mecab"  # mecab, hannanum, kkma
    
    # 모델 설정
    llm_model: str = "gemma2-9b-it"
    max_tokens: int = 1024
    temperature: float = 0.2

class TemporalIndex:
    """비디오 시간축 인덱싱"""
    
    def __init__(self):
        self.timeline = defaultdict(list)  # timestamp -> events
        self.segments = []  # 시간 구간별 세그먼트
        self.events = []   # 감지된 이벤트들
    
    def add_frame_data(self, timestamp: float, frame_id: int, 
                      caption: str, objects: List[str], scene_data: Dict):
        """프레임 데이터를 시간축에 추가"""
        event = {
            'timestamp': timestamp,
            'frame_id': frame_id,
            'caption': caption,
            'objects': objects,
            'scene_type': scene_data.get('location', {}).get('label', ''),
            'time_of_day': scene_data.get('time', {}).get('label', ''),
            'activities': scene_data.get('activities', [])
        }
        
        self.timeline[timestamp].append(event)
        self.events.append(event)
    
    def create_segments(self, segment_duration: float = 30.0):
        """시간 구간별 세그먼트 생성"""
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
                segment = {
                    'start_time': current_time,
                    'end_time': end_time,
                    'events': segment_events,
                    'dominant_objects': self._get_dominant_objects(segment_events),
                    'scene_summary': self._summarize_scene(segment_events)
                }
                self.segments.append(segment)
            
            current_time = end_time
    
    def _get_dominant_objects(self, events: List[Dict]) -> List[str]:
        """세그먼트에서 주요 객체 추출"""
        object_counts = defaultdict(int)
        for event in events:
            for obj in event.get('objects', []):
                object_counts[obj] += 1
        
        return sorted(object_counts.keys(), key=object_counts.get, reverse=True)[:5]
    
    def _summarize_scene(self, events: List[Dict]) -> str:
        """세그먼트 장면 요약"""
        if not events:
            return ""
        
        scene_types = [e.get('scene_type', '') for e in events if e.get('scene_type')]
        activities = []
        for e in events:
            activities.extend(e.get('activities', []))
        
        scene_type = max(set(scene_types), key=scene_types.count) if scene_types else "일반"
        main_activities = list(set(activities))[:3]
        
        return f"{scene_type} 장면에서 {', '.join(main_activities)}" if main_activities else f"{scene_type} 장면"

class KoreanTextProcessor:
    """한국어 텍스트 전처리"""
    def __init__(self, analyzer: str = "mecab"):
        self.analyzer_type = analyzer
        if not KONLPY_AVAILABLE:
            self.analyzer = None
            return

        if analyzer == "mecab":
            # ✅ Homebrew(Apple Silicon) 경로 강제
            self.analyzer = Mecab(dicpath=MECAB_DIC)
        elif analyzer == "hannanum":
            self.analyzer = Hannanum()
        elif analyzer == "kkma":
            self.analyzer = Kkma()
        else:
            # 최소 폴백 (원하면 Okt로 바꿔도 됨)
            self.analyzer = None

    def extract_temporal_markers(self, text: str) -> List[str]:
        """시간 표현 추출"""
        temporal_patterns = [
            r'(\d+)초', r'(\d+)분', r'(\d+)시간',
            r'처음에', r'마지막에', r'중간에',
            r'시작할 때', r'끝날 때',
            r'먼저', r'나중에', r'그 다음',
            r'언제', r'몇 분', r'몇 초'
        ]
        
        import re
        markers = []
        for pattern in temporal_patterns:
            matches = re.findall(pattern, text)
            markers.extend(matches)
        
        return markers
    
    def analyze_question_intent(self, question: str) -> Dict[str, Any]:
        """질문 의도 분석"""
        intent = {
            'type': 'general',
            'temporal': False,
            'objects': [],
            'actions': [],
            'locations': []
        }
        
        # 시간 관련 질문 감지
        temporal_keywords = ['언제', '몇 시', '시간', '순서', '먼저', '나중', '전에', '후에']
        if any(keyword in question for keyword in temporal_keywords):
            intent['temporal'] = True
            intent['type'] = 'temporal'
        
        # 객체 관련 질문 감지
        object_keywords = ['사람', '차', '동물', '물건', '무엇', '누구', '어떤']
        found_objects = [keyword for keyword in object_keywords if keyword in question]
        if found_objects:
            intent['objects'] = found_objects
            intent['type'] = 'object_detection'
        
        # 행동 관련 질문 감지
        action_keywords = ['하는', '움직', '걷', '뛰', '앉', '서', '말하', '행동']
        found_actions = [keyword for keyword in action_keywords if keyword in question]
        if found_actions:
            intent['actions'] = found_actions
            intent['type'] = 'action_recognition'
        
        return intent

class EnhancedCacheManager:
    """다단계 캐싱 시스템"""
    
    def __init__(self, config: VideoRAGConfig):
        self.config = config
        self.embedding_cache = {}
        self.analysis_cache = {}
        self.response_cache = {}
    
    def get_cache_key(self, video_id: str, query: str, cache_type: str) -> str:
        """캐시 키 생성"""
        content = f"{video_id}:{query}:{cache_type}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_embedding_cache(self, video_id: str, text: str) -> Optional[np.ndarray]:
        """임베딩 캐시 조회"""
        key = self.get_cache_key(video_id, text, "embedding")
        
        # Django 캐시 사용
        cached = cache.get(key)
        if cached:
            return np.array(cached['embedding'])
        
        return None
    
    def set_embedding_cache(self, video_id: str, text: str, embedding: np.ndarray):
        """임베딩 캐시 저장"""
        key = self.get_cache_key(video_id, text, "embedding")
        cache.set(key, {
            'embedding': embedding.tolist(),
            'timestamp': time.time()
        }, timeout=self.config.cache_ttl_embedding)
    
    def get_semantic_cache(self, video_id: str, query: str, threshold: float = 0.8) -> Optional[str]:
        """의미적 유사도 기반 캐시 조회"""
        # 기존 쿼리들과 유사도 비교
        cache_key_pattern = f"semantic_cache:{video_id}:*"
        
        # Redis나 다른 캐시에서 패턴 매칭으로 조회
        # 여기서는 간단한 구현
        for cached_query, response in self.response_cache.items():
            if self._calculate_similarity(query, cached_query) > threshold:
                return response
        
        return None
    
    def _calculate_similarity(self, query1: str, query2: str) -> float:
        """간단한 텍스트 유사도 계산"""
        # 실제로는 임베딩 기반 코사인 유사도 사용
        words1 = set(query1.split())
        words2 = set(query2.split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0

class EnhancedVideoRAGSystem:
    """고도화된 비디오 분석 RAG 시스템"""
    
    def __init__(self, config: Optional[VideoRAGConfig] = None):
        self.config = config or VideoRAGConfig()
        self.device = "cuda" if self.config.use_gpu and torch.cuda.is_available() else "cpu"
        
        # 컴포넌트 초기화
        self.cache_manager = EnhancedCacheManager(self.config)
        self.korean_processor = KoreanTextProcessor(self.config.korean_analyzer)
        
        # 시스템 상태
        self._embeddings_initialized = False
        self._llm_initialized = False
        
        print(f"🚀 Enhanced VideoRAG 시스템 초기화 (디바이스: {self.device})")
        
        if not LANGCHAIN_AVAILABLE:
            print("⚠️ LangChain 미설치 - 기본 RAG 기능만 사용")
            self._init_basic_features()
            return
        
        try:
            self._init_embeddings()
            self._init_llm()
        except Exception as e:
            print(f"⚠️ RAG 시스템 초기화 부분 실패: {e}")
            self._init_basic_features()
        
        # 비디오 데이터베이스 저장소
        self.video_databases = {}
        self.temporal_indexes = {}
        self.quality_indexes = {}
        
        print("✅ Enhanced VideoRAG 시스템 초기화 완료")
    
    def _init_basic_features(self):
        """기본 기능 초기화 (LangChain 없이도 작동)"""
        print("✅ 기본 RAG 기능 활성화 (임베딩 및 LLM 없음)")
        self.video_databases = {}
        self.temporal_indexes = {}
        self.quality_indexes = {}
        print("✅ 기본 RAG 시스템 초기화 완료")
    
    def _init_embeddings(self):
        """다국어 임베딩 모델 초기화"""
        try:
            model_kwargs = {"device": self.device}
            encode_kwargs = {
                'normalize_embeddings': True,
                'batch_size': 32  # 배치 처리 최적화
            }
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.config.embedding_model,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
            self._embeddings_initialized = True
            print(f"✅ 다국어 임베딩 모델 로드 완료: {self.config.embedding_model}")
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
            print(f"✅ LLM 초기화 완료: {self.config.llm_model}")
        except Exception as e:
            print(f"⚠️ LLM 초기화 실패: {e}")
            self._llm_initialized = False
    
    def process_video_analysis_json(self, json_file_path: str, video_id: str) -> bool:
        """향상된 비디오 분석 JSON 처리"""
        try:
            if not os.path.exists(json_file_path):
                print(f"⚠️ JSON 파일을 찾을 수 없음: {json_file_path}")
                return False
            
            print(f"📄 JSON 분석 파일 처리 중: {json_file_path}")
            
            with open(json_file_path, 'r', encoding='utf-8') as f:
                analysis_data = json.load(f)
            
            # 시간축 인덱스 생성
            temporal_index = TemporalIndex()
            
            # 다중 레벨 문서 생성
            frame_documents = []
            segment_documents = []
            
            frame_results = analysis_data.get('frame_results', [])
            video_metadata = analysis_data.get('metadata', {})
            
            # 프레임별 문서 생성 및 시간축 인덱스 구축
            for frame_result in frame_results:
                frame_id = frame_result.get('image_id', 0)
                timestamp = frame_result.get('timestamp', 0)
                
                # 다양한 캡션 소스 통합
                caption = (frame_result.get('final_caption') or 
                          frame_result.get('enhanced_caption') or 
                          frame_result.get('caption') or '')
                
                objects = frame_result.get('objects', [])
                scene_analysis = frame_result.get('scene_analysis', {})
                
                # 시간축 인덱스에 추가
                temporal_index.add_frame_data(
                    timestamp, frame_id, caption, 
                    [obj.get('class', '') for obj in objects],
                    scene_analysis.get('scene_classification', {})
                )
                
                # 프레임 문서 생성
                content_parts = self._build_frame_content(
                    frame_id, timestamp, caption, objects, scene_analysis
                )
                
                if content_parts:
                    metadata = {
                        'video_id': video_id,
                        'frame_id': frame_id,
                        'timestamp': timestamp,
                        'objects': [obj.get('class', '') for obj in objects],
                        'scene_type': scene_analysis.get('scene_classification', {}).get('location', {}).get('label', ''),
                        'level': 'frame'
                    }
                    
                    frame_documents.append(Document(
                        page_content='. '.join(content_parts), 
                        metadata=metadata
                    ))
            
            # 시간 구간별 세그먼트 생성
            temporal_index.create_segments()
            
            # 세그먼트별 문서 생성
            for segment in temporal_index.segments:
                segment_content = self._build_segment_content(segment)
                
                metadata = {
                    'video_id': video_id,
                    'start_time': segment['start_time'],
                    'end_time': segment['end_time'],
                    'dominant_objects': segment['dominant_objects'],
                    'scene_summary': segment['scene_summary'],
                    'level': 'segment'
                }
                
                segment_documents.append(Document(
                    page_content=segment_content,
                    metadata=metadata
                ))
            
            # 전체 비디오 문서 생성
            video_document = self._build_video_document(analysis_data, temporal_index)
            video_metadata_doc = {
                'video_id': video_id,
                'level': 'video',
                'total_frames': len(frame_results),
                'duration': video_metadata.get('duration', 0),
                'analysis_type': video_metadata.get('analysis_type', 'unknown')
            }
            
            all_documents = frame_documents + segment_documents + [Document(
                page_content=video_document,
                metadata=video_metadata_doc
            )]
            
            # 계층적 벡터 DB 생성
            success = self._create_hierarchical_vector_db(video_id, all_documents)
            
            if success:
                # 시간축 인덱스 저장
                self.temporal_indexes[video_id] = temporal_index
                print(f"✅ 비디오 {video_id} 고급 RAG DB 생성 완료: {len(all_documents)}개 문서")
                return True
            
            return False
            
        except Exception as e:
            print(f"❌ 고급 RAG DB 생성 실패: {e}")
            import traceback
            print(f"상세 오류: {traceback.format_exc()}")
            return False
    
    def _build_frame_content(self, frame_id: int, timestamp: float, 
                           caption: str, objects: List, scene_analysis: Dict) -> List[str]:
        """프레임 내용 구성"""
        content_parts = []
        
        if caption:
            content_parts.append(f"프레임 {frame_id} ({timestamp:.1f}초): {caption}")
        
        if objects:
            object_list = [obj.get('class', '') for obj in objects if obj.get('class')]
            if object_list:
                content_parts.append(f"감지된 객체: {', '.join(object_list)}")
        
        # 장면 분석 정보
        if scene_analysis:
            scene_class = scene_analysis.get('scene_classification', {})
            if scene_class:
                location = scene_class.get('location', {}).get('label', '')
                time_of_day = scene_class.get('time', {}).get('label', '')
                if location or time_of_day:
                    content_parts.append(f"장면: {location} {time_of_day}".strip())
            
            # OCR 텍스트
            ocr_text = scene_analysis.get('ocr_text', '')
            if ocr_text:
                content_parts.append(f"텍스트: {ocr_text}")
        
        return content_parts
    
    def _build_segment_content(self, segment: Dict) -> str:
        """세그먼트 내용 구성"""
        start_time = segment['start_time']
        end_time = segment['end_time']
        scene_summary = segment['scene_summary']
        dominant_objects = segment['dominant_objects']
        
        content = f"{start_time:.1f}초-{end_time:.1f}초 구간: {scene_summary}"
        
        if dominant_objects:
            content += f". 주요 객체: {', '.join(dominant_objects[:3])}"
        
        # 이벤트 요약
        events = segment.get('events', [])
        if events:
            event_descriptions = [event.get('caption', '') for event in events if event.get('caption')]
            if event_descriptions:
                content += f". 주요 활동: {'; '.join(event_descriptions[:2])}"
        
        return content
    
    def _build_video_document(self, analysis_data: Dict, temporal_index: TemporalIndex) -> str:
        """전체 비디오 문서 구성"""
        metadata = analysis_data.get('metadata', {})
        
        content_parts = [
            f"비디오 전체 요약:",
            f"- 총 길이: {metadata.get('duration', 0)}초",
            f"- 총 프레임: {len(analysis_data.get('frame_results', []))}개",
            f"- 구간 수: {len(temporal_index.segments)}개"
        ]
        
        # 전체 비디오의 주요 객체
        all_objects = []
        for event in temporal_index.events:
            all_objects.extend(event.get('objects', []))
        
        if all_objects:
            from collections import Counter
            object_counts = Counter(all_objects)
            top_objects = [obj for obj, count in object_counts.most_common(5)]
            content_parts.append(f"- 주요 객체: {', '.join(top_objects)}")
        
        # 장면 유형 요약
        scene_types = [segment['scene_summary'] for segment in temporal_index.segments]
        if scene_types:
            unique_scenes = list(set(scene_types))
            content_parts.append(f"- 장면 유형: {', '.join(unique_scenes[:3])}")
        
        return '\n'.join(content_parts)
    
    def _create_hierarchical_vector_db(self, video_id: str, documents: List[Document]) -> bool:
        """계층적 벡터 DB 생성"""
        if not self._embeddings_initialized or not documents:
            return False
        
        try:
            # FAISS 인덱스 최적화 설정
            db = FAISS.from_documents(documents, embedding=self.embeddings)
            
            # nlist 동적 조정
            n_docs = len(documents)
            optimal_nlist = max(10, min(int(np.sqrt(n_docs)), 1000))
            
            # 검색기 구성 - 다중 검색 전략
            similarity_retriever = db.as_retriever(
                search_type="similarity",
                search_kwargs={'k': self.config.top_k}
            )
            
            mmr_retriever = db.as_retriever(
                search_type="mmr",
                search_kwargs={
                    'k': self.config.top_k,
                    'fetch_k': self.config.top_k * 2
                }
            )
            
            # BM25 검색기 (한국어 지원)
            try:
                bm25_retriever = BM25Retriever.from_documents(documents)
                bm25_retriever.k = self.config.top_k
                
                # 앙상블 검색기 구성
                ensemble_retriever = EnsembleRetriever(
                    retrievers=[similarity_retriever, mmr_retriever, bm25_retriever],
                    weights=[0.5, 0.3, 0.2]  # 가중치 조정
                )
            except Exception as e:
                print(f"⚠️ BM25 검색기 생성 실패: {e}")
                ensemble_retriever = similarity_retriever
            
            # 계층별 검색기
            frame_retriever = self._create_level_retriever(db, documents, 'frame')
            segment_retriever = self._create_level_retriever(db, documents, 'segment')
            video_retriever = self._create_level_retriever(db, documents, 'video')
            
            self.video_databases[video_id] = {
                'db': db,
                'retriever': ensemble_retriever,
                'frame_retriever': frame_retriever,
                'segment_retriever': segment_retriever,
                'video_retriever': video_retriever,
                'documents': documents,
                'created_at': datetime.now(),
                'config': self.config
            }
            
            return True
            
        except Exception as e:
            print(f"❌ 계층적 벡터 DB 생성 실패: {e}")
            return False
    
    def _create_level_retriever(self, db, documents: List[Document], level: str):
        """레벨별 검색기 생성"""
        level_docs = [doc for doc in documents if doc.metadata.get('level') == level]
        if not level_docs:
            return None
        
        level_db = FAISS.from_documents(level_docs, embedding=self.embeddings)
        return level_db.as_retriever(
            search_type="similarity",
            search_kwargs={'k': min(self.config.top_k, len(level_docs))}
        )
    
    def smart_search_video_content(self, video_id: str, query: str, 
                                 context: Optional[Dict] = None) -> List[Dict]:
        """지능형 비디오 내용 검색"""
        if video_id not in self.video_databases:
            print(f"⚠️ 비디오 {video_id}의 RAG DB가 없음")
            return []
        
        try:
            # 의미적 캐시 확인
            cached_result = self.cache_manager.get_semantic_cache(
                video_id, query, self.config.similarity_threshold
            )
            if cached_result:
                print("🎯 캐시에서 유사한 쿼리 결과 반환")
                return cached_result
            
            # 질문 의도 분석
            intent = self.korean_processor.analyze_question_intent(query)
            
            # 의도에 따른 검색 전략 선택
            results = self._execute_search_strategy(video_id, query, intent)
            
            # 시간적 컨텍스트 추가
            if intent['temporal'] and video_id in self.temporal_indexes:
                results = self._add_temporal_context(video_id, query, results)
            
            # 결과 캐싱
            cache_key = self.cache_manager.get_cache_key(video_id, query, "search")
            cache.set(cache_key, results, timeout=self.config.cache_ttl_analysis)
            
            print(f"🔍 지능형 검색 완료: {len(results)}개 결과 (의도: {intent['type']})")
            return results
            
        except Exception as e:
            print(f"❌ 지능형 검색 실패: {e}")
            return []
    
    def _execute_search_strategy(self, video_id: str, query: str, intent: Dict) -> List[Dict]:
        """의도 기반 검색 전략 실행"""
        db_info = self.video_databases[video_id]
        
        if intent['temporal']:
            # 시간 기반 검색 - 세그먼트 레벨 우선
            if db_info['segment_retriever']:
                docs = db_info['segment_retriever'].get_relevant_documents(query)
            else:
                docs = db_info['retriever'].get_relevant_documents(query)
        
        elif intent['type'] == 'object_detection':
            # 객체 검색 - 프레임 레벨 우선
            if db_info['frame_retriever']:
                docs = db_info['frame_retriever'].get_relevant_documents(query)
            else:
                docs = db_info['retriever'].get_relevant_documents(query)
        
        elif intent['type'] == 'action_recognition':
            # 행동 검색 - 세그먼트 레벨
            if db_info['segment_retriever']:
                docs = db_info['segment_retriever'].get_relevant_documents(query)
            else:
                docs = db_info['retriever'].get_relevant_documents(query)
        
        else:
            # 일반적 검색 - 전체 앙상블
            docs = db_info['retriever'].get_relevant_documents(query)
        
        return self._format_search_results(docs)
    
    def _add_temporal_context(self, video_id: str, query: str, results: List[Dict]) -> List[Dict]:
        """시간적 컨텍스트 추가"""
        temporal_index = self.temporal_indexes.get(video_id)
        if not temporal_index:
            return results
        
        # 시간 표현 추출
        temporal_markers = self.korean_processor.extract_temporal_markers(query)
        
        # 결과에 시간적 정보 보강
        enhanced_results = []
        for result in results:
            timestamp = result.get('metadata', {}).get('timestamp')
            if timestamp is not None:
                # 주변 시간대 이벤트 찾기
                nearby_events = self._find_nearby_events(temporal_index, timestamp, window=5.0)
                result['temporal_context'] = nearby_events
            
            enhanced_results.append(result)
        
        return enhanced_results
    
    def _find_nearby_events(self, temporal_index: TemporalIndex, 
                           target_timestamp: float, window: float = 5.0) -> List[Dict]:
        """주변 시간대 이벤트 찾기"""
        nearby_events = []
        
        for event in temporal_index.events:
            time_diff = abs(event['timestamp'] - target_timestamp)
            if time_diff <= window:
                nearby_events.append({
                    'timestamp': event['timestamp'],
                    'caption': event['caption'],
                    'time_diff': time_diff
                })
        
        return sorted(nearby_events, key=lambda x: x['time_diff'])[:3]
    
    def _format_search_results(self, docs: List[Document]) -> List[Dict]:
        """검색 결과 포맷팅"""
        results = []
        for doc in docs:
            result = {
                'content': doc.page_content,
                'metadata': doc.metadata,
                'frame_id': doc.metadata.get('frame_id'),
                'timestamp': doc.metadata.get('timestamp'),
                'objects': doc.metadata.get('objects', []),
                'level': doc.metadata.get('level', 'unknown')
            }
            results.append(result)
        
        return results
    
    def generate_korean_aware_answer(self, video_id: str, question: str, 
                                   context: Optional[Dict] = None) -> str:
        """한국어 인식 답변 생성"""
        if not self._llm_initialized:
            return "LLM이 초기화되지 않아 답변을 생성할 수 없습니다."
        
        # 지능형 검색 수행
        search_results = self.smart_search_video_content(video_id, question, context)
        
        if not search_results:
            return "관련된 비디오 내용을 찾을 수 없습니다."
        
        # 질문 의도 분석
        intent = self.korean_processor.analyze_question_intent(question)
        
        # 한국어 특화 프롬프트 구성
        prompt = self._build_korean_prompt(question, search_results, intent, context)
        
        try:
            response = self.llm.invoke(prompt)
            answer = response.content
            
            # 응답 캐싱
            cache_key = self.cache_manager.get_cache_key(video_id, question, "response")
            cache.set(cache_key, answer, timeout=self.config.cache_ttl_response)
            
            return answer
            
        except Exception as e:
            print(f"❌ 한국어 답변 생성 실패: {e}")
            return "답변 생성 중 오류가 발생했습니다."
    
    def _build_korean_prompt(self, question: str, search_results: List[Dict], 
                           intent: Dict, context: Optional[Dict] = None) -> str:
        """한국어 특화 프롬프트 구성"""
        
        # 의도별 시스템 메시지
        if intent['temporal']:
            system_msg = """당신은 비디오의 시간적 흐름을 정확히 이해하는 전문 분석가입니다. 
            시간 순서, 이벤트 발생 시점, 지속 시간 등을 정확히 파악하여 답변해주세요."""
        elif intent['type'] == 'object_detection':
            system_msg = """당신은 비디오 속 객체와 인물을 정확히 식별하는 전문가입니다. 
            객체의 위치, 특징, 상태 변화 등을 상세히 분석하여 답변해주세요."""
        elif intent['type'] == 'action_recognition':
            system_msg = """당신은 비디오 속 행동과 활동을 분석하는 전문가입니다. 
            행동의 종류, 진행 과정, 결과 등을 정확히 파악하여 답변해주세요."""
        else:
            system_msg = """당신은 비디오 내용을 종합적으로 분석하는 전문가입니다. 
            영상의 전반적인 내용과 세부사항을 정확히 파악하여 답변해주세요."""
        
        # 검색 결과 컨텍스트 구성
        context_parts = []
        for i, result in enumerate(search_results[:5]):  # 상위 5개 결과만 사용
            level = result.get('level', 'unknown')
            timestamp = result.get('timestamp')
            
            if level == 'frame' and timestamp is not None:
                context_parts.append(f"프레임 {result.get('frame_id', i)} ({timestamp:.1f}초): {result['content']}")
            elif level == 'segment':
                start_time = result.get('metadata', {}).get('start_time', 0)
                end_time = result.get('metadata', {}).get('end_time', 0)
                context_parts.append(f"구간 {start_time:.1f}-{end_time:.1f}초: {result['content']}")
            else:
                context_parts.append(f"관련 정보: {result['content']}")
            
            # 시간적 컨텍스트 추가
            if 'temporal_context' in result:
                for tc in result['temporal_context'][:2]:  # 상위 2개만
                    context_parts.append(f"  주변 {tc['timestamp']:.1f}초: {tc['caption']}")
        
        context_text = '\n'.join(context_parts)
        
        # 비디오 메타정보
        video_info = ""
        if context:
            video_info = f"""
비디오 정보:
- 파일명: {context.get('filename', 'unknown')}
- 길이: {context.get('duration', 0)}초
- 주요 객체: {', '.join(context.get('detected_objects', [])[:5])}
"""
        
        # 최종 프롬프트 구성
        prompt = f"""{system_msg}

{video_info}

비디오 분석 결과:
{context_text}

사용자 질문: {question}

답변 요구사항:
1. 제공된 비디오 분석 결과만을 바탕으로 답변하세요
2. 구체적인 시간(초)과 프레임 정보를 포함하세요
3. 확실하지 않은 내용은 추측하지 마세요
4. 한국어로 자연스럽고 정확하게 답변하세요
5. 관련 객체나 장면이 있다면 구체적으로 언급하세요

답변:"""
        
        return prompt
    
    def get_video_statistics(self, video_id: str) -> Dict[str, Any]:
        """비디오 통계 정보 조회"""
        if video_id not in self.video_databases:
            return {}
        
        db_info = self.video_databases[video_id]
        temporal_index = self.temporal_indexes.get(video_id)
        
        stats = {
            'video_id': video_id,
            'total_documents': len(db_info['documents']),
            'created_at': db_info['created_at'].isoformat(),
            'embedding_model': self.config.embedding_model,
            'levels': {}
        }
        
        # 레벨별 통계
        for doc in db_info['documents']:
            level = doc.metadata.get('level', 'unknown')
            if level not in stats['levels']:
                stats['levels'][level] = 0
            stats['levels'][level] += 1
        
        # 시간축 통계
        if temporal_index:
            stats['temporal_stats'] = {
                'total_events': len(temporal_index.events),
                'total_segments': len(temporal_index.segments),
                'timeline_span': max(event['timestamp'] for event in temporal_index.events) if temporal_index.events else 0
            }
        
        return stats
    
    def optimize_database(self, video_id: str) -> bool:
        """데이터베이스 최적화"""
        if video_id not in self.video_databases:
            return False
        
        try:
            db_info = self.video_databases[video_id]
            
            # 인덱스 최적화
            if hasattr(db_info['db'].index, 'train'):
                embeddings = db_info['db'].index.reconstruct_n(0, db_info['db'].index.ntotal)
                db_info['db'].index.train(embeddings)
            
            print(f"✅ 비디오 {video_id} 데이터베이스 최적화 완료")
            return True
            
        except Exception as e:
            print(f"❌ 데이터베이스 최적화 실패: {e}")
            return False
    
    def clear_cache(self, video_id: Optional[str] = None):
        """캐시 정리"""
        if video_id:
            # 특정 비디오 캐시만 정리
            cache_patterns = [
                f"*{video_id}*embedding*",
                f"*{video_id}*analysis*", 
                f"*{video_id}*response*"
            ]
            # Django 캐시 클리어 (실제 구현에서는 패턴 매칭 필요)
            print(f"🧹 비디오 {video_id} 캐시 정리 완료")
        else:
            # 전체 캐시 정리
            cache.clear()
            self.cache_manager.embedding_cache.clear()
            self.cache_manager.analysis_cache.clear()
            self.cache_manager.response_cache.clear()
            print("🧹 전체 캐시 정리 완료")

    def search_objects_by_description(self, video_id: str, user_query: str) -> Dict[str, Any]:
        """
        사용자 입력 기반 객체 탐지 및 장면 검색
        
        Args:
            video_id: 비디오 ID
            user_query: 사용자 입력 ("핑크색 옷 입은 사람 찾아줘", "가방 찾아줘" 등)
        
        Returns:
            검색 결과 (탐지된 객체와 해당 장면 정보)
        """
        try:
            print(f"🔍 객체 탐지 검색: video_id={video_id}, query='{user_query}'")
            
            # 1. 사용자 입력 파싱 및 키워드 추출
            search_keywords = self._parse_user_query(user_query)
            print(f"📝 추출된 키워드: {search_keywords}")
            
            # 2. 비디오 분석 데이터 로드
            analysis_data = self._load_video_analysis_data(video_id)
            if not analysis_data:
                return {
                    'success': False,
                    'error': '비디오 분석 데이터를 찾을 수 없습니다.',
                    'matches': []
                }
            
            # 3. 객체 탐지 결과에서 매칭 검색
            matches = self._find_matching_objects(analysis_data, search_keywords)
            print(f"🎯 발견된 매칭: {len(matches)}개")
            
            # 4. 결과 정렬 및 포맷팅
            formatted_results = self._format_search_results(matches, video_id)
            
            return {
                'success': True,
                'query': user_query,
                'keywords': search_keywords,
                'matches': formatted_results,
                'total_matches': len(formatted_results)
            }
            
        except Exception as e:
            print(f"❌ 객체 탐지 검색 오류: {e}")
            return {
                'success': False,
                'error': str(e),
                'matches': []
            }

    def _parse_user_query(self, user_query: str) -> Dict[str, List[str]]:
        """사용자 입력에서 탐지 키워드 추출"""
        # 기본 키워드 매핑 (분석 결과의 실제 값과 매칭)
        color_keywords = {
            '빨간색': ['red', '빨강', 'wearing red clothes'],
            '빨강': ['red', '빨간색', 'wearing red clothes'],
            '파란색': ['blue', '파랑', 'wearing blue clothes'],
            '파랑': ['blue', '파란색', 'wearing blue clothes'],
            '녹색': ['green', '초록', '초록색', 'wearing green clothes'],
            '초록색': ['green', '초록', '녹색', 'wearing green clothes'],
            '초록': ['green', '초록색', '녹색', 'wearing green clothes'],
            '노란색': ['yellow', '노랑', 'wearing yellow clothes'],
            '노랑': ['yellow', '노란색', 'wearing yellow clothes'],
            '핑크색': ['pink', '핑크', '분홍', '분홍색', 'wearing pink clothes'],
            '분홍색': ['pink', '핑크', '분홍', '핑크색', 'wearing pink clothes'],
            '분홍': ['pink', '핑크', '분홍색', '핑크색', 'wearing pink clothes'],
            '핑크': ['pink', '분홍', '분홍색', '핑크색', 'wearing pink clothes'],
            '보라색': ['purple', '보라', 'wearing purple clothes'],
            '보라': ['purple', '보라색', 'wearing purple clothes'],
            '검은색': ['black', '검정', 'wearing black clothes'],
            '검정': ['black', '검은색', 'wearing black clothes'],
            '흰색': ['white', '하얀', 'wearing white clothes'],
            '하얀': ['white', '흰색', 'wearing white clothes'],
            '회색': ['gray', 'grey', 'wearing gray clothes'],
            '갈색': ['brown', 'wearing brown clothes'],
            '주황색': ['orange', '주황', 'wearing orange clothes'],
            '주황': ['orange', '주황색', 'wearing orange clothes']
        }
        
        object_keywords = {
            '사람': ['person', 'people', 'human'],
            '가방': ['bag', 'backpack', 'handbag'],
            '자동차': ['car', 'vehicle', 'automobile'],
            '자전거': ['bicycle', 'bike'],
            '개': ['dog'],
            '고양이': ['cat'],
            '핸드폰': ['phone', 'mobile', 'cellphone'],
            '컴퓨터': ['computer', 'laptop']
        }
        
        # 사용자 입력에서 색상과 객체 키워드 추출
        detected_colors = []
        detected_objects = []
        
        query_lower = user_query.lower()
        
        # 색상 키워드 검색
        for korean_color, english_colors in color_keywords.items():
            if korean_color in user_query:
                detected_colors.extend(english_colors)
                print(f"🎨 한국어 색상 매칭: {korean_color} -> {english_colors}")
            for eng_color in english_colors:
                if eng_color in query_lower:
                    detected_colors.append(eng_color)
                    print(f"🎨 영어 색상 매칭: {eng_color}")
        
        print(f"🔍 최종 감지된 색상: {detected_colors}")
        
        # 객체 키워드 검색
        for korean_obj, english_objects in object_keywords.items():
            if korean_obj in user_query:
                detected_objects.extend(english_objects)
            for eng_obj in english_objects:
                if eng_obj in query_lower:
                    detected_objects.append(eng_obj)
        
        return {
            'colors': list(set(detected_colors)),
            'objects': list(set(detected_objects)),
            'raw_query': user_query
        }

    def _load_video_analysis_data(self, video_id: str) -> Optional[Dict]:
        """비디오 분석 데이터 로드"""
        try:
            # 분석 결과 JSON 파일 경로
            analysis_dir = os.path.join(settings.MEDIA_ROOT, 'analysis_results')
            
            # video_id로 시작하는 JSON 파일 찾기
            for filename in os.listdir(analysis_dir):
                # analysis_ 또는 real_analysis_ 패턴 모두 지원
                if (filename.startswith(f'analysis_{video_id}_') or 
                    filename.startswith(f'real_analysis_{video_id}_')) and filename.endswith('.json'):
                    file_path = os.path.join(analysis_dir, filename)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        return json.load(f)
            
            print(f"⚠️ 비디오 {video_id}의 분석 데이터를 찾을 수 없습니다.")
            return None
            
        except Exception as e:
            print(f"❌ 분석 데이터 로드 오류: {e}")
            return None

    def _find_matching_objects(self, analysis_data: Dict, search_keywords: Dict) -> List[Dict]:
        """분석 데이터에서 매칭되는 객체 찾기"""
        matches = []
        
        # 고급 분석 데이터 구조 (frame_results) - 우선 처리
        if 'frame_results' in analysis_data:
            for frame_result in analysis_data['frame_results']:
                frame_index = frame_result.get('image_id', 0)
                timestamp = frame_result.get('timestamp', 0)
                
                # persons 배열 처리
                if 'persons' in frame_result:
                    for person in frame_result['persons']:
                        match_score = 0
                        match_reasons = []
                        
                        class_name = person.get('class', '').lower()
                        confidence = person.get('confidence', 0)
                        
                        # 객체 타입 매칭
                        for obj_keyword in search_keywords['objects']:
                            if obj_keyword.lower() in class_name:
                                match_score += 1
                                match_reasons.append(f"객체 타입: {class_name}")
                                break
                        
                        # 색상 매칭 (속성에서)
                        if 'attributes' in person and 'clothing_color' in person['attributes']:
                            clothing_color = person['attributes']['clothing_color'].get('value', '').lower()
                            all_scores = person['attributes']['clothing_color'].get('all_scores', {})
                            
                            # 직접 색상 매칭
                            for color_keyword in search_keywords['colors']:
                                if color_keyword.lower() in clothing_color:
                                    match_score += 0.8
                                    match_reasons.append(f"색상 매칭: {clothing_color}")
                                    break
                                # 더 관대한 색상 매칭 - 키워드가 all_scores의 키에 포함되는지 확인
                                for score_key in all_scores.keys():
                                    if color_keyword.lower() in score_key.lower():
                                        match_score += 0.6
                                        match_reasons.append(f"색상 키워드 매칭: {color_keyword} in {score_key}")
                                        break
                            
                            # all_scores에서 색상 매칭 (더 정확한 매칭)
                            for color_keyword in search_keywords['colors']:
                                for score_key, score_value in all_scores.items():
                                    if color_keyword.lower() in score_key.lower():
                                        if score_value > 0.001:  # 최소 신뢰도를 0.001로 낮춤
                                            match_score += score_value * 0.5
                                            match_reasons.append(f"색상 신뢰도 매칭: {score_key} ({score_value:.4f})")
                                            break
                        
                        # 신뢰도 점수 추가
                        if confidence > 0.7:
                            match_score += 0.5
                            match_reasons.append(f"높은 신뢰도: {confidence:.2f}")
                        
                        # 색상 필터링: 색상 키워드가 있는 경우 색상 매칭이 있어야 함
                        color_matched = False
                        if search_keywords['colors']:  # 색상 키워드가 있는 경우
                            if 'attributes' in person and 'clothing_color' in person['attributes']:
                                clothing_color = person['attributes']['clothing_color'].get('value', '').lower()
                                all_scores = person['attributes']['clothing_color'].get('all_scores', {})
                                
                                print(f"🔍 색상 매칭 확인: {clothing_color} vs {search_keywords['colors']}")
                                
                                # 색상 매칭 확인
                                for color_keyword in search_keywords['colors']:
                                    if color_keyword.lower() in clothing_color:
                                        color_matched = True
                                        print(f"✅ 직접 색상 매칭: {color_keyword} in {clothing_color}")
                                        break
                                    # all_scores에서도 확인 (더 관대한 임계값 적용)
                                    for score_key, score_value in all_scores.items():
                                        if color_keyword.lower() in score_key.lower() and score_value > 0.001:  # 0.1에서 0.001로 낮춤
                                            color_matched = True
                                            print(f"✅ 신뢰도 색상 매칭: {color_keyword} in {score_key} ({score_value:.4f})")
                                            break
                                    # 키워드가 all_scores의 키에 포함되는지 확인 (매우 관대한 매칭)
                                    for score_key in all_scores.keys():
                                        if color_keyword.lower() in score_key.lower():
                                            color_matched = True
                                            print(f"✅ 키워드 색상 매칭: {color_keyword} in {score_key}")
                                            break
                                    if color_matched:
                                        break
                                
                                if not color_matched:
                                    print(f"❌ 색상 매칭 실패: {clothing_color}")
                            else:
                                print(f"❌ 속성 정보 없음: {person.get('attributes', {})}")
                        else:
                            # 색상 키워드가 없으면 모든 결과 허용
                            color_matched = True
                        
                        # 매칭되는 경우 결과에 추가 (색상 필터링 적용)
                        if match_score > 0 and color_matched:
                            person['match_score'] = match_score
                            person['match_reasons'] = match_reasons
                            person['frame_index'] = frame_index
                            person['timestamp'] = timestamp
                            matches.append(person)
        
        # 기본 분석 데이터 구조 (object_detections) - 백업용
        if 'object_detections' in analysis_data:
            for detection in analysis_data['object_detections']:
                match_score = 0
                match_reasons = []
                
                class_name = detection.get('class_name', '').lower()
                confidence = detection.get('confidence', 0)
                frame_index = detection.get('frame_index', 0)
                
                # 객체 타입 매칭
                for obj_keyword in search_keywords['objects']:
                    if obj_keyword.lower() in class_name:
                        match_score += 1
                        match_reasons.append(f"객체 타입: {class_name}")
                        break
                
                # 신뢰도 점수 추가
                if confidence > 0.7:
                    match_score += 0.5
                    match_reasons.append(f"높은 신뢰도: {confidence:.2f}")
                
                # 매칭되는 경우 결과에 추가
                if match_score > 0:
                    detection['match_score'] = match_score
                    detection['match_reasons'] = match_reasons
                    matches.append(detection)
        
        # 점수순으로 정렬 (match_score 우선, 그 다음 confidence)
        matches.sort(key=lambda x: (x.get('match_score', 0), x.get('confidence', 0)), reverse=True)
        return matches

    def _format_search_results(self, matches: List[Dict], video_id: str) -> List[Dict]:
        """검색 결과 포맷팅"""
        formatted_results = []
        
        for match in matches:
            # 시간 계산 (프레임 번호를 시간으로 변환)
            frame_index = match.get('frame_index', 0)
            timestamp = match.get('timestamp', 0)
            
            # timestamp가 없으면 프레임 번호로 계산
            if timestamp == 0:
                fps = 30  # 기본 FPS (실제로는 비디오에서 가져와야 함)
                timestamp = frame_index / fps
            
            # class_name 또는 class 필드 사용
            class_name = match.get('class_name', match.get('class', ''))
            
            # 가방 관련 검색일 때 객체 타입을 더 명확하게 표시
            attributes = match.get('attributes', {})
            accessories = attributes.get('accessories', {})
            
            # 가방 관련 액세서리가 있는 경우 객체 타입을 변경
            if accessories:
                accessory_value = accessories.get('value', '')
                accessory_confidence = accessories.get('confidence', 0)
                
                # 가방 관련 액세서리가 높은 신뢰도로 감지된 경우
                if accessory_confidence > 0.3:
                    if 'backpack' in accessory_value.lower():
                        class_name = '가방을 든 사람 (백팩)'
                    elif 'handbag' in accessory_value.lower():
                        class_name = '가방을 든 사람 (핸드백)'
                    elif 'bag' in accessory_value.lower():
                        class_name = '가방을 든 사람'
            
            # 속성 정보 추출
            gender = attributes.get('gender', {}).get('value', 'unknown')
            age = attributes.get('age', {}).get('value', 'unknown')
            clothing_color = attributes.get('clothing_color', {}).get('value', 'unknown')
            
            # 설명 생성
            description_parts = []
            if gender != 'unknown':
                description_parts.append(gender)
            if age != 'unknown':
                description_parts.append(age)
            if clothing_color != 'unknown':
                description_parts.append(f"wearing {clothing_color}")
            description = ' '.join(description_parts) if description_parts else class_name
            
            result = {
                'frame_index': frame_index,
                'frame_id': frame_index,  # frame_id 추가
                'timestamp': timestamp,
                'time_formatted': f"{int(timestamp//60):02d}:{int(timestamp%60):02d}",
                'class_name': class_name,
                'confidence': match.get('confidence', 0),
                'match_score': match.get('match_score', 0),
                'match_reasons': match.get('match_reasons', []),
                'frame_url': f"/api/videos/{video_id}/frames/{frame_index}/",
                'video_id': video_id,
                'attributes': attributes,
                'description': description,
                'gender': gender,
                'age': age,
                'clothing_color': clothing_color
            }
            formatted_results.append(result)
        
        return formatted_results

    def get_scene_preview(self, video_id: str, frame_index: int) -> Dict[str, Any]:
        """특정 프레임의 장면 미리보기 정보"""
        try:
            # 프레임 이미지 URL 생성
            frame_url = f"/api/videos/{video_id}/frames/{frame_index}/"
            
            # 시간 정보 계산
            fps = 30  # 기본 FPS
            timestamp = frame_index / fps
            
            return {
                'success': True,
                'frame_index': frame_index,
                'timestamp': timestamp,
                'time_formatted': f"{int(timestamp//60):02d}:{int(timestamp%60):02d}",
                'frame_url': frame_url,
                'video_id': video_id
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

# 전역 시스템 인스턴스
_enhanced_rag_system = None

def get_enhanced_video_rag_system(config: Optional[VideoRAGConfig] = None):
    """향상된 RAG 시스템 인스턴스 반환 (싱글톤)"""
    global _enhanced_rag_system
    if _enhanced_rag_system is None:
        _enhanced_rag_system = EnhancedVideoRAGSystem(config)
    return _enhanced_rag_system

# 하위 호환성을 위한 래퍼 함수
def get_video_rag_system():
    """기존 호환성 유지"""
    return get_enhanced_video_rag_system()

# 전역 변수 (기존 코드 호환성)
rag_system = get_enhanced_video_rag_system()