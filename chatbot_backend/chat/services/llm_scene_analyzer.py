# chat/services/llm_scene_analyzer.py - LLM 기반 장면 분석기
import os
import json
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple
from django.conf import settings
from ..models import Video, VideoScene, SceneAnalysis, SemanticEmbedding

# LLM 클라이언트 import
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)

class LLMSceneAnalyzer:
    """LLM 기반 장면 분석기 - 장면 설명 생성 및 의미적 임베딩"""
    
    def __init__(self):
        self.ollama_available = OLLAMA_AVAILABLE
        self.embedding_available = SENTENCE_TRANSFORMERS_AVAILABLE
        
        # 임베딩 모델 초기화
        self.embedding_model = None
        if self.embedding_available:
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("✅ 임베딩 모델 초기화 완료")
            except Exception as e:
                logger.warning(f"⚠️ 임베딩 모델 초기화 실패: {e}")
                self.embedding_available = False
        
        logger.info(f"🤖 LLM 장면 분석기 초기화 완료 - Ollama: {self.ollama_available}, Embedding: {self.embedding_available}")
    
    def analyze_scene_with_llm(self, scene: VideoScene) -> Dict:
        """LLM을 사용하여 장면을 분석하고 설명 생성"""
        try:
            logger.info(f"🎬 LLM 장면 분석 시작: Scene {scene.scene_id}")
            
            # 장면 정보 수집
            scene_info = self._collect_scene_info(scene)
            
            # LLM으로 장면 설명 생성
            scene_description = self._generate_scene_description(scene_info)
            
            # 의미적 임베딩 생성
            semantic_embedding = self._create_semantic_embedding(scene_description)
            
            # 검색 키워드 및 태그 추출
            keywords, tags = self._extract_keywords_and_tags(scene_description, scene_info)
            
            # 결과 구성
            analysis_result = {
                'scene_description': scene_description,
                'semantic_embedding': semantic_embedding,
                'search_keywords': keywords,
                'semantic_tags': tags,
                'confidence_score': self._calculate_confidence_score(scene_info)
            }
            
            # DB 업데이트
            self._update_scene_with_llm_analysis(scene, analysis_result)
            
            logger.info(f"✅ LLM 장면 분석 완료: Scene {scene.scene_id}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"❌ LLM 장면 분석 실패: {e}")
            raise
    
    def _collect_scene_info(self, scene: VideoScene) -> Dict:
        """장면의 모든 정보를 수집"""
        try:
            # 기본 장면 정보
            scene_info = {
                'scene_id': scene.scene_id,
                'start_timestamp': scene.start_timestamp,
                'end_timestamp': scene.end_timestamp,
                'duration': scene.duration,
                'scene_type': scene.scene_type,
                'weather_condition': scene.weather_condition,
                'time_of_day': scene.time_of_day,
                'lighting_condition': scene.lighting_condition,
                'dominant_colors': scene.dominant_colors,
                'dominant_objects': scene.dominant_objects,
                'quality_score': scene.quality_score
            }
            
            # 분석 정보 추가
            if hasattr(scene, 'analysis'):
                analysis = scene.analysis
                scene_info.update({
                    'detected_persons': analysis.detected_persons,
                    'detected_objects': analysis.detected_objects,
                    'person_count': analysis.person_count,
                    'object_count': analysis.object_count,
                    'activity_type': analysis.activity_type,
                    'activity_intensity': analysis.activity_intensity,
                    'emotional_tone': analysis.emotional_tone,
                    'atmosphere': analysis.atmosphere,
                    'brightness_level': analysis.brightness_level,
                    'contrast_level': analysis.contrast_level,
                    'sharpness_level': analysis.sharpness_level
                })
            
            return scene_info
            
        except Exception as e:
            logger.warning(f"장면 정보 수집 실패: {e}")
            return {}
    
    def _generate_scene_description(self, scene_info: Dict) -> str:
        """LLM을 사용하여 장면 설명 생성"""
        try:
            if not self.ollama_available:
                return self._generate_fallback_description(scene_info)
            
            # 프롬프트 구성
            prompt = self._create_scene_description_prompt(scene_info)
            
            # Ollama로 설명 생성
            response = ollama.chat(
                model='llama3.2:latest',
                messages=[
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                options={
                    'temperature': 0.7,
                    'num_predict': 500
                }
            )
            
            description = response['message']['content'].strip()
            
            # 설명 검증 및 정제
            description = self._validate_and_refine_description(description)
            
            return description
            
        except Exception as e:
            logger.warning(f"LLM 설명 생성 실패: {e}")
            return self._generate_fallback_description(scene_info)
    
    def _create_scene_description_prompt(self, scene_info: Dict) -> str:
        """장면 설명 생성을 위한 프롬프트 생성"""
        prompt = f"""
다음 비디오 장면 정보를 바탕으로 자연스럽고 상세한 장면 설명을 한국어로 생성해주세요.

장면 정보:
- 장면 번호: {scene_info.get('scene_id', 'N/A')}
- 시간: {scene_info.get('start_timestamp', 0):.1f}초 - {scene_info.get('end_timestamp', 0):.1f}초 ({scene_info.get('duration', 0):.1f}초간)
- 장면 유형: {scene_info.get('scene_type', 'unknown')}
- 날씨: {scene_info.get('weather_condition', 'unknown')}
- 시간대: {scene_info.get('time_of_day', 'unknown')}
- 조명: {scene_info.get('lighting_condition', 'unknown')}
- 주요 색상: {', '.join(scene_info.get('dominant_colors', []))}
- 주요 객체: {', '.join(scene_info.get('dominant_objects', []))}
- 사람 수: {scene_info.get('person_count', 0)}명
- 객체 수: {scene_info.get('object_count', 0)}개
- 활동 유형: {scene_info.get('activity_type', 'unknown')}
- 활동 강도: {scene_info.get('activity_intensity', 'unknown')}
- 감정적 톤: {scene_info.get('emotional_tone', 'unknown')}
- 분위기: {scene_info.get('atmosphere', 'unknown')}
- 품질 점수: {scene_info.get('quality_score', 0):.2f}

요구사항:
1. 자연스럽고 구체적인 장면 설명을 작성해주세요
2. 시간, 장소, 인물, 활동, 분위기를 포함해주세요
3. 2-3문장으로 간결하게 작성해주세요
4. 한국어로 작성해주세요

장면 설명:
"""
        return prompt
    
    def _validate_and_refine_description(self, description: str) -> str:
        """생성된 설명을 검증하고 정제"""
        try:
            # 기본 검증
            if not description or len(description.strip()) < 10:
                return "이 장면은 일반적인 상황을 보여줍니다."
            
            # 불필요한 접두사 제거
            prefixes_to_remove = [
                "장면 설명:", "설명:", "이 장면은", "이 프레임은", 
                "다음은", "위 장면은", "장면 정보를 바탕으로"
            ]
            
            for prefix in prefixes_to_remove:
                if description.startswith(prefix):
                    description = description[len(prefix):].strip()
            
            # 길이 제한 (너무 길면 자르기)
            if len(description) > 500:
                description = description[:500] + "..."
            
            return description.strip()
            
        except Exception as e:
            logger.warning(f"설명 정제 실패: {e}")
            return description
    
    def _generate_fallback_description(self, scene_info: Dict) -> str:
        """LLM 사용 불가 시 대체 설명 생성"""
        try:
            # 기본 정보로 간단한 설명 생성
            scene_type = scene_info.get('scene_type', '일반적인')
            time_of_day = scene_info.get('time_of_day', '시간대')
            weather = scene_info.get('weather_condition', '날씨')
            person_count = scene_info.get('person_count', 0)
            
            description_parts = []
            
            # 시간대와 날씨
            if time_of_day != 'unknown' and weather != 'unknown':
                description_parts.append(f"{time_of_day} 시간대의 {weather} 날씨")
            elif time_of_day != 'unknown':
                description_parts.append(f"{time_of_day} 시간대")
            elif weather != 'unknown':
                description_parts.append(f"{weather} 날씨")
            
            # 사람 수
            if person_count > 0:
                if person_count == 1:
                    description_parts.append("사람 1명이 등장")
                else:
                    description_parts.append(f"사람 {person_count}명이 등장")
            
            # 장면 유형
            if scene_info.get('scene_type') != 'unknown':
                description_parts.append(f"{scene_info.get('scene_type')} 장면")
            
            if description_parts:
                description = f"이 장면은 {', '.join(description_parts)}을 보여줍니다."
            else:
                description = "이 장면은 일반적인 상황을 보여줍니다."
            
            return description
            
        except Exception as e:
            logger.warning(f"대체 설명 생성 실패: {e}")
            return "이 장면은 일반적인 상황을 보여줍니다."
    
    def _create_semantic_embedding(self, text: str) -> List[float]:
        """텍스트의 의미적 임베딩 생성"""
        try:
            if not self.embedding_available or not self.embedding_model:
                return []
            
            # 임베딩 생성
            embedding = self.embedding_model.encode(text)
            return embedding.tolist()
            
        except Exception as e:
            logger.warning(f"임베딩 생성 실패: {e}")
            return []
    
    def _extract_keywords_and_tags(self, description: str, scene_info: Dict) -> Tuple[List[str], List[str]]:
        """설명에서 키워드와 태그 추출"""
        try:
            keywords = []
            tags = []
            
            # 기본 키워드 추출
            keywords.extend(scene_info.get('dominant_colors', []))
            keywords.extend(scene_info.get('dominant_objects', []))
            
            # 시간대 태그
            if scene_info.get('time_of_day') != 'unknown':
                tags.append(f"시간대_{scene_info.get('time_of_day')}")
            
            # 날씨 태그
            if scene_info.get('weather_condition') != 'unknown':
                tags.append(f"날씨_{scene_info.get('weather_condition')}")
            
            # 장면 유형 태그
            if scene_info.get('scene_type') != 'unknown':
                tags.append(f"장면_{scene_info.get('scene_type')}")
            
            # 활동 태그
            if scene_info.get('activity_type') != 'unknown':
                tags.append(f"활동_{scene_info.get('activity_type')}")
            
            # 사람 수 태그
            person_count = scene_info.get('person_count', 0)
            if person_count > 0:
                tags.append("사람_감지")
                if person_count == 1:
                    tags.append("사람_1명")
                elif person_count > 1:
                    tags.append(f"사람_{person_count}명")
            
            # 중복 제거
            keywords = list(set(keywords))
            tags = list(set(tags))
            
            return keywords, tags
            
        except Exception as e:
            logger.warning(f"키워드/태그 추출 실패: {e}")
            return [], []
    
    def _calculate_confidence_score(self, scene_info: Dict) -> float:
        """분석 신뢰도 점수 계산"""
        try:
            score = 0.5  # 기본 점수
            
            # 정보 완성도에 따른 점수 조정
            if scene_info.get('scene_type') != 'unknown':
                score += 0.1
            if scene_info.get('time_of_day') != 'unknown':
                score += 0.1
            if scene_info.get('weather_condition') != 'unknown':
                score += 0.1
            if scene_info.get('person_count', 0) > 0:
                score += 0.1
            if scene_info.get('quality_score', 0) > 0.5:
                score += 0.1
            
            return min(1.0, score)
            
        except Exception as e:
            logger.warning(f"신뢰도 계산 실패: {e}")
            return 0.5
    
    def _update_scene_with_llm_analysis(self, scene: VideoScene, analysis_result: Dict):
        """LLM 분석 결과로 장면 정보 업데이트"""
        try:
            # 장면 정보 업데이트
            scene.scene_description = analysis_result.get('scene_description', '')
            scene.semantic_embedding = analysis_result.get('semantic_embedding', [])
            scene.search_keywords = analysis_result.get('search_keywords', [])
            scene.semantic_tags = analysis_result.get('semantic_tags', [])
            scene.confidence_score = analysis_result.get('confidence_score', 0.5)
            scene.save()
            
            # 의미적 임베딩 저장
            if analysis_result.get('semantic_embedding'):
                self._save_semantic_embedding(scene, analysis_result['semantic_embedding'])
            
            logger.info(f"✅ 장면 LLM 분석 결과 저장 완료: Scene {scene.scene_id}")
            
        except Exception as e:
            logger.error(f"❌ 장면 LLM 분석 결과 저장 실패: {e}")
            raise
    
    def _save_semantic_embedding(self, scene: VideoScene, embedding: List[float]):
        """의미적 임베딩을 DB에 저장"""
        try:
            if not embedding:
                return
            
            # 기존 임베딩 삭제
            SemanticEmbedding.objects.filter(
                embedding_type='scene',
                content_id=scene.id,
                content_type='VideoScene'
            ).delete()
            
            # 새 임베딩 저장
            SemanticEmbedding.objects.create(
                embedding_type='scene',
                content_id=scene.id,
                content_type='VideoScene',
                embedding_vector=embedding,
                embedding_dimension=len(embedding),
                embedding_model='all-MiniLM-L6-v2',
                original_text=scene.scene_description
            )
            
        except Exception as e:
            logger.warning(f"의미적 임베딩 저장 실패: {e}")

class QueryProcessor:
    """자연어 쿼리 처리기"""
    
    def __init__(self):
        self.ollama_available = OLLAMA_AVAILABLE
    
    def parse_natural_query(self, query: str) -> Dict:
        """자연어 쿼리를 구조화된 검색 조건으로 변환"""
        try:
            if not self.ollama_available:
                return self._parse_query_fallback(query)
            
            # LLM을 사용한 쿼리 분석
            prompt = self._create_query_analysis_prompt(query)
            
            response = ollama.chat(
                model='llama3.2:latest',
                messages=[
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                options={
                    'temperature': 0.3,
                    'num_predict': 300
                }
            )
            
            result = response['message']['content'].strip()
            return self._parse_structured_response(result)
            
        except Exception as e:
            logger.warning(f"쿼리 분석 실패: {e}")
            return self._parse_query_fallback(query)
    
    def _create_query_analysis_prompt(self, query: str) -> str:
        """쿼리 분석을 위한 프롬프트 생성"""
        prompt = f"""
다음 비디오 검색 쿼리를 분석하여 구조화된 검색 조건을 JSON 형식으로 생성해주세요.

쿼리: "{query}"

다음 형식으로 응답해주세요:
{{
    "search_type": "cross_video|intra_video|time_analysis",
    "target_video_id": null,
    "conditions": {{
        "weather": ["rain", "snow", "sunny", "cloudy"],
        "time_of_day": ["morning", "afternoon", "evening", "night", "dawn"],
        "objects": ["person", "car", "building", "tree"],
        "colors": ["red", "blue", "green", "yellow", "orange", "purple", "pink", "black", "white", "gray"],
        "activities": ["walking", "running", "standing", "sitting"],
        "scene_context": ["indoor", "outdoor", "street", "building", "park"]
    }},
    "temporal_constraints": {{
        "start_time": null,
        "end_time": null,
        "duration_range": null
    }},
    "semantic_intent": "사용자의 의도 설명"
}}

분석 가이드:
1. 날씨 관련 키워드: 비, 눈, 맑음, 흐림, 비오는, 눈오는 등
2. 시간대 키워드: 아침, 점심, 저녁, 밤, 새벽, 낮, 밤 등
3. 색상 키워드: 빨간, 파란, 초록, 노란, 주황, 보라, 분홍, 검은, 흰, 회색 등
4. 장소 키워드: 실내, 실외, 거리, 건물, 공원, 집, 사무실 등
5. 활동 키워드: 걷는, 뛰는, 서있는, 앉아있는, 대화하는 등

JSON만 응답해주세요:
"""
        return prompt
    
    def _parse_structured_response(self, response: str) -> Dict:
        """LLM 응답을 구조화된 데이터로 파싱"""
        try:
            # JSON 부분만 추출
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("JSON 형식을 찾을 수 없습니다")
            
            json_str = response[start_idx:end_idx]
            result = json.loads(json_str)
            
            # 기본값 설정
            if 'search_type' not in result:
                result['search_type'] = 'cross_video'
            if 'conditions' not in result:
                result['conditions'] = {}
            if 'temporal_constraints' not in result:
                result['temporal_constraints'] = {}
            
            return result
            
        except Exception as e:
            logger.warning(f"구조화된 응답 파싱 실패: {e}")
            return self._parse_query_fallback(response)
    
    def _parse_query_fallback(self, query: str) -> Dict:
        """LLM 사용 불가 시 대체 쿼리 분석"""
        try:
            query_lower = query.lower()
            
            # 기본 구조
            result = {
                'search_type': 'cross_video',
                'target_video_id': None,
                'conditions': {},
                'temporal_constraints': {},
                'semantic_intent': query
            }
            
            # 날씨 키워드
            weather_keywords = {
                '비': 'rain', '눈': 'snow', '맑음': 'sunny', '흐림': 'cloudy',
                '비오는': 'rain', '눈오는': 'snow', '맑은': 'sunny', '흐린': 'cloudy'
            }
            
            detected_weather = []
            for korean, english in weather_keywords.items():
                if korean in query:
                    detected_weather.append(english)
            
            if detected_weather:
                result['conditions']['weather'] = detected_weather
            
            # 시간대 키워드
            time_keywords = {
                '아침': 'morning', '점심': 'afternoon', '저녁': 'evening', 
                '밤': 'night', '새벽': 'dawn', '낮': 'afternoon'
            }
            
            detected_time = []
            for korean, english in time_keywords.items():
                if korean in query:
                    detected_time.append(english)
            
            if detected_time:
                result['conditions']['time_of_day'] = detected_time
            
            # 색상 키워드
            color_keywords = {
                '빨간': 'red', '파란': 'blue', '초록': 'green', '노란': 'yellow',
                '주황': 'orange', '보라': 'purple', '분홍': 'pink', '검은': 'black',
                '흰': 'white', '회색': 'gray'
            }
            
            detected_colors = []
            for korean, english in color_keywords.items():
                if korean in query:
                    detected_colors.append(english)
            
            if detected_colors:
                result['conditions']['colors'] = detected_colors
            
            return result
            
        except Exception as e:
            logger.warning(f"대체 쿼리 분석 실패: {e}")
            return {
                'search_type': 'cross_video',
                'target_video_id': None,
                'conditions': {},
                'temporal_constraints': {},
                'semantic_intent': query
            }

# 전역 인스턴스
llm_scene_analyzer = LLMSceneAnalyzer()
query_processor = QueryProcessor()
