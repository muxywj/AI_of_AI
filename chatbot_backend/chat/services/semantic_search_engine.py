# chat/services/semantic_search_engine.py - 의미적 검색 엔진
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
from django.db.models import Q
from sklearn.metrics.pairwise import cosine_similarity
from ..models import Video, VideoScene, SceneAnalysis, SemanticEmbedding
from .llm_scene_analyzer import llm_scene_analyzer, query_processor

logger = logging.getLogger(__name__)

class SemanticSearchEngine:
    """의미적 검색 엔진 - 벡터 임베딩 기반 유사도 검색"""
    
    def __init__(self):
        self.similarity_threshold = 0.3  # 유사도 임계값
        self.max_results = 50  # 최대 결과 수
        
    def search_scenes_by_query(self, query: str, video_id: Optional[int] = None) -> List[Dict]:
        """자연어 쿼리로 장면 검색"""
        try:
            logger.info(f"🔍 의미적 검색 시작: '{query}' (비디오: {video_id})")
            
            # 쿼리 분석
            parsed_query = query_processor.parse_natural_query(query)
            
            # 검색 타입에 따른 분기
            if parsed_query.get('search_type') == 'intra_video' and video_id:
                results = self._search_within_video(video_id, parsed_query)
            else:
                results = self._search_across_videos(parsed_query)
            
            # 결과 정렬 및 필터링
            filtered_results = self._filter_and_rank_results(results, parsed_query)
            
            logger.info(f"✅ 의미적 검색 완료: {len(filtered_results)}개 결과")
            return filtered_results
            
        except Exception as e:
            logger.error(f"❌ 의미적 검색 실패: {e}")
            return []
    
    def _search_within_video(self, video_id: int, parsed_query: Dict) -> List[Dict]:
        """특정 비디오 내에서 검색"""
        try:
            # 비디오의 모든 장면 가져오기
            scenes = VideoScene.objects.filter(video_id=video_id).select_related('analysis')
            
            results = []
            for scene in scenes:
                # 조건 매칭 점수 계산
                match_score = self._calculate_condition_match_score(scene, parsed_query['conditions'])
                
                if match_score > self.similarity_threshold:
                    # 의미적 유사도 계산 (임베딩이 있는 경우)
                    semantic_score = self._calculate_semantic_similarity(scene, parsed_query['semantic_intent'])
                    
                    # 최종 점수 계산
                    final_score = (match_score * 0.7) + (semantic_score * 0.3)
                    
                    results.append({
                        'scene_id': scene.scene_id,
                        'video_id': video_id,
                        'start_timestamp': scene.start_timestamp,
                        'end_timestamp': scene.end_timestamp,
                        'duration': scene.duration,
                        'scene_description': scene.scene_description,
                        'match_score': match_score,
                        'semantic_score': semantic_score,
                        'final_score': final_score,
                        'match_reasons': self._get_match_reasons(scene, parsed_query['conditions']),
                        'metadata': self._extract_scene_metadata(scene)
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"❌ 비디오 내 검색 실패: {e}")
            return []
    
    def _search_across_videos(self, parsed_query: Dict) -> List[Dict]:
        """전체 비디오에서 검색"""
        try:
            # 모든 분석된 비디오의 장면들 가져오기
            scenes = VideoScene.objects.filter(
                video__is_analyzed=True
            ).select_related('video', 'analysis')
            
            results = []
            for scene in scenes:
                # 조건 매칭 점수 계산
                match_score = self._calculate_condition_match_score(scene, parsed_query['conditions'])
                
                if match_score > self.similarity_threshold:
                    # 의미적 유사도 계산
                    semantic_score = self._calculate_semantic_similarity(scene, parsed_query['semantic_intent'])
                    
                    # 최종 점수 계산
                    final_score = (match_score * 0.7) + (semantic_score * 0.3)
                    
                    results.append({
                        'scene_id': scene.scene_id,
                        'video_id': scene.video_id,
                        'video_name': scene.video.original_name,
                        'start_timestamp': scene.start_timestamp,
                        'end_timestamp': scene.end_timestamp,
                        'duration': scene.duration,
                        'scene_description': scene.scene_description,
                        'match_score': match_score,
                        'semantic_score': semantic_score,
                        'final_score': final_score,
                        'match_reasons': self._get_match_reasons(scene, parsed_query['conditions']),
                        'metadata': self._extract_scene_metadata(scene)
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"❌ 전체 비디오 검색 실패: {e}")
            return []
    
    def _calculate_condition_match_score(self, scene: VideoScene, conditions: Dict) -> float:
        """조건 매칭 점수 계산"""
        try:
            total_score = 0.0
            total_weight = 0.0
            
            # 날씨 조건
            if 'weather' in conditions and conditions['weather']:
                weather_score = self._match_weather_condition(scene, conditions['weather'])
                total_score += weather_score * 0.2
                total_weight += 0.2
            
            # 시간대 조건
            if 'time_of_day' in conditions and conditions['time_of_day']:
                time_score = self._match_time_condition(scene, conditions['time_of_day'])
                total_score += time_score * 0.2
                total_weight += 0.2
            
            # 색상 조건
            if 'colors' in conditions and conditions['colors']:
                color_score = self._match_color_condition(scene, conditions['colors'])
                total_score += color_score * 0.15
                total_weight += 0.15
            
            # 객체 조건
            if 'objects' in conditions and conditions['objects']:
                object_score = self._match_object_condition(scene, conditions['objects'])
                total_score += object_score * 0.15
                total_weight += 0.15
            
            # 장면 맥락 조건
            if 'scene_context' in conditions and conditions['scene_context']:
                context_score = self._match_scene_context(scene, conditions['scene_context'])
                total_score += context_score * 0.15
                total_weight += 0.15
            
            # 활동 조건
            if 'activities' in conditions and conditions['activities']:
                activity_score = self._match_activity_condition(scene, conditions['activities'])
                total_score += activity_score * 0.15
                total_weight += 0.15
            
            return total_score / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            logger.warning(f"조건 매칭 점수 계산 실패: {e}")
            return 0.0
    
    def _match_weather_condition(self, scene: VideoScene, weather_conditions: List[str]) -> float:
        """날씨 조건 매칭"""
        scene_weather = scene.weather_condition.lower()
        
        for weather in weather_conditions:
            if weather.lower() in scene_weather or scene_weather in weather.lower():
                return 1.0
        
        return 0.0
    
    def _match_time_condition(self, scene: VideoScene, time_conditions: List[str]) -> float:
        """시간대 조건 매칭"""
        scene_time = scene.time_of_day.lower()
        
        for time in time_conditions:
            if time.lower() in scene_time or scene_time in time.lower():
                return 1.0
        
        return 0.0
    
    def _match_color_condition(self, scene: VideoScene, color_conditions: List[str]) -> float:
        """색상 조건 매칭"""
        scene_colors = [color.lower() for color in scene.dominant_colors]
        
        if not scene_colors:
            return 0.0
        
        matches = 0
        for color in color_conditions:
            if color.lower() in scene_colors:
                matches += 1
        
        return matches / len(color_conditions) if color_conditions else 0.0
    
    def _match_object_condition(self, scene: VideoScene, object_conditions: List[str]) -> float:
        """객체 조건 매칭"""
        scene_objects = [obj.lower() for obj in scene.dominant_objects]
        
        if not scene_objects:
            return 0.0
        
        matches = 0
        for obj in object_conditions:
            if obj.lower() in scene_objects:
                matches += 1
        
        return matches / len(object_conditions) if object_conditions else 0.0
    
    def _match_scene_context(self, scene: VideoScene, context_conditions: List[str]) -> float:
        """장면 맥락 조건 매칭"""
        scene_type = scene.scene_type.lower()
        
        for context in context_conditions:
            if context.lower() in scene_type or scene_type in context.lower():
                return 1.0
        
        return 0.0
    
    def _match_activity_condition(self, scene: VideoScene, activity_conditions: List[str]) -> float:
        """활동 조건 매칭"""
        if not hasattr(scene, 'analysis'):
            return 0.0
        
        activity_type = scene.analysis.activity_type.lower()
        
        for activity in activity_conditions:
            if activity.lower() in activity_type or activity_type in activity.lower():
                return 1.0
        
        return 0.0
    
    def _calculate_semantic_similarity(self, scene: VideoScene, query_intent: str) -> float:
        """의미적 유사도 계산"""
        try:
            if not scene.semantic_embedding:
                return 0.0
            
            # 쿼리 임베딩 생성
            query_embedding = llm_scene_analyzer._create_semantic_embedding(query_intent)
            
            if not query_embedding:
                return 0.0
            
            # 코사인 유사도 계산
            similarity = cosine_similarity(
                [scene.semantic_embedding],
                [query_embedding]
            )[0][0]
            
            return float(similarity)
            
        except Exception as e:
            logger.warning(f"의미적 유사도 계산 실패: {e}")
            return 0.0
    
    def _get_match_reasons(self, scene: VideoScene, conditions: Dict) -> List[str]:
        """매칭 이유 추출"""
        reasons = []
        
        # 날씨 매칭
        if 'weather' in conditions and scene.weather_condition:
            for weather in conditions['weather']:
                if weather.lower() in scene.weather_condition.lower():
                    reasons.append(f"날씨: {scene.weather_condition}")
                    break
        
        # 시간대 매칭
        if 'time_of_day' in conditions and scene.time_of_day:
            for time in conditions['time_of_day']:
                if time.lower() in scene.time_of_day.lower():
                    reasons.append(f"시간대: {scene.time_of_day}")
                    break
        
        # 색상 매칭
        if 'colors' in conditions and scene.dominant_colors:
            matched_colors = []
            for color in conditions['colors']:
                if color.lower() in [c.lower() for c in scene.dominant_colors]:
                    matched_colors.append(color)
            if matched_colors:
                reasons.append(f"색상: {', '.join(matched_colors)}")
        
        # 객체 매칭
        if 'objects' in conditions and scene.dominant_objects:
            matched_objects = []
            for obj in conditions['objects']:
                if obj.lower() in [o.lower() for o in scene.dominant_objects]:
                    matched_objects.append(obj)
            if matched_objects:
                reasons.append(f"객체: {', '.join(matched_objects)}")
        
        return reasons
    
    def _extract_scene_metadata(self, scene: VideoScene) -> Dict:
        """장면 메타데이터 추출"""
        metadata = {
            'scene_type': scene.scene_type,
            'weather_condition': scene.weather_condition,
            'time_of_day': scene.time_of_day,
            'lighting_condition': scene.lighting_condition,
            'dominant_colors': scene.dominant_colors,
            'dominant_objects': scene.dominant_objects,
            'quality_score': scene.quality_score,
            'confidence_score': scene.confidence_score
        }
        
        # 분석 정보 추가
        if hasattr(scene, 'analysis'):
            analysis = scene.analysis
            metadata.update({
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
        
        return metadata
    
    def _filter_and_rank_results(self, results: List[Dict], parsed_query: Dict) -> List[Dict]:
        """결과 필터링 및 순위 정렬"""
        try:
            # 시간 제약 조건 적용
            temporal_constraints = parsed_query.get('temporal_constraints', {})
            if temporal_constraints:
                results = self._apply_temporal_constraints(results, temporal_constraints)
            
            # 점수순 정렬
            results.sort(key=lambda x: x['final_score'], reverse=True)
            
            # 상위 결과만 반환
            return results[:self.max_results]
            
        except Exception as e:
            logger.warning(f"결과 필터링/정렬 실패: {e}")
            return results
    
    def _apply_temporal_constraints(self, results: List[Dict], constraints: Dict) -> List[Dict]:
        """시간 제약 조건 적용"""
        try:
            filtered_results = []
            
            start_time = constraints.get('start_time')
            end_time = constraints.get('end_time')
            
            for result in results:
                scene_start = result['start_timestamp']
                scene_end = result['end_timestamp']
                
                # 시간 범위 체크
                if start_time is not None and scene_end < start_time:
                    continue
                if end_time is not None and scene_start > end_time:
                    continue
                
                filtered_results.append(result)
            
            return filtered_results
            
        except Exception as e:
            logger.warning(f"시간 제약 조건 적용 실패: {e}")
            return results

class HybridSearchEngine:
    """하이브리드 검색 엔진 - 의미적 + 키워드 + 메타데이터 검색"""
    
    def __init__(self):
        self.semantic_engine = SemanticSearchEngine()
        
    def search(self, query: str, video_id: Optional[int] = None, search_type: str = 'hybrid') -> Dict:
        """하이브리드 검색 실행"""
        try:
            logger.info(f"🔍 하이브리드 검색 시작: '{query}' (타입: {search_type})")
            
            results = {
                'query': query,
                'search_type': search_type,
                'total_results': 0,
                'results': [],
                'search_metadata': {}
            }
            
            if search_type in ['semantic', 'hybrid']:
                # 의미적 검색
                semantic_results = self.semantic_engine.search_scenes_by_query(query, video_id)
                results['semantic_results'] = semantic_results
                results['search_metadata']['semantic_count'] = len(semantic_results)
            
            if search_type in ['keyword', 'hybrid']:
                # 키워드 검색
                keyword_results = self._keyword_search(query, video_id)
                results['keyword_results'] = keyword_results
                results['search_metadata']['keyword_count'] = len(keyword_results)
            
            if search_type in ['metadata', 'hybrid']:
                # 메타데이터 검색
                metadata_results = self._metadata_search(query, video_id)
                results['metadata_results'] = metadata_results
                results['search_metadata']['metadata_count'] = len(metadata_results)
            
            # 결과 통합
            if search_type == 'hybrid':
                results['results'] = self._merge_search_results(results)
            elif search_type == 'semantic':
                results['results'] = results.get('semantic_results', [])
            elif search_type == 'keyword':
                results['results'] = results.get('keyword_results', [])
            elif search_type == 'metadata':
                results['results'] = results.get('metadata_results', [])
            
            results['total_results'] = len(results['results'])
            
            logger.info(f"✅ 하이브리드 검색 완료: {results['total_results']}개 결과")
            return results
            
        except Exception as e:
            logger.error(f"❌ 하이브리드 검색 실패: {e}")
            return {
                'query': query,
                'search_type': search_type,
                'total_results': 0,
                'results': [],
                'error': str(e)
            }
    
    def _keyword_search(self, query: str, video_id: Optional[int] = None) -> List[Dict]:
        """키워드 기반 검색"""
        try:
            # 검색 키워드 추출
            keywords = query.lower().split()
            
            # 장면 설명에서 키워드 검색
            q_objects = Q()
            for keyword in keywords:
                q_objects |= Q(scene_description__icontains=keyword)
                q_objects |= Q(search_keywords__icontains=keyword)
                q_objects |= Q(semantic_tags__icontains=keyword)
            
            scenes_query = VideoScene.objects.filter(q_objects)
            
            if video_id:
                scenes_query = scenes_query.filter(video_id=video_id)
            
            scenes = scenes_query.select_related('video', 'analysis')
            
            results = []
            for scene in scenes:
                # 키워드 매칭 점수 계산
                match_score = self._calculate_keyword_match_score(scene, keywords)
                
                if match_score > 0.1:  # 낮은 임계값
                    results.append({
                        'scene_id': scene.scene_id,
                        'video_id': scene.video_id,
                        'video_name': scene.video.original_name if not video_id else None,
                        'start_timestamp': scene.start_timestamp,
                        'end_timestamp': scene.end_timestamp,
                        'duration': scene.duration,
                        'scene_description': scene.scene_description,
                        'match_score': match_score,
                        'semantic_score': 0.0,
                        'final_score': match_score,
                        'match_reasons': [f"키워드: {', '.join(keywords)}"],
                        'metadata': self.semantic_engine._extract_scene_metadata(scene)
                    })
            
            # 점수순 정렬
            results.sort(key=lambda x: x['final_score'], reverse=True)
            return results[:50]  # 상위 50개
            
        except Exception as e:
            logger.warning(f"키워드 검색 실패: {e}")
            return []
    
    def _metadata_search(self, query: str, video_id: Optional[int] = None) -> List[Dict]:
        """메타데이터 기반 검색"""
        try:
            # 쿼리 분석
            parsed_query = query_processor.parse_natural_query(query)
            conditions = parsed_query.get('conditions', {})
            
            # 메타데이터 필터링
            q_objects = Q()
            
            if 'weather' in conditions:
                for weather in conditions['weather']:
                    q_objects |= Q(weather_condition__icontains=weather)
            
            if 'time_of_day' in conditions:
                for time in conditions['time_of_day']:
                    q_objects |= Q(time_of_day__icontains=time)
            
            if 'colors' in conditions:
                for color in conditions['colors']:
                    q_objects |= Q(dominant_colors__icontains=color)
            
            if 'objects' in conditions:
                for obj in conditions['objects']:
                    q_objects |= Q(dominant_objects__icontains=obj)
            
            scenes_query = VideoScene.objects.filter(q_objects)
            
            if video_id:
                scenes_query = scenes_query.filter(video_id=video_id)
            
            scenes = scenes_query.select_related('video', 'analysis')
            
            results = []
            for scene in scenes:
                # 메타데이터 매칭 점수 계산
                match_score = self.semantic_engine._calculate_condition_match_score(scene, conditions)
                
                if match_score > 0.1:
                    results.append({
                        'scene_id': scene.scene_id,
                        'video_id': scene.video_id,
                        'video_name': scene.video.original_name if not video_id else None,
                        'start_timestamp': scene.start_timestamp,
                        'end_timestamp': scene.end_timestamp,
                        'duration': scene.duration,
                        'scene_description': scene.scene_description,
                        'match_score': match_score,
                        'semantic_score': 0.0,
                        'final_score': match_score,
                        'match_reasons': self.semantic_engine._get_match_reasons(scene, conditions),
                        'metadata': self.semantic_engine._extract_scene_metadata(scene)
                    })
            
            # 점수순 정렬
            results.sort(key=lambda x: x['final_score'], reverse=True)
            return results[:50]  # 상위 50개
            
        except Exception as e:
            logger.warning(f"메타데이터 검색 실패: {e}")
            return []
    
    def _calculate_keyword_match_score(self, scene: VideoScene, keywords: List[str]) -> float:
        """키워드 매칭 점수 계산"""
        try:
            text_to_search = f"{scene.scene_description} {' '.join(scene.search_keywords)} {' '.join(scene.semantic_tags)}"
            text_lower = text_to_search.lower()
            
            matches = 0
            for keyword in keywords:
                if keyword in text_lower:
                    matches += 1
            
            return matches / len(keywords) if keywords else 0.0
            
        except Exception as e:
            logger.warning(f"키워드 매칭 점수 계산 실패: {e}")
            return 0.0
    
    def _merge_search_results(self, search_results: Dict) -> List[Dict]:
        """검색 결과 통합"""
        try:
            all_results = []
            seen_scenes = set()
            
            # 모든 검색 결과 수집
            for result_type in ['semantic_results', 'keyword_results', 'metadata_results']:
                if result_type in search_results:
                    for result in search_results[result_type]:
                        scene_key = f"{result['video_id']}_{result['scene_id']}"
                        if scene_key not in seen_scenes:
                            all_results.append(result)
                            seen_scenes.add(scene_key)
                        else:
                            # 중복된 장면의 점수 업데이트
                            for existing_result in all_results:
                                if (existing_result['video_id'] == result['video_id'] and 
                                    existing_result['scene_id'] == result['scene_id']):
                                    existing_result['final_score'] = max(
                                        existing_result['final_score'], 
                                        result['final_score']
                                    )
                                    break
            
            # 점수순 정렬
            all_results.sort(key=lambda x: x['final_score'], reverse=True)
            return all_results[:100]  # 상위 100개
            
        except Exception as e:
            logger.warning(f"검색 결과 통합 실패: {e}")
            return []

# 전역 인스턴스
semantic_search_engine = SemanticSearchEngine()
hybrid_search_engine = HybridSearchEngine()
