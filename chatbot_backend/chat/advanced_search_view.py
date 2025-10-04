from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from chat.models import Video
from django.conf import settings
import json
import os
import logging
import re
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class InterVideoSearchView(APIView):
    """영상 간 검색 (비가오는 밤 영상 찾기 등)"""
    permission_classes = [AllowAny]
    
    def post(self, request):
        try:
            query = request.data.get('query', '')
            search_criteria = request.data.get('criteria', {})
            
            logger.info(f"🔍 영상 간 검색 요청: 쿼리='{query}', 기준={search_criteria}")
            
            if not query:
                return Response({'error': '검색 쿼리가 필요합니다.'}, status=400)
            
            # 모든 분석 완료된 영상 가져오기
            videos = Video.objects.filter(analysis_status='completed')
            
            if not videos.exists():
                return Response({
                    'query': query,
                    'results': [],
                    'total_results': 0,
                    'message': '분석 완료된 영상이 없습니다.'
                })
            
            # 검색 결과 생성
            search_results = self._perform_inter_video_search(videos, query, search_criteria)
            
            return Response({
                'query': query,
                'search_type': 'inter_video',
                'results': search_results,
                'total_results': len(search_results),
                'analysis_type': 'inter_video_search'
            })
            
        except Exception as e:
            logger.error(f"❌ 영상 간 검색 오류: {e}")
            return Response({'error': str(e)}, status=500)
    
    def _perform_inter_video_search(self, videos, query, criteria):
        """영상 간 검색 수행"""
        results = []
        query_lower = query.lower()
        
        for video in videos:
            try:
                # TeletoVision 형식 파일 찾기
                detection_db_path = os.path.join(settings.MEDIA_ROOT, f"{video.original_name}-detection_db.json")
                meta_db_path = os.path.join(settings.MEDIA_ROOT, f"{video.original_name}-meta_db.json")
                
                if not os.path.exists(detection_db_path) or not os.path.exists(meta_db_path):
                    continue
                
                # Detection DB와 Meta DB 읽기
                with open(detection_db_path, 'r', encoding='utf-8') as f:
                    detection_db = json.load(f)
                
                with open(meta_db_path, 'r', encoding='utf-8') as f:
                    meta_db = json.load(f)
                
                # 검색 점수 계산
                relevance_score = self._calculate_inter_video_score(detection_db, meta_db, query_lower, criteria)
                
                if relevance_score > 0.1:
                    result = {
                        'video_id': video.id,
                        'video_name': video.original_name,
                        'filename': video.filename,
                        'relevance_score': relevance_score,
                        'duration': video.duration,
                        'uploaded_at': video.uploaded_at,
                        'matched_scenes': self._find_matching_scenes(meta_db, query_lower),
                        'summary': self._generate_inter_video_summary(detection_db, meta_db, query_lower)
                    }
                    results.append(result)
                    
            except Exception as e:
                logger.warning(f"영상 {video.id} 검색 중 오류: {e}")
                continue
        
        # 관련도 순으로 정렬
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        return results[:10]
    
    def _calculate_inter_video_score(self, detection_db, meta_db, query_lower, criteria):
        """영상 간 검색 점수 계산"""
        score = 0.0
        
        # 비가오는 밤 검색
        if any(keyword in query_lower for keyword in ['비', 'rain', '밤', 'night', '어두운', 'dark']):
            score += self._calculate_weather_time_score(meta_db, 'rainy_night')
        
        # 시간대 검색
        if any(keyword in query_lower for keyword in ['오전', 'morning', '오후', 'afternoon', '저녁', 'evening']):
            score += self._calculate_time_score(meta_db, query_lower)
        
        # 객체 검색
        if any(keyword in query_lower for keyword in ['자동차', 'car', '사람', 'person', '오토바이', 'motorcycle']):
            score += self._calculate_object_score(detection_db, query_lower)
        
        # 색상 검색
        colors = ['빨간', '파란', '노란', '초록', '검은', '흰', '주황', '보라']
        if any(color in query_lower for color in colors):
            score += self._calculate_color_score(meta_db, query_lower)
        
        return min(score, 1.0)
    
    def _calculate_weather_time_score(self, meta_db, condition):
        """날씨/시간 조건 점수 계산"""
        score = 0.0
        frames = meta_db.get('frame', [])
        
        for frame in frames:
            scene_context = frame.get('objects', [{}])[0].get('scene_context', {}) if frame.get('objects') else {}
            lighting = scene_context.get('lighting', '').lower()
            
            if condition == 'rainy_night':
                if 'dark' in lighting:
                    score += 0.3
                # 비 관련 키워드는 실제로는 날씨 데이터가 필요하지만, 여기서는 조명으로 추정
                if 'dark' in lighting:
                    score += 0.2
        
        return min(score, 1.0)
    
    def _calculate_time_score(self, meta_db, query_lower):
        """시간대 점수 계산"""
        score = 0.0
        frames = meta_db.get('frame', [])
        
        for frame in frames:
            timestamp = frame.get('timestamp', 0)
            # 시간대 추정 (실제로는 메타데이터에서 시간 정보를 가져와야 함)
            if '오전' in query_lower or 'morning' in query_lower:
                if 6 <= timestamp % 24 <= 12:
                    score += 0.2
            elif '오후' in query_lower or 'afternoon' in query_lower:
                if 12 <= timestamp % 24 <= 18:
                    score += 0.2
            elif '저녁' in query_lower or 'evening' in query_lower:
                if 18 <= timestamp % 24 <= 22:
                    score += 0.2
        
        return min(score, 1.0)
    
    def _calculate_object_score(self, detection_db, query_lower):
        """객체 검색 점수 계산"""
        score = 0.0
        frames = detection_db.get('frame', [])
        
        for frame in frames:
            objects = frame.get('objects', [])
            for obj in objects:
                class_name = obj.get('class', '').lower()
                if '자동차' in query_lower or 'car' in query_lower:
                    if 'car' in class_name:
                        score += 0.3
                elif '사람' in query_lower or 'person' in query_lower:
                    if 'person' in class_name:
                        score += 0.2
                elif '오토바이' in query_lower or 'motorcycle' in query_lower:
                    if 'motorcycle' in class_name:
                        score += 0.3
        
        return min(score, 1.0)
    
    def _calculate_color_score(self, meta_db, query_lower):
        """색상 검색 점수 계산"""
        score = 0.0
        frames = meta_db.get('frame', [])
        
        for frame in frames:
            objects = frame.get('objects', [])
            for obj in objects:
                attributes = obj.get('attributes', {})
                clothing = attributes.get('clothing', {})
                dominant_color = clothing.get('dominant_color', '').lower()
                
                if any(color in query_lower for color in ['빨간', 'red']):
                    if 'red' in dominant_color:
                        score += 0.3
                elif any(color in query_lower for color in ['파란', 'blue']):
                    if 'blue' in dominant_color:
                        score += 0.3
                elif any(color in query_lower for color in ['주황', 'orange']):
                    if 'orange' in dominant_color:
                        score += 0.3
        
        return min(score, 1.0)
    
    def _find_matching_scenes(self, meta_db, query_lower):
        """매칭되는 장면 찾기"""
        matching_scenes = []
        frames = meta_db.get('frame', [])
        
        for frame in frames:
            timestamp = frame.get('timestamp', 0)
            objects = frame.get('objects', [])
            
            for obj in objects:
                if obj.get('class') == 'person':
                    attributes = obj.get('attributes', {})
                    clothing = attributes.get('clothing', {})
                    dominant_color = clothing.get('dominant_color', '').lower()
                    
                    if any(keyword in query_lower for keyword in ['주황', 'orange']):
                        if 'orange' in dominant_color:
                            matching_scenes.append({
                                'timestamp': timestamp,
                                'description': f"주황색 옷을 입은 사람 발견",
                                'confidence': obj.get('confidence', 0.0)
                            })
        
        return matching_scenes[:5]
    
    def _generate_inter_video_summary(self, detection_db, meta_db, query_lower):
        """영상 간 검색 결과 요약 생성"""
        frames = detection_db.get('frame', [])
        total_frames = len(frames)
        
        summary_parts = []
        
        # 기본 정보
        summary_parts.append(f"총 프레임: {total_frames}개")
        
        # 객체 통계
        total_persons = sum(
            sum(obj.get('num', 0) for obj in frame.get('objects', []) if obj.get('class') == 'person')
            for frame in frames
        )
        if total_persons > 0:
            summary_parts.append(f"감지된 사람: {total_persons}명")
        
        # 시간대 정보
        if any(keyword in query_lower for keyword in ['밤', 'night', '어두운', 'dark']):
            dark_frames = sum(
                1 for frame in meta_db.get('frame', [])
                if any(
                    obj.get('scene_context', {}).get('lighting', '').lower() == 'dark'
                    for obj in frame.get('objects', [])
                )
            )
            if dark_frames > 0:
                summary_parts.append(f"어두운 장면: {dark_frames}개")
        
        return " | ".join(summary_parts) if summary_parts else "관련 정보 없음"


class IntraVideoSearchView(APIView):
    """영상 내 검색 (주황색 상의 남성 추적 등)"""
    permission_classes = [AllowAny]
    
    def post(self, request):
        try:
            video_id = request.data.get('video_id')
            query = request.data.get('query', '')
            search_criteria = request.data.get('criteria', {})
            
            logger.info(f"🔍 영상 내 검색 요청: 비디오={video_id}, 쿼리='{query}'")
            
            if not video_id or not query:
                return Response({'error': '비디오 ID와 검색 쿼리가 필요합니다.'}, status=400)
            
            try:
                video = Video.objects.get(id=video_id)
            except Video.DoesNotExist:
                return Response({'error': '비디오를 찾을 수 없습니다.'}, status=404)
            
            # TeletoVision 형식 파일 찾기
            detection_db_path = os.path.join(settings.MEDIA_ROOT, f"{video.original_name}-detection_db.json")
            meta_db_path = os.path.join(settings.MEDIA_ROOT, f"{video.original_name}-meta_db.json")
            
            if not os.path.exists(detection_db_path) or not os.path.exists(meta_db_path):
                return Response({'error': '분석 결과 파일을 찾을 수 없습니다.'}, status=404)
            
            # Detection DB와 Meta DB 읽기
            with open(detection_db_path, 'r', encoding='utf-8') as f:
                detection_db = json.load(f)
            
            with open(meta_db_path, 'r', encoding='utf-8') as f:
                meta_db = json.load(f)
            
            # 영상 내 검색 수행
            search_results = self._perform_intra_video_search(detection_db, meta_db, query, search_criteria)
            
            return Response({
                'video_id': video_id,
                'video_name': video.original_name,
                'query': query,
                'search_type': 'intra_video',
                'results': search_results,
                'total_results': len(search_results),
                'analysis_type': 'intra_video_search'
            })
            
        except Exception as e:
            logger.error(f"❌ 영상 내 검색 오류: {e}")
            return Response({'error': str(e)}, status=500)
    
    def _perform_intra_video_search(self, detection_db, meta_db, query, criteria):
        """영상 내 검색 수행"""
        results = []
        query_lower = query.lower()
        
        frames = meta_db.get('frame', [])
        
        for frame in frames:
            timestamp = frame.get('timestamp', 0)
            objects = frame.get('objects', [])
            
            for obj in objects:
                if obj.get('class') == 'person':
                    # 주황색 상의 남성 검색
                    if self._matches_person_criteria(obj, query_lower, criteria):
                        result = {
                            'timestamp': timestamp,
                            'frame_id': frame.get('image_id', 1),
                            'person_id': obj.get('id', 1),
                            'bbox': obj.get('bbox', [0, 0, 0, 0]),
                            'confidence': obj.get('confidence', 0.0),
                            'attributes': obj.get('attributes', {}),
                            'scene_context': obj.get('scene_context', {}),
                            'description': self._generate_person_description(obj, query_lower)
                        }
                        results.append(result)
        
        # 시간순으로 정렬
        results.sort(key=lambda x: x['timestamp'])
        return results
    
    def _matches_person_criteria(self, person_obj, query_lower, criteria):
        """사람 객체가 검색 조건에 맞는지 확인"""
        attributes = person_obj.get('attributes', {})
        clothing = attributes.get('clothing', {})
        
        # 주황색 상의 검색
        if any(keyword in query_lower for keyword in ['주황', 'orange', '주황색']):
            dominant_color = clothing.get('dominant_color', '').lower()
            if 'orange' in dominant_color:
                return True
        
        # 남성 검색
        if any(keyword in query_lower for keyword in ['남성', '남자', 'man', 'male']):
            gender = attributes.get('gender', '').lower()
            if 'man' in gender or 'male' in gender:
                return True
        
        # 나이 검색
        if any(keyword in query_lower for keyword in ['성인', 'adult', '어린이', 'child']):
            age = attributes.get('age', '').lower()
            if 'adult' in query_lower and 'adult' in age:
                return True
            elif 'child' in query_lower and 'child' in age:
                return True
        
        return False
    
    def _generate_person_description(self, person_obj, query_lower):
        """사람 객체 설명 생성"""
        attributes = person_obj.get('attributes', {})
        clothing = attributes.get('clothing', {})
        
        gender = attributes.get('gender', 'unknown')
        age = attributes.get('age', 'unknown')
        dominant_color = clothing.get('dominant_color', 'unknown')
        
        description_parts = []
        
        if '주황' in query_lower or 'orange' in query_lower:
            description_parts.append(f"주황색 옷")
        
        if '남성' in query_lower or '남자' in query_lower:
            description_parts.append(f"{gender}")
        
        if '성인' in query_lower or 'adult' in query_lower:
            description_parts.append(f"{age}")
        
        return f"{', '.join(description_parts)}" if description_parts else "사람"


class TemporalAnalysisView(APIView):
    """시간대별 분석 (3:00-5:00 성비 분포 등)"""
    permission_classes = [AllowAny]
    
    def post(self, request):
        try:
            video_id = request.data.get('video_id')
            time_range = request.data.get('time_range', {})
            analysis_type = request.data.get('analysis_type', 'gender_distribution')
            
            logger.info(f"📊 시간대별 분석 요청: 비디오={video_id}, 범위={time_range}, 타입={analysis_type}")
            
            if not video_id:
                return Response({'error': '비디오 ID가 필요합니다.'}, status=400)
            
            try:
                video = Video.objects.get(id=video_id)
            except Video.DoesNotExist:
                return Response({'error': '비디오를 찾을 수 없습니다.'}, status=404)
            
            # TeletoVision 형식 파일 찾기
            meta_db_path = os.path.join(settings.MEDIA_ROOT, f"{video.original_name}-meta_db.json")
            
            if not os.path.exists(meta_db_path):
                return Response({'error': '분석 결과 파일을 찾을 수 없습니다.'}, status=404)
            
            # Meta DB 읽기
            with open(meta_db_path, 'r', encoding='utf-8') as f:
                meta_db = json.load(f)
            
            # 시간대별 분석 수행
            analysis_result = self._perform_temporal_analysis(meta_db, time_range, analysis_type)
            
            return Response({
                'video_id': video_id,
                'video_name': video.original_name,
                'time_range': time_range,
                'analysis_type': analysis_type,
                'result': analysis_result,
                'analysis_type': 'temporal_analysis'
            })
            
        except Exception as e:
            logger.error(f"❌ 시간대별 분석 오류: {e}")
            return Response({'error': str(e)}, status=500)
    
    def _perform_temporal_analysis(self, meta_db, time_range, analysis_type):
        """시간대별 분석 수행"""
        start_time = time_range.get('start', 0)  # 초 단위
        end_time = time_range.get('end', 0)      # 초 단위
        
        frames = meta_db.get('frame', [])
        
        # 시간 범위 내 프레임 필터링
        filtered_frames = [
            frame for frame in frames
            if start_time <= frame.get('timestamp', 0) <= end_time
        ]
        
        if analysis_type == 'gender_distribution':
            return self._analyze_gender_distribution(filtered_frames)
        elif analysis_type == 'age_distribution':
            return self._analyze_age_distribution(filtered_frames)
        elif analysis_type == 'activity_pattern':
            return self._analyze_activity_pattern(filtered_frames)
        else:
            return {'error': '지원하지 않는 분석 타입입니다.'}
    
    def _analyze_gender_distribution(self, frames):
        """성비 분포 분석"""
        gender_count = {'male': 0, 'female': 0, 'unknown': 0}
        total_persons = 0
        
        for frame in frames:
            objects = frame.get('objects', [])
            for obj in objects:
                if obj.get('class') == 'person':
                    gender = obj.get('attributes', {}).get('gender', 'unknown').lower()
                    if 'man' in gender or 'male' in gender:
                        gender_count['male'] += 1
                    elif 'woman' in gender or 'female' in gender:
                        gender_count['female'] += 1
                    else:
                        gender_count['unknown'] += 1
                    total_persons += 1
        
        # 비율 계산
        if total_persons > 0:
            gender_ratio = {
                'male': round(gender_count['male'] / total_persons * 100, 1),
                'female': round(gender_count['female'] / total_persons * 100, 1),
                'unknown': round(gender_count['unknown'] / total_persons * 100, 1)
            }
        else:
            gender_ratio = {'male': 0, 'female': 0, 'unknown': 0}
        
        return {
            'total_persons': total_persons,
            'gender_count': gender_count,
            'gender_ratio': gender_ratio,
            'analysis_summary': f"총 {total_persons}명 중 남성 {gender_ratio['male']}%, 여성 {gender_ratio['female']}%"
        }
    
    def _analyze_age_distribution(self, frames):
        """나이 분포 분석"""
        age_count = {'child': 0, 'teenager': 0, 'adult': 0, 'elderly': 0, 'unknown': 0}
        total_persons = 0
        
        for frame in frames:
            objects = frame.get('objects', [])
            for obj in objects:
                if obj.get('class') == 'person':
                    age = obj.get('attributes', {}).get('age', 'unknown').lower()
                    if 'child' in age:
                        age_count['child'] += 1
                    elif 'teenager' in age:
                        age_count['teenager'] += 1
                    elif 'adult' in age:
                        age_count['adult'] += 1
                    elif 'elderly' in age:
                        age_count['elderly'] += 1
                    else:
                        age_count['unknown'] += 1
                    total_persons += 1
        
        return {
            'total_persons': total_persons,
            'age_count': age_count,
            'analysis_summary': f"총 {total_persons}명의 나이 분포 분석 완료"
        }
    
    def _analyze_activity_pattern(self, frames):
        """활동 패턴 분석"""
        activity_levels = {'low': 0, 'medium': 0, 'high': 0, 'unknown': 0}
        
        for frame in frames:
            objects = frame.get('objects', [])
            for obj in objects:
                scene_context = obj.get('scene_context', {})
                activity_level = scene_context.get('activity_level', 'unknown').lower()
                if activity_level in activity_levels:
                    activity_levels[activity_level] += 1
        
        return {
            'activity_levels': activity_levels,
            'analysis_summary': f"활동 패턴 분석 완료"
        }
