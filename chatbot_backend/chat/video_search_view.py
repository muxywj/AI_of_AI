from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from chat.models import Video
from django.conf import settings
import json
import logging
import os

logger = logging.getLogger(__name__)


class VideoSearchView(APIView):
    """자연어 기반 영상 검색"""
    permission_classes = [AllowAny]
    
    def post(self, request):
        try:
            query = request.data.get('query', '')
            search_type = request.data.get('search_type', 'semantic')
            
            logger.info(f"🔍 영상 검색 요청: 쿼리='{query}', 타입={search_type}")
            
            if not query:
                return Response({'error': '검색 쿼리가 필요합니다.'}, status=400)
            
            videos = Video.objects.filter(analysis_status='completed')
            
            if not videos.exists():
                return Response({
                    'query': query,
                    'search_type': search_type,
                    'results': [],
                    'total_results': 0,
                    'message': '분석 완료된 영상이 없습니다.'
                })
            
            search_results = self._perform_search(videos, query, search_type)
            
            return Response({
                'query': query,
                'search_type': search_type,
                'results': search_results,
                'total_results': len(search_results),
                'analysis_type': 'video_search'
            })
            
        except Exception as e:
            logger.error(f"❌ 영상 검색 오류: {e}")
            return Response({'error': str(e)}, status=500)
    
    def _perform_search(self, videos, query, search_type):
        """실제 검색 수행"""
        results = []
        
        for video in videos:
            try:
                if not video.analysis_json_path:
                    continue
                    
                analysis_file_path = os.path.join(settings.MEDIA_ROOT, video.analysis_json_path)
                if not os.path.exists(analysis_file_path):
                    continue
                
                with open(analysis_file_path, 'r', encoding='utf-8') as f:
                    analysis_data = json.load(f)
                
                relevance_score = self._calculate_relevance_score(analysis_data, query, search_type)
                
                if relevance_score > 0.1:
                    result = {
                        'video_id': video.id,
                        'video_name': video.original_name,
                        'filename': video.filename,
                        'relevance_score': relevance_score,
                        'duration': video.duration,
                        'uploaded_at': video.uploaded_at,
                        'matched_frames': self._find_matching_frames(analysis_data, query),
                        'summary': self._generate_search_summary(analysis_data, query)
                    }
                    results.append(result)
                    
            except Exception as e:
                logger.warning(f"영상 {video.id} 검색 중 오류: {e}")
                continue
        
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        return results[:10]
    
    def _calculate_relevance_score(self, analysis_data, query, search_type):
        """관련도 점수 계산"""
        query_lower = query.lower()
        score = 0.0
        
        detected_objects = analysis_data.get('detected_objects', [])
        for obj in detected_objects:
            if any(keyword in obj.get('class_name', '').lower() for keyword in query_lower.split()):
                score += 0.3
        
        frame_results = analysis_data.get('frame_results', [])
        for frame in frame_results:
            frame_caption = frame.get('frame_caption', '').lower()
            if any(keyword in frame_caption for keyword in query_lower.split()):
                score += 0.2
        
        if any(word in query_lower for word in ['시간', '시', '분', '오전', '오후', '밤', '낮']):
            score += 0.1
        
        colors = ['빨간', '파란', '노란', '초록', '검은', '흰', '주황', '보라']
        if any(color in query_lower for color in colors):
            score += 0.1
        
        return min(score, 1.0)
    
    def _find_matching_frames(self, analysis_data, query):
        """매칭되는 프레임 찾기"""
        query_lower = query.lower()
        matching_frames = []
        
        frame_results = analysis_data.get('frame_results', [])
        for frame in frame_results:
            frame_caption = frame.get('frame_caption', '').lower()
            if any(keyword in frame_caption for keyword in query_lower.split()):
                matching_frames.append({
                    'timestamp': frame.get('timestamp', 0),
                    'caption': frame.get('frame_caption', ''),
                    'image_path': frame.get('frame_image_path', ''),
                    'relevance_score': frame.get('relevance_score', 0)
                })
        
        return matching_frames[:5]
    
    def _generate_search_summary(self, analysis_data, query):
        """검색 결과 요약 생성"""
        video_summary = analysis_data.get('video_summary', {})
        summary_parts = []
        
        if video_summary.get('total_time_span'):
            summary_parts.append(f"영상 길이: {video_summary['total_time_span']:.1f}초")
        
        detected_objects = analysis_data.get('detected_objects', [])
        if detected_objects:
            object_classes = [obj.get('class_name', '') for obj in detected_objects]
            unique_objects = list(set(object_classes))
            summary_parts.append(f"감지된 객체: {', '.join(unique_objects[:5])}")
        
        matching_frames = self._find_matching_frames(analysis_data, query)
        if matching_frames:
            summary_parts.append(f"관련 프레임: {len(matching_frames)}개 발견")
        
        return " | ".join(summary_parts) if summary_parts else "관련 정보 없음"
