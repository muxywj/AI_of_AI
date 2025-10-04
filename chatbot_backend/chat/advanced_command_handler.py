import json
import os
import logging
import re
from django.conf import settings
from chat.models import Video
from .advanced_search_view import InterVideoSearchView, IntraVideoSearchView, TemporalAnalysisView

logger = logging.getLogger(__name__)


def handle_inter_video_search_command(message, video_id):
    """영상 간 검색 명령어 처리"""
    try:
        logger.info(f"🔍 영상 간 검색 명령어 처리: {message}")
        
        # InterVideoSearchView 인스턴스 생성
        search_view = InterVideoSearchView()
        
        # 검색 수행
        request_data = {
            'query': message,
            'criteria': {}
        }
        
        # Mock request 객체 생성
        class MockRequest:
            def __init__(self, data):
                self.data = data
        
        mock_request = MockRequest(request_data)
        response = search_view.post(mock_request)
        
        if response.status_code == 200:
            results = response.data.get('results', [])
            
            if results:
                result_text = f"🌧️ **영상 간 검색 결과**\n\n"
                result_text += f"📊 **검색된 영상: {len(results)}개**\n\n"
                
                for i, result in enumerate(results[:3], 1):  # 상위 3개만 표시
                    result_text += f"{i}. **{result['video_name']}**\n"
                    result_text += f"   - 관련도: {result['relevance_score']:.2f}\n"
                    result_text += f"   - 요약: {result['summary']}\n"
                    result_text += f"   - 매칭 장면: {len(result['matched_scenes'])}개\n\n"
                
                if len(results) > 3:
                    result_text += f"... 및 {len(results) - 3}개 영상 더\n"
                
                return result_text
            else:
                return "❌ 조건에 맞는 영상을 찾을 수 없습니다."
        else:
            return f"❌ 영상 간 검색 중 오류가 발생했습니다: {response.data.get('error', '알 수 없는 오류')}"
            
    except Exception as e:
        logger.error(f"❌ 영상 간 검색 명령어 처리 오류: {e}")
        return f"❌ 영상 간 검색 중 오류가 발생했습니다: {str(e)}"


def handle_intra_video_search_command(message, video_id):
    """영상 내 검색 명령어 처리"""
    try:
        logger.info(f"🔍 영상 내 검색 명령어 처리: {message}")
        
        # IntraVideoSearchView 인스턴스 생성
        search_view = IntraVideoSearchView()
        
        # 검색 수행
        request_data = {
            'video_id': video_id,
            'query': message,
            'criteria': {}
        }
        
        # Mock request 객체 생성
        class MockRequest:
            def __init__(self, data):
                self.data = data
        
        mock_request = MockRequest(request_data)
        response = search_view.post(mock_request)
        
        if response.status_code == 200:
            results = response.data.get('results', [])
            
            if results:
                result_text = f"🧡 **영상 내 검색 결과**\n\n"
                result_text += f"📊 **검색된 장면: {len(results)}개**\n\n"
                
                for i, result in enumerate(results[:5], 1):  # 상위 5개만 표시
                    result_text += f"{i}. **{result['timestamp']:.1f}초**\n"
                    result_text += f"   - 설명: {result['description']}\n"
                    result_text += f"   - 신뢰도: {result['confidence']:.2f}\n"
                    result_text += f"   - 위치: {result['bbox']}\n\n"
                
                if len(results) > 5:
                    result_text += f"... 및 {len(results) - 5}개 장면 더\n"
                
                return result_text
            else:
                return "❌ 조건에 맞는 장면을 찾을 수 없습니다."
        else:
            return f"❌ 영상 내 검색 중 오류가 발생했습니다: {response.data.get('error', '알 수 없는 오류')}"
            
    except Exception as e:
        logger.error(f"❌ 영상 내 검색 명령어 처리 오류: {e}")
        return f"❌ 영상 내 검색 중 오류가 발생했습니다: {str(e)}"


def handle_temporal_analysis_command(message, video_id):
    """시간대별 분석 명령어 처리"""
    try:
        logger.info(f"📊 시간대별 분석 명령어 처리: {message}")
        
        # TemporalAnalysisView 인스턴스 생성
        analysis_view = TemporalAnalysisView()
        
        # 시간 범위 추출 (기본값: 3:00-5:00)
        time_range = {'start': 180, 'end': 300}  # 3분-5분 (초 단위)
        
        # 메시지에서 시간 추출
        time_pattern = r'(\d+):(\d+).*?(\d+):(\d+)'
        match = re.search(time_pattern, message)
        if match:
            start_hour, start_min, end_hour, end_min = map(int, match.groups())
            time_range = {
                'start': start_hour * 3600 + start_min * 60,
                'end': end_hour * 3600 + end_min * 60
            }
        
        # 분석 수행
        request_data = {
            'video_id': video_id,
            'time_range': time_range,
            'analysis_type': 'gender_distribution'
        }
        
        # Mock request 객체 생성
        class MockRequest:
            def __init__(self, data):
                self.data = data
        
        mock_request = MockRequest(request_data)
        response = analysis_view.post(mock_request)
        
        if response.status_code == 200:
            result = response.data.get('result', {})
            
            result_text = f"📊 **시간대별 성비 분석 결과**\n\n"
            result_text += f"⏰ **분석 시간대**: {time_range['start']//60}분 - {time_range['end']//60}분\n"
            result_text += f"👥 **총 인원**: {result.get('total_persons', 0)}명\n\n"
            
            gender_ratio = result.get('gender_ratio', {})
            result_text += f"📈 **성별 분포**\n"
            result_text += f"• 남성: {gender_ratio.get('male', 0)}%\n"
            result_text += f"• 여성: {gender_ratio.get('female', 0)}%\n"
            result_text += f"• 미상: {gender_ratio.get('unknown', 0)}%\n\n"
            
            result_text += f"💡 **분석 요약**: {result.get('analysis_summary', '')}\n"
            
            return result_text
        else:
            return f"❌ 시간대별 분석 중 오류가 발생했습니다: {response.data.get('error', '알 수 없는 오류')}"
            
    except Exception as e:
        logger.error(f"❌ 시간대별 분석 명령어 처리 오류: {e}")
        return f"❌ 시간대별 분석 중 오류가 발생했습니다: {str(e)}"
