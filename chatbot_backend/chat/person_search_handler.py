import json
import os
import logging
from django.conf import settings
from chat.models import Video

logger = logging.getLogger(__name__)


def handle_person_search_command(message, video_id):
    """사람 찾기 명령어 처리"""
    try:
        logger.info(f"🔍 사람 찾기 명령어 처리: {message}")
        
        # 분석 결과 JSON 파일 읽기
        video = Video.objects.get(id=video_id)
        if not video.analysis_json_path:
            return "❌ 영상 분석 결과가 없습니다. 먼저 영상을 분석해주세요."
        
        json_path = os.path.join(settings.MEDIA_ROOT, video.analysis_json_path)
        if not os.path.exists(json_path):
            return "❌ 영상 분석 파일을 찾을 수 없습니다."
        
        with open(json_path, 'r', encoding='utf-8') as f:
            analysis_data = json.load(f)
        
        # 사람 관련 정보 추출
        frame_results = analysis_data.get('frame_results', [])
        person_info = []
        
        for frame in frame_results:
            persons = frame.get('persons', [])
            if persons:
                frame_info = {
                    'timestamp': frame.get('timestamp', 0),
                    'person_count': len(persons),
                    'persons': persons,
                    'caption': frame.get('frame_caption', ''),
                    'image_path': frame.get('frame_image_path', '')
                }
                person_info.append(frame_info)
        
        if not person_info:
            return "❌ 영상에서 사람을 찾을 수 없습니다."
        
        # 사람 정보 요약 생성
        total_persons = sum(info['person_count'] for info in person_info)
        unique_timestamps = len(person_info)
        
        result_text = f"👥 **사람 검색 결과**\n\n"
        result_text += f"📊 **전체 통계**\n"
        result_text += f"• 총 감지된 사람 수: {total_persons}명\n"
        result_text += f"• 사람이 등장한 프레임: {unique_timestamps}개\n"
        result_text += f"• 영상 길이: {analysis_data.get('video_summary', {}).get('total_time_span', 0):.1f}초\n\n"
        
        result_text += f"🎬 **주요 장면**\n"
        for i, info in enumerate(person_info[:5], 1):  # 상위 5개만 표시
            result_text += f"{i}. **{info['timestamp']:.1f}초** - {info['person_count']}명\n"
            result_text += f"   - 설명: {info['caption']}\n"
            
            # 사람별 상세 정보
            if info['persons']:
                person_details = []
                for person in info['persons'][:3]:  # 최대 3명만 표시
                    age = person.get('age', '미상')
                    gender = person.get('gender', '미상')
                    clothing = person.get('clothing', {})
                    color = clothing.get('dominant_color', '미상')
                    person_details.append(f"{gender}({age}) - {color} 옷")
                
                if person_details:
                    result_text += f"   - 인물: {', '.join(person_details)}\n"
            
            result_text += "\n"
        
        if len(person_info) > 5:
            result_text += f"... 및 {len(person_info) - 5}개 장면 더\n"
        
        return result_text
        
    except Exception as e:
        logger.error(f"❌ 사람 찾기 명령어 처리 오류: {e}")
        return f"❌ 사람 검색 중 오류가 발생했습니다: {str(e)}"
