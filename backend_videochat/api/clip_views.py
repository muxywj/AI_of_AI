from rest_framework.views import APIView
from rest_framework.response import Response
from django.http import JsonResponse, HttpResponse
from .models import Video, Frame
import os
from django.conf import settings


class ClipPreviewView(APIView):
    """비디오 클립 미리보기 뷰"""
    
    def get(self, request, video_id, timestamp):
        try:
            # 비디오 존재 확인
            video = Video.objects.get(id=video_id)
            
            # timestamp를 float로 변환
            timestamp = float(timestamp)
            
            # 해당 시점에 가장 가까운 프레임 찾기
            frame = Frame.objects.filter(
                video=video,
                timestamp__lte=timestamp
            ).order_by('-timestamp').first()
            
            if not frame:
                return JsonResponse({'error': '해당 시점의 프레임을 찾을 수 없습니다'}, status=404)
            
            # 프레임 이미지 파일 경로
            frame_image_path = os.path.join(
                settings.MEDIA_ROOT, 
                'images', 
                f'video{video_id}_frame{frame.image_id}.jpg'
            )
            
            if not os.path.exists(frame_image_path):
                return JsonResponse({'error': '프레임 이미지 파일을 찾을 수 없습니다'}, status=404)
            
            # 이미지 파일을 HTTP 응답으로 반환
            with open(frame_image_path, 'rb') as f:
                response = HttpResponse(f.read(), content_type='image/jpeg')
                response['Content-Disposition'] = f'inline; filename="clip_video{video_id}_frame{frame.image_id}.jpg"'
                response['X-Timestamp'] = str(timestamp)
                response['X-Frame-ID'] = str(frame.image_id)
                response['X-Duration'] = str(request.GET.get('duration', 4))
                return response
            
        except Video.DoesNotExist:
            return JsonResponse({'error': '비디오를 찾을 수 없습니다'}, status=404)
        except Exception as e:
            return JsonResponse({'error': f'클립 생성 실패: {str(e)}'}, status=500)
    
    def _find_video_file(self, video):
        """비디오 파일 경로 찾기"""
        # 여러 가능한 파일명으로 시도
        possible_names = [
            video.filename,
            video.original_name,
        ]
        
        uploads_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
        
        for name in possible_names:
            if name:
                # 정확한 파일명으로 찾기
                file_path = os.path.join(uploads_dir, name)
                if os.path.exists(file_path):
                    return file_path
                
                # 부분 매칭으로 찾기 (original_name이 포함된 파일)
                if video.original_name:
                    for filename in os.listdir(uploads_dir):
                        if video.original_name in filename:
                            return os.path.join(uploads_dir, filename)
        
        return None
