# api/views_basic_analysis.py - 기본 분석을 위한 간단한 뷰
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from api.models import Video
from .services.basic_video_analysis import get_basic_video_analysis_service
import threading

@method_decorator(csrf_exempt, name='dispatch')
class BasicAnalyzeVideoView(APIView):
    """기본 비디오 분석 시작"""
    permission_classes = [AllowAny]
    
    def post(self, request, pk):
        try:
            print(f"🔬 기본 비디오 분석 시작: video_id={pk}")
            
            # 비디오 존재 확인
            try:
                video = Video.objects.get(id=pk)
            except Video.DoesNotExist:
                return Response({
                    'error': '해당 비디오를 찾을 수 없습니다.'
                }, status=status.HTTP_404_NOT_FOUND)
            
            # 이미 분석 중인지 확인
            if video.analysis_status == 'processing':
                return Response({
                    'error': '이미 분석이 진행 중입니다.',
                    'current_status': video.analysis_status
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # 이미 분석 완료되었는지 확인
            if video.analysis_status == 'completed':
                return Response({
                    'success': True,
                    'message': '이미 분석이 완료되었습니다.',
                    'analysis_status': 'completed'
                })
            
            # 기본 분석 서비스 가져오기
            basic_service = get_basic_video_analysis_service()
            
            # 분석 상태 업데이트
            video.analysis_status = 'processing'
            video.save()
            
            print(f"✅ 비디오 상태를 'processing'으로 변경: {video.original_name}")
            
            # 백그라운드에서 기본 분석 시작
            analysis_thread = threading.Thread(
                target=self._run_basic_analysis,
                args=(basic_service, video),
                daemon=True
            )
            analysis_thread.start()
            
            print("🧵 기본 분석 스레드 시작됨")
            
            return Response({
                'success': True,
                'message': '기본 분석이 시작되었습니다.',
                'video_id': video.id,
                'analysis_type': 'basic',
                'status': 'processing'
            })
            
        except Exception as e:
            print(f"❌ 기본 분석 시작 오류: {e}")
            return Response({
                'error': f'분석 시작 중 오류가 발생했습니다: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    def _run_basic_analysis(self, basic_service, video):
        """기본 분석 실행"""
        try:
            print(f"🔬 기본 분석 실행 시작: {video.original_name}")
            result = basic_service.analyze_video(video.id, 'basic')
            
            if result['success']:
                print(f"✅ 기본 분석 완료: {video.original_name}")
                print(f"   - 감지된 객체: {result.get('features_detected', 0)}개")
            else:
                print(f"❌ 기본 분석 실패: {result.get('error', '알 수 없는 오류')}")
                
        except Exception as e:
            print(f"❌ 기본 분석 중 예외 발생: {e}")
            # 분석 상태를 실패로 변경
            try:
                video.analysis_status = 'failed'
                video.error_message = f'기본 분석 실패: {str(e)}'
                video.save()
            except:
                pass
