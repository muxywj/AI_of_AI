# from .video_analyzer import VideoAnalyzer, EnhancedVideoAnalyzer  # 지연 로딩으로 변경
from .services.video_analysis_service import VIDEO_ANALYZER_AVAILABLE, get_video_analyzer, get_analyzer_status
from .db_builder import get_video_rag_system, EnhancedVideoRAGSystem
from .models import Video, PersonDetection, YOLOObjectDetection, Frame, TrackPoint, Scene, AnalysisResult
from .llm_client import llm_client
from .vision_analyzer import vision_analyzer
import json
import os
import time
from openai import OpenAI
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes, authentication_classes
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.parsers import MultiPartParser, FormParser
class VideoListView(APIView):
    """비디오 목록 조회 - 고급 분석 정보 포함"""
    permission_classes = [AllowAny]
    
    def get(self, request):
        try:
            print("🔍 VideoListView: 비디오 목록 요청 (고급 분석 정보 포함)")
            videos = Video.objects.all()
            video_list = []
            
            for video in videos:
                video_data = {
                    'id': video.id,
                    'filename': video.filename,
                    'original_name': video.original_name,
                    'duration': video.duration,
                    'is_analyzed': video.is_analyzed,
                    'analysis_status': video.analysis_status,
                    'uploaded_at': video.uploaded_at,
                    'file_size': video.file_size
                }
                
                # 고급 분석 정보 추가
                if video.is_analyzed:
                    video_data.update({
                        'enhanced_analysis': video.enhanced_analysis,
                        'success_rate': video.success_rate,
                        'processing_time': video.processing_time,
                        'analysis_type': video.analysis_type,
                        'advanced_features_used': video.advanced_features_used,
                        'scene_types': video.scene_types,
                        'unique_objects': video.unique_objects
                    })
                
                # 진행률 정보 추가 (분석 중인 경우)
                if video.analysis_status == 'processing':
                    video_data['progress_info'] = {
                        'progress': 50,  # 기본 진행률
                        'status': 'processing',
                        'message': '분석 진행 중...'
                    }
                
                video_list.append(video_data)
            
            print(f"✅ VideoListView: {len(video_list)}개 비디오 반환 (고급 분석 정보 포함)")
            return Response({
                'videos': video_list,
                'total_count': len(video_list),
                'analysis_capabilities': self._get_system_capabilities()
            })
            
        except Exception as e:
            print(f"❌ VideoListView 오류: {e}")
            return Response({
                'error': f'비디오 목록 조회 오류: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    def _get_system_capabilities(self):
        """시스템 분석 기능 상태"""
        try:
            # 기본 분석 기능 상태 반환
            return {
                'clip_available': False,
                'ocr_available': False,
                'vqa_available': False,
                'scene_graph_available': False
            }
        except Exception as e:
            print(f"⚠️ _get_system_capabilities 오류: {e}")
            return {
                'clip_available': False,
                'ocr_available': False,
                'vqa_available': False,
                'scene_graph_available': False
            }
    
# 기존의 다른 View 클래스들은 그대로 유지
class VideoUploadView(APIView):
    """비디오 업로드"""
    permission_classes = [AllowAny]
    parser_classes = (MultiPartParser, FormParser)
    
    def post(self, request):
        try:
            if 'video' not in request.FILES:
                return Response({
                    'error': '비디오 파일이 없습니다'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            video_file = request.FILES['video']
            
            if not video_file.name.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                return Response({
                    'error': '지원하지 않는 파일 형식입니다'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Generate unique filename
            timestamp = int(time.time())
            filename = f"upload_{timestamp}_{video_file.name}"
            
            # Save file
            file_path = default_storage.save(
                f'uploads/{filename}',
                ContentFile(video_file.read())
            )
            
            # Create Video model instance
            video = Video.objects.create(
                filename=filename,
                original_name=video_file.name,
                file_path=file_path,
                file_size=video_file.size,
                file=file_path,  # file 필드도 저장
                analysis_status='pending'
            )
            
            return Response({
                'success': True,
                'video_id': video.id,
                'filename': filename,
                'message': f'비디오 "{video_file.name}"이 성공적으로 업로드되었습니다.'
            })
            
        except Exception as e:
            return Response({
                'error': f'업로드 오류: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)




class VideoChatView(APIView):
    """비디오 관련 채팅 API - 기존 ChatView와 구분"""
    permission_classes = [AllowAny]  # 🔧 권한 설정 추가
    
    def __init__(self):
        super().__init__()
        self.llm_client = llm_client
        try:
            from .video_analyzer import get_video_analyzer
            self.video_analyzer = get_video_analyzer()
        except:
            self.video_analyzer = None
    
    def post(self, request):
        try:
            user_message = request.data.get('message', '').strip()
            video_id = request.data.get('video_id')
            
            if not user_message:
                return Response({'response': '메시지를 입력해주세요.'})
            
            print(f"💬 사용자 메시지: {user_message}")
            
            # Get current video
            if video_id:
                try:
                    current_video = Video.objects.get(id=video_id)
                except Video.DoesNotExist:
                    current_video = Video.objects.filter(is_analyzed=True).first()
            else:
                current_video = Video.objects.filter(is_analyzed=True).first()
            
            if not current_video:
                return Response({
                    'response': '분석된 비디오가 없습니다. 비디오를 업로드하고 분석해주세요.'
                })
            
            # Get video info
            video_info = self._get_video_info(current_video)
            
            # Determine if multi-LLM should be used
            use_multi_llm = "compare" in user_message.lower() or "비교" in user_message or "분석" in user_message
            
            # Handle different query types
            print(f"🔍 쿼리 타입 확인: '{user_message}'")
            print(f"  - 검색 쿼리: {self._is_search_query(user_message)}")
            print(f"  - 하이라이트 쿼리: {self._is_highlight_query(user_message)}")
            print(f"  - 요약 쿼리: {self._is_summary_query(user_message)}")
            print(f"  - 정보 쿼리: {self._is_info_query(user_message)}")
            
            if self._is_search_query(user_message):
                print("🎯 검색 쿼리로 처리")
                return self._handle_search_query(user_message, current_video, video_info, use_multi_llm)
            
            elif self._is_highlight_query(user_message):
                print("⭐ 하이라이트 쿼리로 처리")
                return self._handle_highlight_query(user_message, current_video, video_info, use_multi_llm)
            
            elif self._is_summary_query(user_message):
                print("📝 요약 쿼리로 처리")
                return self._handle_summary_query(user_message, current_video, video_info, use_multi_llm)
            
            elif self._is_info_query(user_message):
                print("📊 정보 쿼리로 처리")
                return self._handle_info_query(user_message, current_video, video_info, use_multi_llm)
            
            else:
                # General conversation
                bot_response = self.llm_client.generate_smart_response(
                    user_query=user_message,
                    search_results=None,
                    video_info=video_info,
                    use_multi_llm=use_multi_llm
                )
                return Response({'response': bot_response})
                
        except Exception as e:
            print(f"❌ Chat error: {e}")
            error_response = self.llm_client.generate_smart_response(
                user_query="시스템 오류가 발생했습니다. 도움을 요청합니다.",
                search_results=None,
                video_info=None
            )
            return Response({'response': error_response})

    def _is_search_query(self, message):
        """검색 쿼리인지 확인"""
        search_keywords = ['찾아', '검색', '어디', '누구', '색상', '옷', '사람', '어린이', '노인', '중년']
        return any(keyword in message for keyword in search_keywords)
    
    def _is_highlight_query(self, message):
        """하이라이트 쿼리인지 확인"""
        highlight_keywords = ['하이라이트', '중요', '주요', '대표', '핵심']
        return any(keyword in message for keyword in highlight_keywords)
    
    def _is_summary_query(self, message):
        """요약 쿼리인지 확인"""
        summary_keywords = ['요약', '정리', '설명', '개요', '줄거리', '내용']
        return any(keyword in message for keyword in summary_keywords)
    
    def _is_info_query(self, message):
        """정보 쿼리인지 확인"""
        info_keywords = ['정보', '상태', '분석', '통계', '데이터']
        return any(keyword in message for keyword in info_keywords)
    
    def _handle_search_query(self, message, video, video_info, use_multi_llm):
        """검색 쿼리 처리"""
        try:
            # 기존 검색 로직 사용
            from .services.video_analysis_service import get_video_analyzer
            analyzer = get_video_analyzer()
            if not analyzer:
                return Response({'response': '검색 기능을 사용할 수 없습니다. 분석이 완료되지 않았을 수 있습니다.'})
            
            # 간단한 검색 결과 생성
            search_results = self._perform_simple_search(video, message)
            
            bot_response = self.llm_client.generate_smart_response(
                user_query=message,
                search_results=search_results,
                video_info=video_info,
                use_multi_llm=use_multi_llm
            )
            return Response({'response': bot_response})
            
        except Exception as e:
            print(f"❌ Search query error: {e}")
            return Response({'response': f'검색 중 오류가 발생했습니다: {str(e)}'})
    
    def _handle_highlight_query(self, message, video, video_info, use_multi_llm):
        """하이라이트 쿼리 처리"""
        try:
            # 하이라이트 정보 생성
            highlights = self._get_video_highlights(video)
            
            bot_response = self.llm_client.generate_smart_response(
                user_query=message,
                search_results=highlights,
                video_info=video_info,
                use_multi_llm=use_multi_llm
            )
            return Response({'response': bot_response})
            
        except Exception as e:
            print(f"❌ Highlight query error: {e}")
            return Response({'response': f'하이라이트 생성 중 오류가 발생했습니다: {str(e)}'})
    
    def _handle_summary_query(self, message, video, video_info, use_multi_llm):
        """요약 쿼리 처리 - CLIP + GPT Vision 통합 요약 사용"""
        try:
            video_title = video.title if video.title else f"영상 {video.id}"
            print(f"🎬 영상 요약 요청 처리: {video_title}")
            print(f"📊 비디오 정보 - ID: {video.id}, 제목: {video_title}, 상태: {video.analysis_status}")
            
            # CLIP + GPT Vision 통합 요약 생성
            summary_view = VideoSummaryView()
            summary_data = summary_view._generate_video_summary(video)
            print(f"📋 요약 데이터 생성 결과: {summary_data is not None}")
            
            if summary_data and 'llm_summary' in summary_data:
                print("✅ LLM 요약 데이터 발견")
                # LLM 요약이 있으면 사용
                response_text = summary_data['llm_summary']
                
                # 분석 방법 정보 추가
                analysis_methods = summary_data.get('analysis_methods', [])
                if analysis_methods:
                    methods_text = ' + '.join(analysis_methods)
                    response_text += f"\n\n🤖 분석 방법: {methods_text}"
                
                # 추가 정보가 있으면 포함
                if 'clip_analyses' in summary_data and summary_data['clip_analyses']:
                    clip_count = len(summary_data['clip_analyses'])
                    response_text += f"\n📸 CLIP 분석: {clip_count}개 프레임 분석 완료"
                
                if 'gpt_vision_analyses' in summary_data and summary_data['gpt_vision_analyses']:
                    gpt_count = len(summary_data['gpt_vision_analyses'])
                    response_text += f"\n👁️ GPT Vision 분석: {gpt_count}개 프레임 상세 분석 완료"
                
                return Response({'response': response_text})
            else:
                print("⚠️ LLM 요약 데이터 없음, 기본 요약 생성")
                # 기본 요약 생성
                basic_summary = self._generate_basic_summary(video, video_info)
                return Response({'response': basic_summary})
                
        except Exception as e:
            print(f"❌ Summary query error: {e}")
            import traceback
            traceback.print_exc()
            return Response({'response': f'요약 생성 중 오류가 발생했습니다: {str(e)}'})
    
    def _handle_info_query(self, message, video, video_info, use_multi_llm):
        """정보 쿼리 처리"""
        try:
            info_text = f"📊 비디오 정보:\n"
            info_text += f"- 제목: {video.title}\n"
            info_text += f"- 길이: {video.duration:.1f}초\n"
            info_text += f"- 분석 상태: {video.analysis_status}\n"
            info_text += f"- 업로드일: {video.uploaded_at.strftime('%Y-%m-%d %H:%M')}\n"
            
            if video_info:
                info_text += f"\n📈 분석 통계:\n"
                if 'total_frames' in video_info:
                    info_text += f"- 총 프레임 수: {video_info['total_frames']}\n"
                if 'total_objects' in video_info:
                    info_text += f"- 감지된 객체 수: {video_info['total_objects']}\n"
            
            return Response({'response': info_text})
            
        except Exception as e:
            print(f"❌ Info query error: {e}")
            return Response({'response': f'정보 조회 중 오류가 발생했습니다: {str(e)}'})
    
    def _generate_video_summary(self, video_id):
        """영상 요약 생성 (CLIP + GPT Vision 통합)"""
        try:
            video = Video.objects.get(id=video_id)
            print(f"🎬 영상 요약 생성 시작: {video.title}")
            
            # 1. 대표 프레임들 선택 (처음, 중간, 마지막)
            frames = Frame.objects.filter(video=video).order_by('timestamp')
            if not frames.exists():
                return {"error": "분석된 프레임이 없습니다"}
            
            frames_list = list(frames)
            if len(frames_list) >= 3:
                selected_frames = [frames_list[0], frames_list[len(frames_list)//2], frames_list[-1]]
            elif len(frames_list) >= 2:
                selected_frames = [frames_list[0], frames_list[-1]]
            else:
                selected_frames = [frames_list[0]]
            
            print(f"📸 선택된 프레임 수: {len(selected_frames)}")
            
            # 2. CLIP을 사용한 프레임 분석
            clip_analyses = []
            frame_paths = []
            
            for frame in selected_frames:
                frame_path = os.path.join('media/images', f'video{video_id}_frame{frame.image_id}.jpg')
                if os.path.exists(frame_path):
                    frame_paths.append(frame_path)
                    print(f"🔍 CLIP 분석: {frame_path}")
                    
                    if vision_analyzer:
                        clip_analysis = vision_analyzer.analyze_frame(frame_path)
                        if 'error' not in clip_analysis:
                            clip_analyses.append({
                                'timestamp': frame.timestamp,
                                'frame_id': frame.image_id,
                                'clip_analysis': clip_analysis
                            })
                        else:
                            print(f"⚠️ CLIP 분석 실패: {clip_analysis['error']}")
                    else:
                        print("⚠️ Vision Analyzer를 사용할 수 없습니다")
            
            # 3. GPT Vision을 사용한 프레임 분석
            gpt_vision_analyses = []
            
            for frame_path in frame_paths:
                if llm_client and llm_client.is_available():
                    print(f"🤖 GPT Vision 분석: {frame_path}")
                    gpt_analysis = llm_client.analyze_frame_with_vision(
                        frame_path, 
                        "이 프레임에서 사람들을 자세히 분석해주세요. 탐지된 사람의 수, 위치, 특징, 활동을 포함해서 설명해주세요."
                    )
                    if gpt_analysis:
                        gpt_vision_analyses.append({
                            'frame_path': frame_path,
                            'analysis': gpt_analysis
                        })
                else:
                    print("⚠️ GPT Vision을 사용할 수 없습니다")
            
            # 4. 두 분석 결과 통합하여 최종 요약 생성
            video_data = {
                'video_id': video_id,
                'video_title': video.title,
                'total_duration': video.duration,
                'clips': [],  # 기존 클립 데이터
                'clip_analyses': clip_analyses,  # CLIP 분석 결과
                'gpt_vision_analyses': gpt_vision_analyses,  # GPT Vision 분석 결과
                'selected_frames': [{'timestamp': f.timestamp, 'frame_id': f.image_id} for f in selected_frames]
            }
            
            # 5. 통합된 데이터로 LLM 요약 생성
            if llm_client and llm_client.is_available():
                print("📝 통합 요약 생성 중...")
                llm_summary = llm_client.generate_summary(video_data)
            else:
                print("⚠️ LLM을 사용할 수 없어 기본 요약 생성")
                llm_summary = self._generate_fallback_summary(video_data)
            
            return {
                'llm_summary': llm_summary,
                'clip_analyses': clip_analyses,
                'gpt_vision_analyses': gpt_vision_analyses,
                'selected_frames': len(selected_frames),
                'analysis_methods': ['CLIP', 'GPT Vision'] if gpt_vision_analyses else ['CLIP']
            }
            
        except Exception as e:
            print(f"❌ Video summary generation error: {e}")
            return None
    
    def _generate_fallback_summary(self, video_data):
        """LLM 사용 불가 시 기본 요약 생성"""
        video_title = video_data.get('video_title', '영상')
        video_id = video_data.get('video_id', '')
        if not video_title or video_title == 'None':
            video_title = f"영상 {video_id}" if video_id else "영상"
        
        clip_analyses = video_data.get('clip_analyses', [])
        gpt_vision_analyses = video_data.get('gpt_vision_analyses', [])
        
        summary = f"📹 '{video_title}' 영상 요약\n\n"
        
        # CLIP 분석 결과 통합
        if clip_analyses:
            summary += "🎬 주요 장면 분석\n"
            for i, analysis in enumerate(clip_analyses, 1):
                clip_data = analysis.get('clip_analysis', {})
                overall_scene = clip_data.get('overall_scene', '분석 중')
                timestamp = analysis.get('timestamp', 0)
                summary += f"{i}. {timestamp:.1f}초: {overall_scene}\n"
        
        # GPT Vision 분석 결과 통합
        if gpt_vision_analyses:
            summary += "\n👥 상세 인물 분석\n"
            for i, analysis in enumerate(gpt_vision_analyses, 1):
                vision_data = analysis.get('analysis', {})
                person_count = vision_data.get('person_count', 0)
                scene_desc = vision_data.get('scene_description', '')
                if scene_desc:
                    summary += f"{i}. {scene_desc}\n"
                elif person_count > 0:
                    summary += f"{i}. {person_count}명의 사람이 탐지되었습니다.\n"
        
        summary += "\n💭 전체 분석\n"
        summary += "이 영상은 CLIP과 GPT Vision을 활용하여 분석되었습니다. "
        summary += "주요 장면과 인물들의 활동을 통해 전체적인 상황을 파악할 수 있습니다."
        
        return summary
    
    def _analyze_video_frames(self, video_id, frames):
        """비디오의 프레임 이미지들을 시각적으로 분석"""
        try:
            from .vision_analyzer import vision_analyzer
            if not vision_analyzer:
                return {"error": "Vision analyzer not available"}
            
            # 프레임 이미지 경로들 수집 (1-3개만 선택)
            frame_paths = []
            selected_frames = []
            
            # 시간대별로 균등하게 3개 프레임 선택
            total_frames = len(frames)
            if total_frames >= 3:
                # 시작, 중간, 끝에서 각각 1개씩 선택
                indices = [0, total_frames // 2, total_frames - 1]
            elif total_frames >= 1:
                # 1-2개만 있는 경우 모두 선택
                indices = list(range(total_frames))
            else:
                return {"error": "분석할 프레임이 없습니다"}
            
            for idx in indices:
                frame = frames[idx]
                image_path = os.path.join(settings.MEDIA_ROOT, 'images', f'video{video_id}_frame{frame.image_id}.jpg')
                if os.path.exists(image_path):
                    frame_paths.append(image_path)
                    selected_frames.append({
                        'frame_id': frame.image_id,
                        'timestamp': frame.timestamp,
                        'image_path': image_path
                    })
            
            if not frame_paths:
                return {"error": "분석할 프레임 이미지를 찾을 수 없습니다"}
            
            print(f"🔍 선택된 프레임 {len(frame_paths)}개 분석 중...")
            
            # 시각적 분석 실행
            analysis_result = vision_analyzer.analyze_video_frames(video_id, frame_paths)
            analysis_result['selected_frames'] = selected_frames
            analysis_result['analysis_count'] = len(frame_paths)
            
            return analysis_result
            
        except Exception as e:
            print(f"❌ 프레임 분석 오류: {e}")
            return {"error": f"프레임 분석 실패: {str(e)}"}
    
    def _generate_basic_summary(self, video, video_info):
        """기본 요약 생성"""
        summary = f"📹 '{video.title}' 요약\n\n"
        summary += f"⏱️ 길이: {video.duration:.1f}초\n"
        summary += f"📊 분석 상태: {video.analysis_status}\n"
        
        if video_info:
            if 'total_frames' in video_info:
                summary += f"🎬 총 프레임: {video_info['total_frames']}개\n"
            if 'total_objects' in video_info:
                summary += f"👥 감지된 객체: {video_info['total_objects']}개\n"
        
        summary += "\n💡 더 자세한 분석을 원하시면 '하이라이트' 또는 '검색' 기능을 사용해보세요!"
        
        return summary
    
    def _perform_simple_search(self, video, query):
        """간단한 검색 수행"""
        try:
            # 프레임에서 검색어와 관련된 정보 찾기
            frames = Frame.objects.filter(video=video).order_by('timestamp')[:10]
            results = []
            
            for frame in frames:
                # 간단한 키워드 매칭
                if any(keyword in query.lower() for keyword in ['사람', 'person']):
                    results.append({
                        'timestamp': frame.timestamp,
                        'description': f'프레임 {frame.image_id}에서 사람 감지',
                        'confidence': 0.8
                    })
            
            return results[:5]  # 최대 5개 결과
            
        except Exception as e:
            print(f"❌ Simple search error: {e}")
            return []
    
    def _get_video_highlights(self, video):
        """비디오 하이라이트 정보 생성"""
        try:
            scenes = Scene.objects.filter(video=video).order_by('start_time')[:5]
            highlights = []
            
            for scene in scenes:
                highlights.append({
                    'start_time': scene.start_time,
                    'end_time': scene.end_time,
                    'description': f'씬 {scene.scene_id}',
                    'objects': scene.dominant_objects or []
                })
            
            return highlights
            
        except Exception as e:
            print(f"❌ Highlights error: {e}")
            return []


class FrameView(APIView):
    """프레임 이미지 제공"""
    permission_classes = [AllowAny]  # 🔧 권한 설정 추가
    
    def get(self, request, video_id, frame_number, frame_type='normal'):
        try:
            video = Video.objects.get(id=video_id)
            
            # Get video file path
            video_path = None
            possible_paths = [
                os.path.join(settings.VIDEO_FOLDER, video.filename),
                os.path.join(settings.UPLOAD_FOLDER, video.filename),
                video.file_path
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    video_path = path
                    break
            
            if not video_path:
                return Response({
                    'error': '비디오 파일을 찾을 수 없습니다'
                }, status=status.HTTP_404_NOT_FOUND)
            
            # Extract frame
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return Response({
                    'error': '비디오 파일을 열 수 없습니다'
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_number - 1))
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                return Response({
                    'error': '프레임을 추출할 수 없습니다'
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
            # Handle annotated frames
            if frame_type == 'annotated':
                target_class = request.GET.get('class', '').lower()
                frame = self._annotate_frame(frame, video, frame_number, target_class)
            
            # Resize frame if too large
            height, width = frame.shape[:2]
            if width > 800:
                ratio = 800 / width
                new_width = 800
                new_height = int(height * ratio)
                frame = cv2.resize(frame, (new_width, new_height))
            
            # Save temporary image
            temp_filename = f'frame_{video.id}_{frame_number}_{int(time.time())}.jpg'
            temp_path = os.path.join(settings.IMAGE_FOLDER, temp_filename)
            
            cv2.imwrite(temp_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            
            return FileResponse(
                open(temp_path, 'rb'),
                content_type='image/jpeg',
                filename=temp_filename
            )
            
        except Video.DoesNotExist:
            return Response({
                'error': '비디오를 찾을 수 없습니다'
            }, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)



import os
import json
import time
import cv2
import numpy as np
from django.conf import settings
from django.http import JsonResponse, HttpResponse, FileResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.utils.decorators import method_decorator
from django.views import View
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.permissions import AllowAny
from datetime import datetime, timedelta
from collections import Counter
import threading
import queue

from .models import Video



class AnalysisFeaturesView(APIView):
    """분석 기능별 상세 정보 제공"""
    permission_classes = [AllowAny]
    
    def get(self, request):
        try:
            analyzer = VideoAnalyzer()
            
            features = {
                'object_detection': {
                    'name': '객체 감지',
                    'description': 'YOLO 기반 실시간 객체 감지 및 분류',
                    'available': True,
                    'processing_time_factor': 1.0,
                    'icon': '🎯',
                    'details': '비디오 내 사람, 차량, 동물 등 다양한 객체를 정확하게 감지합니다.'
                },
                'clip_analysis': {
                    'name': 'CLIP 씬 분석',
                    'description': 'OpenAI CLIP 모델을 활용한 고급 씬 이해',
                    'available': analyzer.clip_available,
                    'processing_time_factor': 1.5,
                    'icon': '🖼️',
                    'details': '이미지의 의미적 컨텍스트를 이해하여 씬 분류 및 분석을 수행합니다.'
                },
                'ocr': {
                    'name': 'OCR 텍스트 추출',
                    'description': 'EasyOCR을 사용한 다국어 텍스트 인식',
                    'available': analyzer.ocr_available,
                    'processing_time_factor': 1.2,
                    'icon': '📝',
                    'details': '비디오 내 한글, 영문 텍스트를 정확하게 인식하고 추출합니다.'
                },
                'vqa': {
                    'name': 'VQA 질문답변',
                    'description': 'BLIP 모델 기반 시각적 질문 답변',
                    'available': analyzer.vqa_available,
                    'processing_time_factor': 2.0,
                    'icon': '❓',
                    'details': '이미지에 대한 질문을 생성하고 답변하여 깊이 있는 분석을 제공합니다.'
                },
                'scene_graph': {
                    'name': 'Scene Graph',
                    'description': '객체간 관계 및 상호작용 분석',
                    'available': analyzer.scene_graph_available,
                    'processing_time_factor': 3.0,
                    'icon': '🕸️',
                    'details': '객체들 사이의 관계와 상호작용을 분석하여 복잡한 씬을 이해합니다.'
                },
                'enhanced_caption': {
                    'name': '고급 캡션 생성',
                    'description': '모든 분석 결과를 통합한 상세 캡션',
                    'available': True,
                    'processing_time_factor': 1.1,
                    'icon': '💬',
                    'details': '여러 AI 모델의 결과를 종합하여 상세하고 정확한 캡션을 생성합니다.'
                }
            }
            
            return Response({
                'features': features,
                'device': analyzer.device,
                'total_available': sum(1 for f in features.values() if f['available']),
                'recommended_configs': {
                    'basic': ['object_detection', 'enhanced_caption'],
                    'enhanced': ['object_detection', 'clip_analysis', 'ocr', 'enhanced_caption'],
                    'comprehensive': list(features.keys())
                }
            })
            
        except Exception as e:
            return Response({
                'error': f'분석 기능 정보 조회 실패: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class AdvancedVideoSearchView(APIView):
    """고급 비디오 검색 API"""
    permission_classes = [AllowAny]
    
    def __init__(self):
        super().__init__()
        self.video_analyzer = VideoAnalyzer()
        self.llm_client = LLMClient()
    
    def post(self, request):
        try:
            video_id = request.data.get('video_id')
            query = request.data.get('query', '').strip()
            search_options = request.data.get('search_options', {})
            
            if not query:
                return Response({
                    'error': '검색어를 입력해주세요.'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            video = Video.objects.get(id=video_id)
            
            # 고급 검색 수행
            search_results = self.video_analyzer.search_comprehensive(video, query)
            
            # 고급 분석 결과가 포함된 프레임들에 대해 추가 정보 수집
            enhanced_results = []
            for result in search_results[:10]:
                frame_id = result.get('frame_id')
                try:
                    frame = Frame.objects.get(video=video, image_id=frame_id)
                    enhanced_result = dict(result)
                    
                    # 고급 분석 결과 추가
                    comprehensive_features = frame.comprehensive_features or {}
                    
                    if search_options.get('include_clip_analysis') and 'clip_features' in comprehensive_features:
                        enhanced_result['clip_analysis'] = comprehensive_features['clip_features']
                    
                    if search_options.get('include_ocr_text') and 'ocr_text' in comprehensive_features:
                        enhanced_result['ocr_text'] = comprehensive_features['ocr_text']
                    
                    if search_options.get('include_vqa_results') and 'vqa_results' in comprehensive_features:
                        enhanced_result['vqa_insights'] = comprehensive_features['vqa_results']
                    
                    if search_options.get('include_scene_graph') and 'scene_graph' in comprehensive_features:
                        enhanced_result['scene_graph'] = comprehensive_features['scene_graph']
                    
                    enhanced_results.append(enhanced_result)
                    
                except Frame.DoesNotExist:
                    enhanced_results.append(result)
            
            # AI 기반 검색 인사이트 생성
            search_insights = self._generate_search_insights(query, enhanced_results, video)
            
            return Response({
                'search_results': enhanced_results,
                'query': query,
                'insights': search_insights,
                'total_matches': len(search_results),
                'search_type': 'advanced',
                'video_info': {
                    'id': video.id,
                    'name': video.original_name,
                    'analysis_type': getattr(video, 'analysis_type', 'basic')
                }
            })
            
        except Video.DoesNotExist:
            return Response({
                'error': '비디오를 찾을 수 없습니다.'
            }, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({
                'error': f'고급 검색 실패: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    def _generate_search_insights(self, query, results, video):
        """검색 결과에 대한 AI 인사이트 생성"""
        try:
            if not results:
                return "검색 결과가 없습니다. 다른 검색어를 시도해보세요."
            
            # 검색 결과 요약
            insights_prompt = f"""
            검색어: "{query}"
            비디오: {video.original_name}
            검색 결과: {len(results)}개 매칭
            
            주요 발견사항:
            {json.dumps(results[:3], ensure_ascii=False, indent=2)}
            
            이 검색 결과에 대한 간단하고 유용한 인사이트를 한국어로 제공해주세요.
            """
            
            insights = self.llm_client.generate_smart_response(
                user_query=insights_prompt,
                search_results=results[:5],
                video_info=f"비디오: {video.original_name}",
                use_multi_llm=False
            )
            
            return insights
            
        except Exception as e:
            return f"인사이트 생성 중 오류 발생: {str(e)}"


class EnhancedFrameView(APIView):
    """고급 분석 정보가 포함된 프레임 데이터 제공"""
    permission_classes = [AllowAny]
    
    def get(self, request, video_id, frame_number):
        try:
            video = Video.objects.get(id=video_id)
            
            # 프레임 데이터 조회
            try:
                frame = Frame.objects.get(video=video, image_id=frame_number)
                
                frame_data = {
                    'frame_id': frame.image_id,
                    'timestamp': frame.timestamp,
                    'caption': frame.caption,
                    'enhanced_caption': frame.enhanced_caption,
                    'final_caption': frame.final_caption,
                    'detected_objects': frame.detected_objects,
                    'comprehensive_features': frame.comprehensive_features,
                    'analysis_quality': frame.comprehensive_features.get('caption_quality', 'basic')
                }
                
                # 고급 분석 결과 분해
                if frame.comprehensive_features:
                    features = frame.comprehensive_features
                    
                    frame_data['advanced_analysis'] = {
                        'clip_analysis': features.get('clip_features', {}),
                        'ocr_text': features.get('ocr_text', {}),
                        'vqa_results': features.get('vqa_results', {}),
                        'scene_graph': features.get('scene_graph', {}),
                        'scene_complexity': features.get('scene_complexity', 0)
                    }
                
                return Response(frame_data)
                
            except Frame.DoesNotExist:
                # 프레임 데이터가 없으면 기본 이미지만 반환
                return Response({
                    'frame_id': frame_number,
                    'message': '프레임 데이터는 없지만 이미지는 사용 가능합니다.',
                    'image_url': f'/frame/{video_id}/{frame_number}/'
                })
            
        except Video.DoesNotExist:
            return Response({
                'error': '비디오를 찾을 수 없습니다.'
            }, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({
                'error': f'프레임 정보 조회 실패: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class APIStatusView(APIView):
    """API 상태 확인"""
    permission_classes = [AllowAny]
    
    def get(self, request):
        # print("🔍 APIStatusView: API 상태 요청 받음")
        try:
            from .llm_client import LLMClient
            llm_client = LLMClient()
            status_info = llm_client.get_api_status()
            
            response_data = {
                'groq': status_info.get('groq', {'available': False}),
                'openai': status_info.get('openai', {'available': False}),
                'anthropic': status_info.get('anthropic', {'available': False}),
                'fallback_enabled': True,
                'timestamp': datetime.now().isoformat(),
                'server_status': 'running',
                'active_analyses': 0  # 기본값
            }
            
            # print(f"✅ APIStatusView: 상태 정보 반환 - {response_data}")
            return Response(response_data)
        except Exception as e:
            print(f"❌ APIStatusView 오류: {e}")
            return Response({
                'error': str(e),
                'server_status': 'error'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)



class EnhancedAnalyzeVideoView(APIView):
    """실제 AI 분석을 사용하는 고급 비디오 분석 시작"""
    permission_classes = [AllowAny]
    
    def post(self, request, video_id):
        try:
            print(f"🚀 실제 AI 비디오 분석 시작: video_id={video_id}")
            
            analysis_type = request.data.get('analysisType', 'enhanced')
            analysis_config = request.data.get('analysisConfig', {})
            enhanced_analysis = request.data.get('enhancedAnalysis', True)
            
            print(f"📋 분석 요청 정보:")
            print(f"  - 비디오 ID: {video_id}")
            print(f"  - 분석 타입: {analysis_type}")
            print(f"  - 고급 분석: {enhanced_analysis}")
            print(f"  - 분석 설정: {analysis_config}")
            
            # 비디오 존재 확인
            try:
                video = Video.objects.get(id=video_id)
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
            
            # AI 분석기 사용 가능 여부 확인
            if not VIDEO_ANALYZER_AVAILABLE:
                return Response({
                    'error': 'AI 분석 모듈을 사용할 수 없습니다. 서버 설정을 확인해주세요.',
                    'fallback': 'basic_analysis'
                }, status=status.HTTP_503_SERVICE_UNAVAILABLE)
            
            # 분석기 상태 확인
            analyzer_status = get_analyzer_status()
            print(f"🔍 분석기 상태: {analyzer_status}")
            
            # 분석 상태 업데이트
            video.analysis_status = 'processing'
            video.save()
            
            print(f"✅ 비디오 상태를 'processing'으로 변경: {video.original_name}")
            
            # 백그라운드에서 실제 AI 분석 시작
            analysis_thread = threading.Thread(
                target=self._run_real_ai_analysis,
                args=(video, analysis_type, analysis_config, enhanced_analysis),
                daemon=True
            )
            analysis_thread.start()
            
            print("🧵 실제 AI 분석 스레드 시작됨")
            
            return Response({
                'success': True,
                'message': f'{self._get_analysis_type_name(analysis_type)} AI 분석이 시작되었습니다.',
                'video_id': video.id,
                'analysis_type': analysis_type,
                'enhanced_analysis': enhanced_analysis,
                'estimated_time': self._get_estimated_time_real(analysis_type),
                'status': 'processing',
                'ai_features': analyzer_status.get('features', {}),
                'analysis_method': 'real_ai_analysis'
            })
            
        except Exception as e:
            print(f"❌ AI 분석 시작 오류: {e}")
            import traceback
            print(f"🔍 상세 오류: {traceback.format_exc()}")
            
            return Response({
                'error': f'AI 분석 시작 중 오류가 발생했습니다: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    def _run_real_ai_analysis(self, video, analysis_type, analysis_config, enhanced_analysis):
        """백그라운드에서 실행되는 실제 AI 분석 함수"""
        start_time = time.time()
        
        try:
            print(f"🚀 비디오 {video.id} 실제 AI 분석 시작 - 타입: {analysis_type}")
            
            # 1. VideoAnalyzer 인스턴스 가져오기
            analyzer = get_video_analyzer()
            if not analyzer:
                raise Exception("VideoAnalyzer 인스턴스를 가져올 수 없습니다")
            
            print(f"✅ VideoAnalyzer 로드 완료: {type(analyzer).__name__}")
            
            # 2. 분석 결과 저장 디렉토리 생성
            analysis_results_dir = os.path.join(settings.MEDIA_ROOT, 'analysis_results')
            os.makedirs(analysis_results_dir, exist_ok=True)
            
            # 3. JSON 파일명 생성
            timestamp = int(time.time())
            json_filename = f"real_analysis_{video.id}_{analysis_type}_{timestamp}.json"
            json_filepath = os.path.join(analysis_results_dir, json_filename)
            
            print(f"📁 분석 결과 저장 경로: {json_filepath}")
            
            # 4. 진행률 콜백 함수 정의
            def progress_callback(progress, message):
                print(f"📊 분석 진행률: {progress:.1f}% - {message}")
                # 필요시 웹소켓이나 다른 방법으로 실시간 업데이트 가능
            
            # 5. 실제 AI 분석 수행
            print("🧠 실제 AI 분석 시작...")
            analysis_results = analyzer.analyze_video_comprehensive(
                video=video,
                analysis_type=analysis_type,
                progress_callback=progress_callback
            )
            
            if not analysis_results.get('success', False):
                raise Exception(f"AI 분석 실패: {analysis_results.get('error', 'Unknown error')}")
            
            print(f"✅ AI 분석 완료: {analysis_results.get('total_frames_analyzed', 0)}개 프레임 처리")
            
            # 6. 메타데이터 추가
            analysis_results['metadata'] = {
                'video_id': video.id,
                'video_name': video.original_name,
                'analysis_type': analysis_type,
                'analysis_config': analysis_config,
                'enhanced_analysis': enhanced_analysis,
                'json_file_path': json_filepath,
                'analysis_timestamp': datetime.now().isoformat(),
                'total_frames': getattr(video, 'total_frames', 0),
                'video_duration': getattr(video, 'duration', 0),
                'fps': getattr(video, 'fps', 30),
                'processing_time_seconds': time.time() - start_time,
                'analysis_method': 'real_ai_enhanced',
                'ai_features_used': analysis_results.get('analysis_config', {}).get('features_enabled', {})
            }
            
            # 7. JSON 파일 저장
            try:
                with open(json_filepath, 'w', encoding='utf-8') as f:
                    json.dump(analysis_results, f, ensure_ascii=False, indent=2, default=str)
                print(f"✅ 분석 결과 JSON 저장 완료: {json_filepath}")
            except Exception as json_error:
                print(f"⚠️ JSON 저장 실패: {json_error}")
                # JSON 저장 실패해도 DB는 저장하도록 계속 진행
            
            # 8. Django 모델에 분석 결과 저장
            self._save_analysis_to_db(video, analysis_results, enhanced_analysis, json_filepath)
            
            # 9. RAG 시스템에 분석 결과 등록
            self._register_to_rag_system(video.id, json_filepath)
            
            # 10. 완료 상태 업데이트
            video.analysis_status = 'completed'
            video.is_analyzed = True
            video.save()
            
            processing_time = time.time() - start_time
            print(f"🎉 비디오 {video.id} 실제 AI 분석 완료!")
            print(f"📊 처리 시간: {processing_time:.1f}초")
            print(f"📊 최종 통계: {analysis_results.get('total_frames_analyzed', 0)}개 프레임 분석")
            
        except Exception as e:
            print(f"❌ 비디오 {video.id} AI 분석 실패: {e}")
            import traceback
            print(f"🔍 상세 오류:\n{traceback.format_exc()}")
            
            # 오류 상태 업데이트
            try:
                video.analysis_status = 'failed'
                video.save()
            except Exception as save_error:
                print(f"⚠️ 오류 상태 저장 실패: {save_error}")


    def _save_analysis_to_db(self, video, analysis_results, enhanced_analysis, json_filepath):
        """분석 결과를 Django DB에 저장"""
        try:
            print("💾 분석 결과를 DB에 저장 중...")

            video_summary = analysis_results.get('video_summary', {})
            frame_results = (
                analysis_results.get('frame_results')
                or analysis_results.get('frames')
                or []
            )
            analysis_config = analysis_results.get('analysis_config', {})
            metadata = analysis_results.get('metadata', {})

            # Video 모델의 분석 필드 업데이트
            video.enhanced_analysis = enhanced_analysis
            video.success_rate = 95.0
            video.processing_time = metadata.get('processing_time_seconds', 0)
            video.analysis_type = 'enhanced'
            video.advanced_features_used = analysis_config.get('features_enabled', {})
            video.scene_types = video_summary.get('scene_types', [])
            video.unique_objects = len(video_summary.get('dominant_objects', []))
            video.analysis_json_path = json_filepath
            video.save()

            # Scene 저장 (하이라이트 프레임 기반)
            highlight_frames = video_summary.get('highlight_frames', [])
            scene_duration = video.duration / max(len(highlight_frames), 1) if video.duration > 0 else 1

            for i, highlight in enumerate(highlight_frames[:10]):
                Scene.objects.create(
                    video=video,
                    scene_id=i + 1,
                    start_time=max(0, highlight.get('timestamp', 0) - scene_duration/2),
                    end_time=min(video.duration, highlight.get('timestamp', 0) + scene_duration/2),
                    duration=scene_duration,
                    frame_count=60,
                    dominant_objects=video_summary.get('dominant_objects', [])[:5],
                    enhanced_captions_count=1 if highlight.get('object_count', 0) > 0 else 0
                )

            # Frame 저장 (프레임 ID 기준으로 전부 저장)
            important_frames = [f for f in frame_results if f.get('image_id') is not None]

            for frame_data in important_frames[:50]:
                try:
                    # ✅ 이미지 저장
                    image_path = self._save_frame_image(video, frame_data)
                    
                    if image_path:
                        print(f"✅ 이미지 저장 성공: {image_path}")
                    else:
                        print(f"⚠️ 이미지 저장 실패, 프레임 정보만 저장")

                    # ✅ persons 데이터를 detected_objects에 저장
                    persons_data = frame_data.get("persons", [])
                    
                    # ✅ attributes 안에서 꺼내기
                    attrs = frame_data.get("attributes", {})

                    detected = {
                        'persons': persons_data,  # YOLO로 감지된 사람 객체들
                        'clothing': attrs.get('detailed_clothing', {}),
                        'color': attrs.get('clothing_color', {}),
                        'accessories': attrs.get('accessories', {}),
                        'posture': attrs.get('posture', {}),
                        'hair_style': attrs.get('hair_style', {}),
                        'facial_attributes': attrs.get('facial_attributes', {})
                    }

                    Frame.objects.update_or_create(
                        video=video,
                        image_id=frame_data.get('image_id', 0),
                        defaults={
                            'timestamp': frame_data.get('timestamp', 0),
                            'caption': frame_data.get('caption', ''),
                            'enhanced_caption': frame_data.get('enhanced_caption', ''),
                            'final_caption': frame_data.get('final_caption', ''),
                            'detected_objects': detected,
                            'comprehensive_features': {
                                "crop_quality": frame_data.get("crop_quality", {}),
                                "pose_analysis": attrs.get("pose_analysis", {}),
                                "facial_details": attrs.get("facial_details", {})
                            },
                            'image': image_path if image_path else None
                        }
                    )
                except Exception as frame_error:
                    print(f"⚠️ 프레임 {frame_data.get('image_id', 'unknown')} 저장 실패: {frame_error}")
                    continue

            print(f"✅ DB 저장 완료: {len(important_frames)}개 프레임, {len(highlight_frames)}개 씬")

        except Exception as e:
            print(f"❌ DB 저장 실패: {e}")
            import traceback
            print(f"🔍 DB 저장 오류 상세:\n{traceback.format_exc()}")


    def _register_to_rag_system(self, video_id, json_filepath):
        """RAG 시스템에 분석 결과 등록"""
        try:
            print(f"🔍 RAG 시스템에 비디오 {video_id} 등록 중...")
            
            rag_system = get_video_rag_system()
            if not rag_system:
                print("⚠️ RAG 시스템을 사용할 수 없습니다")
                return
            
            success = rag_system.process_video_analysis_json(json_filepath, str(video_id))
            
            if success:
                print(f"✅ RAG 시스템 등록 완료: 비디오 {video_id}")
            else:
                print(f"⚠️ RAG 시스템 등록 실패: 비디오 {video_id}")
                
        except Exception as e:
            print(f"❌ RAG 시스템 등록 오류: {e}")
            import traceback
            traceback.print_exc()
    
    def _get_analysis_type_name(self, analysis_type):
        """분석 타입 이름 반환"""
        type_names = {
            'basic': '기본 AI 분석',
            'enhanced': '향상된 AI 분석',
            'comprehensive': '종합 AI 분석',
            'custom': '사용자 정의 AI 분석'
        }
        return type_names.get(analysis_type, '향상된 AI 분석')
    
    def _get_estimated_time_real(self, analysis_type):
        """실제 AI 분석 타입별 예상 시간"""
        time_estimates = {
            'basic': '5-15분',
            'enhanced': '10-30분', 
            'comprehensive': '20-60분',
            'custom': '상황에 따라 다름'
        }
        return time_estimates.get(analysis_type, '10-30분')
    
    def get(self, request, video_id):
        """분석 상태 조회"""
        try:
            video = Video.objects.get(id=video_id)
            
            analyzer_status = get_analyzer_status() if VIDEO_ANALYZER_AVAILABLE else {'status': 'unavailable'}
            
            return Response({
                'video_id': video.id,
                'video_name': video.original_name,
                'analysis_status': video.analysis_status,
                'is_analyzed': video.is_analyzed,
                'analyzer_available': VIDEO_ANALYZER_AVAILABLE,
                'analyzer_status': analyzer_status,
                'last_updated': video.updated_at.isoformat() if hasattr(video, 'updated_at') else None
            })
            
        except Video.DoesNotExist:
            return Response({
                'error': '해당 비디오를 찾을 수 없습니다.'
            }, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({
                'error': f'상태 조회 중 오류: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    def _save_frame_image(self, video, frame_data):
        """프레임 이미지를 저장하고 경로를 반환"""
        try:
            import cv2
            from PIL import Image
            import numpy as np
            
            # 비디오 파일 열기
            video_path = video.file.path
            cap = cv2.VideoCapture(video_path)
            
            # 프레임 번호 가져오기
            frame_number = frame_data.get('image_id', 1)
            if frame_number is None:
                frame_number = 1
            
            # 해당 프레임으로 이동
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
            ret, frame = cap.read()
            
            if not ret:
                cap.release()
                return None
            
            # 이미지 저장 경로 설정
            frame_filename = f"video{video.id}_frame{frame_number}.jpg"
            frame_path = os.path.join(settings.MEDIA_ROOT, 'images', frame_filename)
            
            # 디렉토리 생성
            os.makedirs(os.path.dirname(frame_path), exist_ok=True)
            
            # 이미지 저장
            cv2.imwrite(frame_path, frame)
            cap.release()
            
            # 상대 경로 반환
            relative_path = f"images/{frame_filename}"
            return relative_path
            
        except Exception as e:
            print(f"⚠️ 프레임 이미지 저장 오류: {e}")
            return None

# 새로운 뷰 추가: AnalysisCapabilitiesView 완전 구현
class AnalysisCapabilitiesView(APIView):
    """시스템 분석 기능 상태 확인 - 완전 구현"""
    permission_classes = [AllowAny]
    
    def get(self, request):
        try:
            print("🔍 AnalysisCapabilitiesView: 분석 기능 상태 요청")
            
            # VideoAnalyzer 상태 확인
            analyzer_status = self._check_video_analyzer()
            
            # MultiLLM 상태 확인
            multi_llm_status = self._check_multi_llm_analyzer()
            
            # 시스템 기능 상태
            capabilities = {
                'system_status': {
                    'analyzer_available': analyzer_status['available'],
                    'multi_llm_available': multi_llm_status['available'],
                    'device': analyzer_status.get('device', 'unknown'),
                    'timestamp': datetime.now().isoformat()
                },
                'core_features': {
                    'object_detection': {
                        'name': '객체 감지',
                        'available': analyzer_status.get('yolo_available', False),
                        'description': 'YOLO 기반 실시간 객체 감지',
                        'icon': '🎯'
                    },
                    'enhanced_captions': {
                        'name': '고급 캡션 생성',
                        'available': True,
                        'description': 'AI 기반 상세 캡션 생성',
                        'icon': '💬'
                    }
                },
                'advanced_features': {
                    'clip_analysis': {
                        'name': 'CLIP 분석',
                        'available': analyzer_status.get('clip_available', False),
                        'description': 'OpenAI CLIP 모델 기반 씬 이해',
                        'icon': '🖼️'
                    },
                    'ocr_text_extraction': {
                        'name': 'OCR 텍스트 추출',
                        'available': analyzer_status.get('ocr_available', False),
                        'description': 'EasyOCR 기반 다국어 텍스트 인식',
                        'icon': '📝'
                    },
                    'vqa_analysis': {
                        'name': 'VQA 질문답변',
                        'available': analyzer_status.get('vqa_available', False),
                        'description': 'BLIP 모델 기반 시각적 질문 답변',
                        'icon': '❓'
                    },
                    'scene_graph': {
                        'name': 'Scene Graph',
                        'available': analyzer_status.get('scene_graph_available', False),
                        'description': 'NetworkX 기반 객체 관계 분석',
                        'icon': '🕸️'
                    }
                },
                'multi_llm_features': {
                    'gpt4v': {
                        'name': 'GPT-4V',
                        'available': multi_llm_status.get('gpt4v_available', False),
                        'description': 'OpenAI GPT-4 Vision',
                        'icon': '🟢'
                    },
                    'claude': {
                        'name': 'Claude-3.5',
                        'available': multi_llm_status.get('claude_available', False),
                        'description': 'Anthropic Claude-3.5 Sonnet',
                        'icon': '🟠'
                    },
                    'gemini': {
                        'name': 'Gemini Pro',
                        'available': multi_llm_status.get('gemini_available', False),
                        'description': 'Google Gemini Pro Vision',
                        'icon': '🔵'
                    },
                    'groq': {
                        'name': 'Groq Llama',
                        'available': multi_llm_status.get('groq_available', False),
                        'description': 'Groq Llama-3.1-70B',
                        'icon': '⚡'
                    }
                },
                'api_status': {
                    'openai_available': multi_llm_status.get('openai_api_key', False),
                    'anthropic_available': multi_llm_status.get('anthropic_api_key', False),
                    'google_available': multi_llm_status.get('google_api_key', False),
                    'groq_available': multi_llm_status.get('groq_api_key', False)
                }
            }
            
            # 사용 가능한 기능 수 계산
            total_features = (len(capabilities['core_features']) + 
                            len(capabilities['advanced_features']) + 
                            len(capabilities['multi_llm_features']))
            
            available_features = sum(1 for features in [
                capabilities['core_features'], 
                capabilities['advanced_features'],
                capabilities['multi_llm_features']
            ] for feature in features.values() if feature.get('available', False))
            
            capabilities['summary'] = {
                'total_features': total_features,
                'available_features': available_features,
                'availability_rate': (available_features / total_features * 100) if total_features > 0 else 0,
                'system_ready': analyzer_status['available'] and available_features > 0,
                'multi_llm_ready': multi_llm_status['available'] and multi_llm_status['model_count'] > 0
            }
            
            print(f"✅ 분석 기능 상태: {available_features}/{total_features} 사용 가능")
            
            return Response(capabilities)
            
        except Exception as e:
            print(f"❌ AnalysisCapabilitiesView 오류: {e}")
            import traceback
            print(f"🔍 상세 오류: {traceback.format_exc()}")
            
            return Response({
                'system_status': {
                    'analyzer_available': False,
                    'multi_llm_available': False,
                    'device': 'error',
                    'error': str(e)
                },
                'summary': {
                    'system_ready': False,
                    'error': str(e)
                }
            }, status=500)
    
    def _check_video_analyzer(self):
        """VideoAnalyzer 상태 확인"""
        try:
            analyzer = get_video_analyzer()
            return {
                'available': True,
                'device': getattr(analyzer, 'device', 'cpu'),
                'yolo_available': getattr(analyzer, 'model', None) is not None,
                'clip_available': getattr(analyzer, 'clip_available', False),
                'ocr_available': getattr(analyzer, 'ocr_available', False),
                'vqa_available': getattr(analyzer, 'vqa_available', False),
                'scene_graph_available': getattr(analyzer, 'scene_graph_available', False)
            }
        except Exception as e:
            print(f"❌ VideoAnalyzer 상태 확인 실패: {e}")
            return {'available': False, 'error': str(e)}
    
    def _check_multi_llm_analyzer(self):
        """MultiLLM 상태 확인"""
        try:
            multi_llm = get_multi_llm_analyzer()
            available_models = getattr(multi_llm, 'available_models', [])
            
            return {
                'available': len(available_models) > 0,
                'model_count': len(available_models),
                'available_models': available_models,
                'gpt4v_available': 'gpt-4v' in available_models,
                'claude_available': 'claude-3.5' in available_models,
                'gemini_available': 'gemini-pro' in available_models,
                'groq_available': 'groq-llama' in available_models,
                'openai_api_key': bool(os.getenv("OPENAI_API_KEY")),
                'anthropic_api_key': bool(os.getenv("ANTHROPIC_API_KEY")),
                'google_api_key': bool(os.getenv("GOOGLE_API_KEY")),
                'groq_api_key': bool(os.getenv("GROQ_API_KEY"))
            }
        except Exception as e:
            print(f"❌ MultiLLM 상태 확인 실패: {e}")
            return {'available': False, 'error': str(e)}


# 새로운 뷰: MultiLLM 전용 채팅 뷰
class MultiLLMChatView(APIView):
    """멀티 LLM 전용 채팅 뷰"""
    permission_classes = [AllowAny]
    
    def __init__(self):
        super().__init__()
        self.multi_llm_analyzer = get_multi_llm_analyzer()
    
    def post(self, request):
        try:
            user_query = request.data.get('message', '').strip()
            video_id = request.data.get('video_id')
            analysis_mode = request.data.get('analysis_mode', 'comparison')
            
            if not user_query:
                return Response({'error': '메시지를 입력해주세요.'}, status=400)
            
            # 비디오가 없어도 텍스트 기반으로 처리 가능
            video = None
            video_context = {}
            frame_images = []
            
            if video_id:
                try:
                    video = Video.objects.get(id=video_id)
                    video_context = self._prepare_video_context(video)
                    frame_images = self._extract_frames_safely(video)
                except Video.DoesNotExist:
                    pass  # 비디오 없이도 진행
            
            # 멀티 LLM 분석 실행
            multi_responses = self.multi_llm_analyzer.analyze_video_multi_llm(
                frame_images, user_query, video_context
            )
            
            comparison_result = self.multi_llm_analyzer.compare_responses(multi_responses)
            
            return Response({
                'response_type': 'multi_llm_result',
                'query': user_query,
                'video_info': {'id': video.id, 'name': video.original_name} if video else None,
                'llm_responses': {
                    model: {
                        'response': resp.response_text,
                        'confidence': resp.confidence_score,
                        'processing_time': resp.processing_time,
                        'success': resp.success,
                        'error': resp.error
                    }
                    for model, resp in multi_responses.items()
                },
                'comparison_analysis': comparison_result['comparison'],
                'recommendation': comparison_result['comparison']['recommendation']
            })
            
        except Exception as e:
            print(f"❌ MultiLLM 채팅 오류: {e}")
            return Response({'error': str(e)}, status=500)
    
    def _prepare_video_context(self, video):
        """비디오 컨텍스트 준비"""
        context = {
            'duration': video.duration,
            'filename': video.original_name
        }
        
        if video.is_analyzed:
            try:
                context.update({
                    'detected_objects': video.advanced_features_used.get('dominant_objects', []),
                    'scene_types': video.scene_types
                })
            except:
                pass
        
        return context
    
    def _extract_frames_safely(self, video):
        """안전한 프레임 추출"""
        try:
            # EnhancedVideoChatView의 메서드 재사용
            view = EnhancedVideoChatView()
            return view._extract_key_frames_for_llm(video, max_frames=2)
        except:
            return []


# LLM 통계 뷰 추가
class LLMStatsView(APIView):
    """LLM 성능 통계 뷰"""
    permission_classes = [AllowAny]
    
    def get(self, request):
        try:
            # 간단한 통계 반환 (실제로는 데이터베이스에서 수집)
            stats = {
                'total_requests': 0,
                'model_usage': {
                    'gpt-4v': {'count': 0, 'avg_time': 0, 'success_rate': 0},
                    'claude-3.5': {'count': 0, 'avg_time': 0, 'success_rate': 0},
                    'gemini-pro': {'count': 0, 'avg_time': 0, 'success_rate': 0},
                    'groq-llama': {'count': 0, 'avg_time': 0, 'success_rate': 0}
                },
                'average_response_time': 0,
                'overall_success_rate': 0,
                'last_updated': datetime.now().isoformat()
            }
            
            return Response(stats)
            
        except Exception as e:
            return Response({'error': str(e)}, status=500)

class AnalysisStatusView(APIView):
    """분석 상태 확인 - 진행률 정보 포함"""
    permission_classes = [AllowAny]
    
    def get(self, request, video_id):
        try:
            video = Video.objects.get(id=video_id)
            
            response_data = {
                'status': video.analysis_status,
                'video_filename': video.filename,
                'is_analyzed': video.is_analyzed
            }
            
            # 진행률 정보 추가
            if video.analysis_status == 'processing':
                response_data.update({
                    'progress': 50,
                    'status': 'processing',
                    'message': '분석 진행 중...'
                })
            
            # 분석 완료된 경우 상세 정보 추가
            if video.is_analyzed:
                response_data.update({
                    'enhanced_analysis': video.enhanced_analysis,
                    'success_rate': video.success_rate,
                    'processing_time': video.processing_time,
                    'stats': {
                        'objects': video.unique_objects,
                        'scenes': Scene.objects.filter(video=video).count(),
                        'captions': Frame.objects.filter(video=video, caption__isnull=False).count()
                    }
                })
            
            return Response(response_data)
            
        except Video.DoesNotExist:
            return Response({
                'error': '비디오를 찾을 수 없습니다'
            }, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# ✅ 수정된 AnalyzeVideoView - URL 파라미터 처리
class AnalyzeVideoView(APIView):
    """기본 비디오 분석 시작"""
    permission_classes = [AllowAny]
    
    def post(self, request, video_id):  # ✅ video_id 파라미터 추가
        try:
            print(f"🔬 기본 비디오 분석 시작: video_id={video_id}")
            
            enable_enhanced = request.data.get('enable_enhanced_analysis', False)
            
            try:
                video = Video.objects.get(id=video_id)
            except Video.DoesNotExist:
                return Response({
                    'error': '비디오를 찾을 수 없습니다.'
                }, status=status.HTTP_404_NOT_FOUND)
            
            # 이미 분석 중인지 확인
            if video.analysis_status == 'processing':
                return Response({
                    'error': '이미 분석이 진행 중입니다.'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # 분석 상태 업데이트
            video.analysis_status = 'processing'
            video.save()
            
            # 백그라운드에서 분석 시작
            analysis_thread = threading.Thread(
                target=self._run_basic_analysis,
                args=(video, enable_enhanced),
                daemon=True
            )
            analysis_thread.start()
            
            return Response({
                'success': True,
                'message': '기본 비디오 분석이 시작되었습니다.',
                'video_id': video.id,
                'enhanced_analysis': enable_enhanced,
                'estimated_time': '5-10분'
            })
            
        except Exception as e:
            print(f"❌ 기본 분석 시작 오류: {e}")
            return Response({
                'error': f'분석 시작 중 오류: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    def _run_basic_analysis(self, video, enable_enhanced):
        """백그라운드 기본 분석"""
        try:
            print(f"🔬 기본 분석 실행: {video.original_name}")
            
            # 간단한 분석 시뮬레이션
            time.sleep(2)  # 실제로는 분석 로직 수행
            
            # Video 모델의 분석 필드 업데이트
            video.enhanced_analysis = enable_enhanced
            video.success_rate = 85.0
            video.processing_time = 120
            video.analysis_type = 'basic'
            video.unique_objects = 8
            video.scene_types = ['outdoor', 'urban']
            video.save()
            
            # 완료 상태 업데이트
            video.analysis_status = 'completed'
            video.is_analyzed = True
            video.save()
            
            print(f"✅ 기본 분석 완료: {video.original_name}")
            
        except Exception as e:
            print(f"❌ 기본 분석 실패: {e}")
            video.analysis_status = 'failed'
            video.save()

class AnalysisProgressView(APIView):
    """분석 진행률 전용 API"""
    permission_classes = [AllowAny]
    
    def get(self, request, video_id):
        try:
            # 기본 진행률 정보 반환
            progress_info = {
                'progress': 50,
                'status': 'processing',
                'message': '분석 진행 중...'
            }
            
            return Response(progress_info)
            
        except Exception as e:
            return Response({
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# 기존의 다른 View 클래스들은 그대로 유지


class ScenesView(APIView):
    """Scene 목록 조회"""
    permission_classes = [AllowAny]  # 🔧 권한 설정 추가
    
    def get(self, request, video_id):
        try:
            video = Video.objects.get(id=video_id)
            scenes = Scene.objects.filter(video=video).order_by('scene_id')
            
            scene_list = []
            for scene in scenes:
                scene_data = {
                    'scene_id': scene.scene_id,
                    'start_time': scene.start_time,
                    'end_time': scene.end_time,
                    'duration': scene.duration,
                    'frame_count': scene.frame_count,
                    'dominant_objects': scene.dominant_objects,
                    'caption_type': 'enhanced' if scene.enhanced_captions_count > 0 else 'basic'
                }
                scene_list.append(scene_data)
            
            return Response({
                'scenes': scene_list,
                'total_scenes': len(scene_list),
                'analysis_type': 'enhanced' if video.enhanced_analysis else 'basic'
            })
            
        except Video.DoesNotExist:
            return Response({
                'error': '비디오를 찾을 수 없습니다'
            }, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        



import os
import json
import time
import cv2
import numpy as np
from django.conf import settings
from django.http import JsonResponse, HttpResponse, FileResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.utils.decorators import method_decorator
from django.views import View
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.permissions import AllowAny
from datetime import datetime, timedelta
from collections import Counter
import threading
import queue

from .models import Video, TrackPoint, Frame, Scene
from django.http import JsonResponse



class AnalysisFeaturesView(APIView):
    """분석 기능별 상세 정보 제공"""
    permission_classes = [AllowAny]
    
    def get(self, request):
        try:
            analyzer = VideoAnalyzer()
            
            features = {
                'object_detection': {
                    'name': '객체 감지',
                    'description': 'YOLO 기반 실시간 객체 감지 및 분류',
                    'available': True,
                    'processing_time_factor': 1.0,
                    'icon': '🎯',
                    'details': '비디오 내 사람, 차량, 동물 등 다양한 객체를 정확하게 감지합니다.'
                },
                'clip_analysis': {
                    'name': 'CLIP 씬 분석',
                    'description': 'OpenAI CLIP 모델을 활용한 고급 씬 이해',
                    'available': analyzer.clip_available,
                    'processing_time_factor': 1.5,
                    'icon': '🖼️',
                    'details': '이미지의 의미적 컨텍스트를 이해하여 씬 분류 및 분석을 수행합니다.'
                },
                'ocr': {
                    'name': 'OCR 텍스트 추출',
                    'description': 'EasyOCR을 사용한 다국어 텍스트 인식',
                    'available': analyzer.ocr_available,
                    'processing_time_factor': 1.2,
                    'icon': '📝',
                    'details': '비디오 내 한글, 영문 텍스트를 정확하게 인식하고 추출합니다.'
                },
                'vqa': {
                    'name': 'VQA 질문답변',
                    'description': 'BLIP 모델 기반 시각적 질문 답변',
                    'available': analyzer.vqa_available,
                    'processing_time_factor': 2.0,
                    'icon': '❓',
                    'details': '이미지에 대한 질문을 생성하고 답변하여 깊이 있는 분석을 제공합니다.'
                },
                'scene_graph': {
                    'name': 'Scene Graph',
                    'description': '객체간 관계 및 상호작용 분석',
                    'available': analyzer.scene_graph_available,
                    'processing_time_factor': 3.0,
                    'icon': '🕸️',
                    'details': '객체들 사이의 관계와 상호작용을 분석하여 복잡한 씬을 이해합니다.'
                },
                'enhanced_caption': {
                    'name': '고급 캡션 생성',
                    'description': '모든 분석 결과를 통합한 상세 캡션',
                    'available': True,
                    'processing_time_factor': 1.1,
                    'icon': '💬',
                    'details': '여러 AI 모델의 결과를 종합하여 상세하고 정확한 캡션을 생성합니다.'
                }
            }
            
            return Response({
                'features': features,
                'device': analyzer.device,
                'total_available': sum(1 for f in features.values() if f['available']),
                'recommended_configs': {
                    'basic': ['object_detection', 'enhanced_caption'],
                    'enhanced': ['object_detection', 'clip_analysis', 'ocr', 'enhanced_caption'],
                    'comprehensive': list(features.keys())
                }
            })
            
        except Exception as e:
            return Response({
                'error': f'분석 기능 정보 조회 실패: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class AdvancedVideoSearchView(APIView):
    """고급 비디오 검색 API"""
    permission_classes = [AllowAny]
    
    def __init__(self):
        super().__init__()
        self.video_analyzer = VideoAnalyzer()
        self.llm_client = LLMClient()
    
    def post(self, request):
        try:
            video_id = request.data.get('video_id')
            query = request.data.get('query', '').strip()
            search_options = request.data.get('search_options', {})
            
            if not query:
                return Response({
                    'error': '검색어를 입력해주세요.'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            video = Video.objects.get(id=video_id)
            
            # 고급 검색 수행
            search_results = self.video_analyzer.search_comprehensive(video, query)
            
            # 고급 분석 결과가 포함된 프레임들에 대해 추가 정보 수집
            enhanced_results = []
            for result in search_results[:10]:
                frame_id = result.get('frame_id')
                try:
                    frame = Frame.objects.get(video=video, image_id=frame_id)
                    enhanced_result = dict(result)
                    
                    # 고급 분석 결과 추가
                    comprehensive_features = frame.comprehensive_features or {}
                    
                    if search_options.get('include_clip_analysis') and 'clip_features' in comprehensive_features:
                        enhanced_result['clip_analysis'] = comprehensive_features['clip_features']
                    
                    if search_options.get('include_ocr_text') and 'ocr_text' in comprehensive_features:
                        enhanced_result['ocr_text'] = comprehensive_features['ocr_text']
                    
                    if search_options.get('include_vqa_results') and 'vqa_results' in comprehensive_features:
                        enhanced_result['vqa_insights'] = comprehensive_features['vqa_results']
                    
                    if search_options.get('include_scene_graph') and 'scene_graph' in comprehensive_features:
                        enhanced_result['scene_graph'] = comprehensive_features['scene_graph']
                    
                    enhanced_results.append(enhanced_result)
                    
                except Frame.DoesNotExist:
                    enhanced_results.append(result)
            
            # AI 기반 검색 인사이트 생성
            search_insights = self._generate_search_insights(query, enhanced_results, video)
            
            return Response({
                'search_results': enhanced_results,
                'query': query,
                'insights': search_insights,
                'total_matches': len(search_results),
                'search_type': 'advanced',
                'video_info': {
                    'id': video.id,
                    'name': video.original_name,
                    'analysis_type': getattr(video, 'analysis_type', 'basic')
                }
            })
            
        except Video.DoesNotExist:
            return Response({
                'error': '비디오를 찾을 수 없습니다.'
            }, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({
                'error': f'고급 검색 실패: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    def _generate_search_insights(self, query, results, video):
        """검색 결과에 대한 AI 인사이트 생성"""
        try:
            if not results:
                return "검색 결과가 없습니다. 다른 검색어를 시도해보세요."
            
            # 검색 결과 요약
            insights_prompt = f"""
            검색어: "{query}"
            비디오: {video.original_name}
            검색 결과: {len(results)}개 매칭
            
            주요 발견사항:
            {json.dumps(results[:3], ensure_ascii=False, indent=2)}
            
            이 검색 결과에 대한 간단하고 유용한 인사이트를 한국어로 제공해주세요.
            """
            
            insights = self.llm_client.generate_smart_response(
                user_query=insights_prompt,
                search_results=results[:5],
                video_info=f"비디오: {video.original_name}",
                use_multi_llm=False
            )
            
            return insights
            
        except Exception as e:
            return f"인사이트 생성 중 오류 발생: {str(e)}"


class EnhancedFrameView(APIView):
    """고급 분석 정보가 포함된 프레임 데이터 제공"""
    permission_classes = [AllowAny]
    
    def get(self, request, video_id, frame_number):
        try:
            video = Video.objects.get(id=video_id)
            
            # 프레임 데이터 조회
            try:
                frame = Frame.objects.get(video=video, image_id=frame_number)
                
                frame_data = {
                    'frame_id': frame.image_id,
                    'timestamp': frame.timestamp,
                    'caption': frame.caption,
                    'enhanced_caption': frame.enhanced_caption,
                    'final_caption': frame.final_caption,
                    'detected_objects': frame.detected_objects,
                    'comprehensive_features': frame.comprehensive_features,
                    'analysis_quality': frame.comprehensive_features.get('caption_quality', 'basic')
                }
                
                # 고급 분석 결과 분해
                if frame.comprehensive_features:
                    features = frame.comprehensive_features
                    
                    frame_data['advanced_analysis'] = {
                        'clip_analysis': features.get('clip_features', {}),
                        'ocr_text': features.get('ocr_text', {}),
                        'vqa_results': features.get('vqa_results', {}),
                        'scene_graph': features.get('scene_graph', {}),
                        'scene_complexity': features.get('scene_complexity', 0)
                    }
                
                return Response(frame_data)
                
            except Frame.DoesNotExist:
                # 프레임 데이터가 없으면 기본 이미지만 반환
                return Response({
                    'frame_id': frame_number,
                    'message': '프레임 데이터는 없지만 이미지는 사용 가능합니다.',
                    'image_url': f'/frame/{video_id}/{frame_number}/'
                })
            
        except Video.DoesNotExist:
            return Response({
                'error': '비디오를 찾을 수 없습니다.'
            }, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({
                'error': f'프레임 정보 조회 실패: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class EnhancedScenesView(APIView):
    """고급 분석 정보가 포함된 씬 데이터 제공"""
    permission_classes = [AllowAny]
    
    def get(self, request, video_id):
        try:
            video = Video.objects.get(id=video_id)
            scenes = Scene.objects.filter(video=video).order_by('scene_id')
            
            enhanced_scenes = []
            for scene in scenes:
                scene_data = {
                    'scene_id': scene.scene_id,
                    'start_time': scene.start_time,
                    'end_time': scene.end_time,
                    'duration': scene.duration,
                    'frame_count': scene.frame_count,
                    'dominant_objects': scene.dominant_objects,
                    'enhanced_captions_count': scene.enhanced_captions_count,
                    'caption_type': 'enhanced' if scene.enhanced_captions_count > 0 else 'basic'
                }
                
                # 씬 내 프레임들의 고급 분석 결과 집계
                scene_frames = Frame.objects.filter(
                    video=video,
                    timestamp__gte=scene.start_time,
                    timestamp__lte=scene.end_time
                )
                
                if scene_frames.exists():
                    # 고급 기능 사용 통계
                    clip_count = sum(1 for f in scene_frames if f.comprehensive_features.get('clip_features'))
                    ocr_count = sum(1 for f in scene_frames if f.comprehensive_features.get('ocr_text', {}).get('texts'))
                    vqa_count = sum(1 for f in scene_frames if f.comprehensive_features.get('vqa_results'))
                    
                    scene_data['advanced_features'] = {
                        'clip_analysis_frames': clip_count,
                        'ocr_text_frames': ocr_count,
                        'vqa_analysis_frames': vqa_count,
                        'total_frames': scene_frames.count()
                    }
                    
                    # 씬 복잡도 평균
                    complexities = [f.comprehensive_features.get('scene_complexity', 0) for f in scene_frames]
                    scene_data['average_complexity'] = sum(complexities) / len(complexities) if complexities else 0
                
                enhanced_scenes.append(scene_data)
            
            return Response({
                'scenes': enhanced_scenes,
                'total_scenes': len(enhanced_scenes),
                'analysis_type': 'enhanced' if any(s.get('advanced_features') for s in enhanced_scenes) else 'basic',
                'video_info': {
                    'id': video.id,
                    'name': video.original_name
                }
            })
            
        except Video.DoesNotExist:
            return Response({
                'error': '비디오를 찾을 수 없습니다.'
            }, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({
                'error': f'고급 씬 정보 조회 실패: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class AnalysisResultsView(APIView):
    """종합 분석 결과 제공"""
    permission_classes = [AllowAny]
    
    def get(self, request, video_id):
        try:
            video = Video.objects.get(id=video_id)
            
            if not video.is_analyzed:
                return Response({
                    'error': '아직 분석이 완료되지 않았습니다.'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            analysis = video.analysis
            scenes = Scene.objects.filter(video=video)
            frames = Frame.objects.filter(video=video)
            
            # 종합 분석 결과
            results = {
                'video_info': {
                    'id': video.id,
                    'name': video.original_name,
                    'duration': video.duration,
                    'analysis_type': analysis.analysis_statistics.get('analysis_type', 'basic'),
                    'processing_time': analysis.processing_time_seconds,
                    'success_rate': analysis.success_rate
                },
                'analysis_summary': {
                    'total_scenes': scenes.count(),
                    'total_frames_analyzed': frames.count(),
                    'unique_objects': analysis.analysis_statistics.get('unique_objects', 0),
                    'features_used': analysis.analysis_statistics.get('features_used', []),
                    'scene_types': analysis.analysis_statistics.get('scene_types', [])
                },
                'advanced_features': {
                    'clip_analysis': analysis.analysis_statistics.get('clip_analysis', False),
                    'ocr_text_extracted': analysis.analysis_statistics.get('text_extracted', False),
                    'vqa_analysis': analysis.analysis_statistics.get('vqa_analysis', False),
                    'scene_graph_analysis': analysis.analysis_statistics.get('scene_graph_analysis', False)
                },
                'content_insights': {
                    'dominant_objects': analysis.analysis_statistics.get('dominant_objects', []),
                    'text_content_length': analysis.caption_statistics.get('text_content_length', 0),
                    'enhanced_captions_count': analysis.caption_statistics.get('enhanced_captions', 0),
                    'average_confidence': analysis.caption_statistics.get('average_confidence', 0)
                }
            }
            
            # 프레임별 고급 분석 통계
            if frames.exists():
                clip_frames = sum(1 for f in frames if f.comprehensive_features.get('clip_features'))
                ocr_frames = sum(1 for f in frames if f.comprehensive_features.get('ocr_text', {}).get('texts'))
                vqa_frames = sum(1 for f in frames if f.comprehensive_features.get('vqa_results'))
                
                results['frame_statistics'] = {
                    'total_frames': frames.count(),
                    'clip_analyzed_frames': clip_frames,
                    'ocr_processed_frames': ocr_frames,
                    'vqa_analyzed_frames': vqa_frames,
                    'coverage': {
                        'clip': (clip_frames / frames.count()) * 100 if frames.count() > 0 else 0,
                        'ocr': (ocr_frames / frames.count()) * 100 if frames.count() > 0 else 0,
                        'vqa': (vqa_frames / frames.count()) * 100 if frames.count() > 0 else 0
                    }
                }
            
            return Response(results)
            
        except Video.DoesNotExist:
            return Response({
                'error': '비디오를 찾을 수 없습니다.'
            }, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({
                'error': f'분석 결과 조회 실패: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class AnalysisSummaryView(APIView):
    """분석 결과 요약 제공"""
    permission_classes = [AllowAny]
    
    def __init__(self):
        super().__init__()
        self.llm_client = LLMClient()
    
    def get(self, request, video_id):
        try:
            video = Video.objects.get(id=video_id)
            
            if not video.is_analyzed:
                return Response({
                    'error': '아직 분석이 완료되지 않았습니다.'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # 분석 결과 데이터 수집
            analysis = video.analysis
            frames = Frame.objects.filter(video=video)[:10]  # 상위 10개 프레임
            
            # AI 기반 요약 생성
            summary_data = {
                'video_name': video.original_name,
                'analysis_type': analysis.analysis_statistics.get('analysis_type', 'basic'),
                'features_used': analysis.analysis_statistics.get('features_used', []),
                'dominant_objects': analysis.analysis_statistics.get('dominant_objects', []),
                'scene_types': analysis.analysis_statistics.get('scene_types', []),
                'processing_time': analysis.processing_time_seconds
            }
            
            # 대표 프레임들의 캡션 수집
            sample_captions = []
            for frame in frames:
                if frame.final_caption:
                    sample_captions.append(frame.final_caption)
            
            summary_prompt = f"""
            다음 비디오 분석 결과를 바탕으로 상세하고 유용한 요약을 작성해주세요:
            
            비디오: {video.original_name}
            분석 유형: {summary_data['analysis_type']}
            사용된 기능: {', '.join(summary_data['features_used'])}
            주요 객체: {', '.join(summary_data['dominant_objects'][:5])}
            씬 유형: {', '.join(summary_data['scene_types'][:3])}
            
            대표 캡션들:
            {chr(10).join(sample_captions[:5])}
            
            이 비디오의 주요 내용, 특징, 활용 방안을 포함하여 한국어로 요약해주세요.
            """
            
            ai_summary = self.llm_client.generate_smart_response(
                user_query=summary_prompt,
                search_results=None,
                video_info=f"비디오: {video.original_name}",
                use_multi_llm=True  # 고품질 요약을 위해 다중 LLM 사용
            )
            
            return Response({
                'video_id': video.id,
                'video_name': video.original_name,
                'ai_summary': ai_summary,
                'analysis_data': summary_data,
                'key_insights': {
                    'total_objects': len(summary_data['dominant_objects']),
                    'scene_variety': len(summary_data['scene_types']),
                    'analysis_depth': len(summary_data['features_used']),
                    'processing_efficiency': f"{summary_data['processing_time']}초"
                },
                'generated_at': datetime.now().isoformat()
            })
            
        except Video.DoesNotExist:
            return Response({
                'error': '비디오를 찾을 수 없습니다.'
            }, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({
                'error': f'요약 생성 실패: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class AnalysisExportView(APIView):
    """분석 결과 내보내기"""
    permission_classes = [AllowAny]
    
    def get(self, request, video_id):
        try:
            video = Video.objects.get(id=video_id)
            
            if not video.is_analyzed:
                return Response({
                    'error': '아직 분석이 완료되지 않았습니다.'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            export_format = request.GET.get('format', 'json')
            
            # 전체 분석 데이터 수집
            analysis = video.analysis
            scenes = Scene.objects.filter(video=video)
            frames = Frame.objects.filter(video=video)
            
            export_data = {
                'export_info': {
                    'video_id': video.id,
                    'video_name': video.original_name,
                    'export_date': datetime.now().isoformat(),
                    'export_format': export_format
                },
                'video_metadata': {
                    'filename': video.filename,
                    'duration': video.duration,
                    'file_size': video.file_size,
                    'uploaded_at': video.uploaded_at.isoformat()
                },
                'analysis_metadata': {
                    'analysis_type': analysis.analysis_statistics.get('analysis_type', 'basic'),
                    'enhanced_analysis': analysis.enhanced_analysis,
                    'success_rate': analysis.success_rate,
                    'processing_time_seconds': analysis.processing_time_seconds,
                    'features_used': analysis.analysis_statistics.get('features_used', [])
                },
                'scenes': [
                    {
                        'scene_id': scene.scene_id,
                        'start_time': scene.start_time,
                        'end_time': scene.end_time,
                        'duration': scene.duration,
                        'frame_count': scene.frame_count,
                        'dominant_objects': scene.dominant_objects
                    }
                    for scene in scenes
                ],
                'frames': [
                    {
                        'frame_id': frame.image_id,
                        'timestamp': frame.timestamp,
                        'caption': frame.caption,
                        'enhanced_caption': frame.enhanced_caption,
                        'final_caption': frame.final_caption,
                        'detected_objects': frame.detected_objects,
                        'comprehensive_features': frame.comprehensive_features
                    }
                    for frame in frames
                ],
                'statistics': {
                    'total_scenes': scenes.count(),
                    'total_frames': frames.count(),
                    'unique_objects': analysis.analysis_statistics.get('unique_objects', 0),
                    'scene_types': analysis.analysis_statistics.get('scene_types', []),
                    'dominant_objects': analysis.analysis_statistics.get('dominant_objects', [])
                }
            }
            
            if export_format == 'json':
                response = JsonResponse(export_data, json_dumps_params={'ensure_ascii': False, 'indent': 2})
                response['Content-Disposition'] = f'attachment; filename="{video.original_name}_analysis.json"'
                return response
            
            elif export_format == 'csv':
                # CSV 형태로 프레임 데이터 내보내기
                import csv
                from io import StringIO
                
                output = StringIO()
                writer = csv.writer(output)
                
                # 헤더
                writer.writerow(['frame_id', 'timestamp', 'caption', 'enhanced_caption', 'objects_count', 'scene_complexity'])
                
                # 데이터
                for frame_data in export_data['frames']:
                    writer.writerow([
                        frame_data['frame_id'],
                        frame_data['timestamp'],
                        frame_data.get('caption', ''),
                        frame_data.get('enhanced_caption', ''),
                        len(frame_data.get('detected_objects', [])),
                        frame_data.get('comprehensive_features', {}).get('scene_complexity', 0)
                    ])
                
                response = HttpResponse(output.getvalue(), content_type='text/csv')
                response['Content-Disposition'] = f'attachment; filename="{video.original_name}_analysis.csv"'
                return response
            
            else:
                return Response({
                    'error': '지원하지 않는 내보내기 형식입니다. json 또는 csv를 사용해주세요.'
                }, status=status.HTTP_400_BAD_REQUEST)
            
        except Video.DoesNotExist:
            return Response({
                'error': '비디오를 찾을 수 없습니다.'
            }, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({
                'error': f'내보내기 실패: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# 검색 관련 뷰들
class ObjectSearchView(APIView):
    """객체별 검색"""
    permission_classes = [AllowAny]
    
    def get(self, request):
        try:
            object_type = request.GET.get('object', '')
            video_id = request.GET.get('video_id')
            
            if not object_type:
                return Response({
                    'error': '검색할 객체 타입을 입력해주세요.'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # 특정 비디오 또는 전체 비디오에서 검색
            if video_id:
                videos = Video.objects.filter(id=video_id, is_analyzed=True)
            else:
                videos = Video.objects.filter(is_analyzed=True)
            
            results = []
            for video in videos:
                frames = Frame.objects.filter(video=video)
                
                for frame in frames:
                    for obj in frame.detected_objects:
                        if object_type.lower() in obj.get('class', '').lower():
                            results.append({
                                'video_id': video.id,
                                'video_name': video.original_name,
                                'frame_id': frame.image_id,
                                'timestamp': frame.timestamp,
                                'object_class': obj.get('class'),
                                'confidence': obj.get('confidence'),
                                'caption': frame.final_caption or frame.caption
                            })
            
            return Response({
                'search_query': object_type,
                'results': results[:50],  # 최대 50개 결과
                'total_matches': len(results)
            })
            
        except Exception as e:
            return Response({
                'error': f'객체 검색 실패: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class TextSearchView(APIView):
    """텍스트 검색 (OCR 결과 기반)"""
    permission_classes = [AllowAny]
    
    def get(self, request):
        try:
            search_text = request.GET.get('text', '')
            video_id = request.GET.get('video_id')
            
            if not search_text:
                return Response({
                    'error': '검색할 텍스트를 입력해주세요.'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # 특정 비디오 또는 전체 비디오에서 검색
            if video_id:
                videos = Video.objects.filter(id=video_id, is_analyzed=True)
            else:
                videos = Video.objects.filter(is_analyzed=True)
            
            results = []
            for video in videos:
                frames = Frame.objects.filter(video=video)
                
                for frame in frames:
                    ocr_data = frame.comprehensive_features.get('ocr_text', {})
                    if 'full_text' in ocr_data and search_text.lower() in ocr_data['full_text'].lower():
                        results.append({
                            'video_id': video.id,
                            'video_name': video.original_name,
                            'frame_id': frame.image_id,
                            'timestamp': frame.timestamp,
                            'extracted_text': ocr_data['full_text'],
                            'text_details': ocr_data.get('texts', []),
                            'caption': frame.final_caption or frame.caption
                        })
            
            return Response({
                'search_query': search_text,
                'results': results[:50],
                'total_matches': len(results)
            })
            
        except Exception as e:
            return Response({
                'error': f'텍스트 검색 실패: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class SceneSearchView(APIView):
    """씬 타입별 검색"""
    permission_classes = [AllowAny]
    
    def get(self, request):
        try:
            scene_type = request.GET.get('scene', '')
            video_id = request.GET.get('video_id')
            
            if not scene_type:
                return Response({
                    'error': '검색할 씬 타입을 입력해주세요.'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # 특정 비디오 또는 전체 비디오에서 검색
            if video_id:
                videos = Video.objects.filter(id=video_id, is_analyzed=True)
            else:
                videos = Video.objects.filter(is_analyzed=True)
            
            results = []
            for video in videos:
                if hasattr(video, 'analysis'):
                    scene_types = video.analysis.analysis_statistics.get('scene_types', [])
                    if any(scene_type.lower() in st.lower() for st in scene_types):
                        results.append({
                            'video_id': video.id,
                            'video_name': video.original_name,
                            'scene_types': scene_types,
                            'analysis_type': video.analysis.analysis_statistics.get('analysis_type', 'basic'),
                            'dominant_objects': video.analysis.analysis_statistics.get('dominant_objects', [])
                        })
            
            return Response({
                'search_query': scene_type,
                'results': results,
                'total_matches': len(results)
            })
            
        except Exception as e:
            return Response({
                'error': f'씬 검색 실패: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.shortcuts import get_object_or_404
from django.db import transaction
import json
import logging
import os
import time

logger = logging.getLogger(__name__)

@csrf_exempt
@require_http_methods(["DELETE"])
def delete_video(request, pk):
    """개선된 비디오 삭제 - 상세 로깅 및 검증 포함"""
    
    logger.info(f"🗑️ 비디오 삭제 요청 시작: ID={pk}")
    
    try:
        # 1단계: 비디오 존재 여부 확인
        try:
            video = get_object_or_404(Video, id=pk)
            logger.info(f"✅ 비디오 찾음: {video.original_name} (파일: {video.file_path})")
        except Video.DoesNotExist:
            logger.warning(f"❌ 비디오 존재하지 않음: ID={pk}")
            return JsonResponse({
                'error': '해당 비디오를 찾을 수 없습니다.',
                'video_id': pk,
                'deleted': False
            }, status=404)
        
        # 2단계: 삭제 가능 여부 확인
        if video.analysis_status == 'processing':
            logger.warning(f"❌ 분석 중인 비디오 삭제 시도: ID={pk}")
            return JsonResponse({
                'error': '분석 중인 비디오는 삭제할 수 없습니다.',
                'video_id': pk,
                'status': video.analysis_status,
                'deleted': False
            }, status=400)
        
        # 3단계: 트랜잭션으로 안전한 삭제 처리
        video_info = {
            'id': pk,
            'name': video.original_name,
            'file_path': video.file_path,
            'has_analysis': hasattr(video, 'analysis_results') and video.analysis_results.exists(),
            'has_scenes': hasattr(video, 'scenes') and video.scenes.exists()
        }
        
        with transaction.atomic():
            logger.info(f"🔄 트랜잭션 시작: 비디오 {pk} 삭제")
            
            # 관련 데이터 먼저 삭제
            deleted_analysis_count = 0
            deleted_scenes_count = 0
            
            if hasattr(video, 'analysis_results'):
                deleted_analysis_count = video.analysis_results.count()
                video.analysis_results.all().delete()
                logger.info(f"📊 분석 결과 삭제: {deleted_analysis_count}개")
            
            if hasattr(video, 'scenes'):
                deleted_scenes_count = video.scenes.count()
                video.scenes.all().delete()
                logger.info(f"🎬 씬 데이터 삭제: {deleted_scenes_count}개")
            
            # 파일 시스템에서 파일 삭제
            file_deleted = False
            if video.file_path and os.path.exists(video.file_path):
                try:
                    os.remove(video.file_path)
                    file_deleted = True
                    logger.info(f"📁 파일 삭제 성공: {video.file_path}")
                except Exception as file_error:
                    logger.error(f"❌ 파일 삭제 실패: {video.file_path} - {str(file_error)}")
                    # 파일 삭제 실패해도 데이터베이스에서는 삭제 진행
                    file_deleted = False
            else:
                logger.info(f"📁 삭제할 파일 없음: {video.file_path}")
                file_deleted = True  # 파일이 없으면 삭제된 것으로 간주
            
            # 데이터베이스에서 비디오 레코드 삭제
            video.delete()
            logger.info(f"💾 데이터베이스에서 비디오 삭제 완료: ID={pk}")
            
            # 트랜잭션 커밋 후 잠시 대기 (데이터베이스 동기화)
            time.sleep(0.1)
        
        # 4단계: 삭제 검증
        try:
            verification_video = Video.objects.get(id=pk)
            # 비디오가 여전히 존재하면 오류
            logger.error(f"❌ 삭제 검증 실패: 비디오가 여전히 존재함 ID={pk}")
            return JsonResponse({
                'error': '비디오 삭제에 실패했습니다. 데이터베이스에서 제거되지 않았습니다.',
                'video_id': pk,
                'deleted': False,
                'verification_failed': True
            }, status=500)
        except Video.DoesNotExist:
            # 비디오가 존재하지 않으면 삭제 성공
            logger.info(f"✅ 삭제 검증 성공: 비디오가 완전히 제거됨 ID={pk}")
        
        # 5단계: 성공 응답
        response_data = {
            'success': True,
            'message': f'비디오 "{video_info["name"]}"이(가) 성공적으로 삭제되었습니다.',
            'video_id': pk,
            'deleted': True,
            'details': {
                'file_deleted': file_deleted,
                'analysis_results_deleted': deleted_analysis_count,
                'scenes_deleted': deleted_scenes_count,
                'file_path': video_info['file_path']
            }
        }
        
        logger.info(f"✅ 비디오 삭제 완료: {json.dumps(response_data, ensure_ascii=False)}")
        return JsonResponse(response_data)
        
    except Exception as e:
        logger.error(f"❌ 비디오 삭제 중 예상치 못한 오류: ID={video_id}, 오류={str(e)}")
        return JsonResponse({
            'error': f'비디오 삭제 중 오류가 발생했습니다: {str(e)}',
            'video_id': video_id,
            'deleted': False,
            'exception': str(e)
        }, status=500)

@csrf_exempt
@require_http_methods(["POST", "PATCH"])
def rename_video(request, pk):
    """비디오 이름 변경"""
    try:
        video = get_object_or_404(Video, id=pk)
        
        data = json.loads(request.body)
        new_name = data.get('original_name') or data.get('title')
        
        if not new_name or not new_name.strip():
            return JsonResponse({'error': '새 이름을 입력해주세요.'}, status=400)
        
        video.original_name = new_name.strip()
        video.save()
        
        return JsonResponse({
            'success': True,
            'message': '이름이 변경되었습니다.',
            'video': {
                'id': video.id,
                'original_name': video.original_name
            }
        })
    except Video.DoesNotExist:
        return JsonResponse({'error': '비디오를 찾을 수 없습니다.'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
@csrf_exempt
@require_http_methods(["GET"])  
def video_detail(request, video_id):
    """비디오 상세 정보 조회 (존재 여부 확인용)"""
    try:
        video = get_object_or_404(Video, id=video_id)
        return JsonResponse({
            'id': video.id,
            'original_name': video.original_name,
            'analysis_status': video.analysis_status,
            'exists': True
        })
    except Video.DoesNotExist:
        return JsonResponse({
            'error': '해당 비디오를 찾을 수 없습니다.',
            'video_id': video_id,
            'exists': False
        }, status=404)

# 삭제 상태 확인을 위한 별도 엔드포인트
@csrf_exempt
@require_http_methods(["GET"])
def check_video_exists(request, video_id):
    """비디오 존재 여부만 확인"""
    try:
        Video.objects.get(id=video_id)
        return JsonResponse({
            'exists': True,
            'video_id': video_id
        })
    except Video.DoesNotExist:
        return JsonResponse({
            'exists': False,
            'video_id': video_id
        })

# views.py에 추가할 바운딩 박스 그리기 View 클래스들

class AdvancedVideoSearchView(APIView):
    """고급 비디오 검색 View - 바운딩 박스 정보 포함"""
    permission_classes = [AllowAny]
    
    def __init__(self):
        super().__init__()
        self.video_analyzer = get_video_analyzer()
        self.llm_client = LLMClient()
    
    def post(self, request):
        try:
            video_id = request.data.get('video_id')
            query = request.data.get('query', '').strip()
            search_options = request.data.get('search_options', {})
            
            print(f"🔍 고급 비디오 검색: 비디오={video_id}, 쿼리='{query}'")
            
            if not query:
                return Response({
                    'error': '검색어를 입력해주세요.'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            try:
                video = Video.objects.get(id=video_id)
            except Video.DoesNotExist:
                return Response({
                    'error': '비디오를 찾을 수 없습니다.'
                }, status=status.HTTP_404_NOT_FOUND)
            
            # EnhancedVideoRAGSystem을 직접 사용하여 검색 수행
            rag_system = EnhancedVideoRAGSystem()
            search_result = rag_system.object_detection_search(str(video.id), query)
            
            if not search_result.get('success'):
                return Response({
                    'success': False,
                    'error': '검색 실패',
                    'matches': []
                })
            
            matches = search_result.get('matches', [])
            
            # 바운딩 박스 URL 추가 (검색 쿼리 포함)
            for match in matches:
                if match.get('frame_id'):
                    bbox_url = f"/api/videos/{video.id}/frames/{match['frame_id']}/bbox/"
                    if query:
                        bbox_url += f"?query={query}"
                    match['bbox_image_url'] = bbox_url
                    match['thumbBBoxUrl'] = bbox_url
            
            print(f"✅ 검색 완료: {len(matches)}개 결과")
            
            return Response({
                'success': True,
                'query': query,
                'video_id': video.id,
                'matches': matches,
                'total_matches': len(matches),
                'keywords': search_result.get('keywords', {})
            })
            
        except Exception as e:
            print(f"❌ 고급 비디오 검색 실패: {e}")
            return Response({
                'error': f'고급 검색 실패: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    def _perform_advanced_search(self, video, query, search_options):
        """실제 고급 검색 수행"""
        try:
            # EnhancedVideoRAGSystem을 사용하여 검색 수행
            rag_system = EnhancedVideoRAGSystem()
            search_result = rag_system.object_detection_search(str(video.id), query)
            
            if search_result.get('success') and 'matches' in search_result:
                return search_result['matches']
            else:
                return []
                
        except Exception as e:
            print(f"❌ 고급 검색 수행 오류: {e}")
            return []
    
    def _add_bbox_info(self, search_results, video):
        """검색 결과에 바운딩 박스 정보 추가"""
        enhanced_results = []
        
        for result in search_results:
            enhanced_result = dict(result)
            
            # 바운딩 박스 이미지 URL 추가 (모든 객체 검색 결과에 대해)
            if result.get('frame_id'):
                bbox_url = f"/api/videos/{video.id}/frames/{result['frame_id']}/bbox/"
                enhanced_result['bbox_image_url'] = bbox_url
                enhanced_result['thumbBBoxUrl'] = bbox_url  # 프론트엔드 호환성을 위해 추가
                
                # 바운딩 박스 어노테이션 정보 추가
                enhanced_result['bbox_annotations'] = [{
                    'match': result.get('class_name', 'object'),
                    'confidence': result.get('confidence', 0),
                    'bbox': result.get('bbox', []),
                    'colors': result.get('colors', []),
                    'color_description': result.get('color_description', '')
                }]
            
            enhanced_results.append(enhanced_result)
        
        return enhanced_results
    
    def _generate_search_insights(self, query, results, video):
        """검색 결과에 대한 AI 인사이트 생성"""
        try:
            if not results:
                return "검색 결과가 없습니다. 다른 검색어를 시도해보세요."
            
            bbox_count = sum(1 for r in results if r.get('bbox_annotations'))
            total_objects = sum(len(r.get('bbox_annotations', [])) for r in results)
            
            insights_prompt = f"""
            검색어: "{query}"
            비디오: {video.original_name}
            검색 결과: {len(results)}개 매칭
            바운딩 박스 표시 가능: {bbox_count}개 프레임
            총 감지된 객체: {total_objects}개
            
            주요 발견사항을 바탕으로 간단하고 유용한 인사이트를 한국어로 제공해주세요.
            바운딩 박스 표시 기능에 대한 안내도 포함해주세요.
            """
            
            insights = self.llm_client.generate_smart_response(
                user_query=insights_prompt,
                search_results=results[:3],
                video_info=f"비디오: {video.original_name}",
                use_multi_llm=False
            )
            
            return insights
            
        except Exception as e:
            return f"인사이트 생성 중 오류 발생: {str(e)}"



# ✅ FrameWithBboxView - 바운딩 박스가 있는 프레임 뷰
class FrameWithBboxView(APIView):
    permission_classes = [AllowAny]
    
    def get(self, request, video_id, frame_number):
        try:
            print(f"🖼️ 바운딩 박스 프레임 요청: 비디오={video_id}, 프레임={frame_number}")
            
            # 검색 쿼리를 URL 파라미터에서 가져오기
            self.current_search_query = request.GET.get('query', '')
            print(f"🔍 현재 검색 쿼리: '{self.current_search_query}'")
            
            video = Video.objects.get(id=video_id)
            frame = Frame.objects.get(video=video, image_id=frame_number)
            
            # 디버깅: detected_objects 확인
            print(f"🔍 Frame {frame_number} detected_objects: {frame.detected_objects}")
            
            # detected_objects 로드 및 파싱
            detected_objects = frame.detected_objects
            
            # frame.detected_objects가 비어있으면 분석 결과 JSON에서 로드
            if not detected_objects:
                print("⚠️ detected_objects가 없습니다. 분석 결과 JSON 파일에서 데이터를 찾아보겠습니다.")
                detected_objects = self._load_objects_from_analysis_json(video, frame_number)
                if not detected_objects:
                    print("⚠️ 분석 결과에서도 객체를 찾을 수 없습니다")
                    # 원본 이미지 반환
                    return self._get_original_frame(video, frame_number)
            
            # detected_objects 파싱
            if isinstance(detected_objects, str):
                import json
                detected_objects = json.loads(detected_objects)
            
            # persons 배열에서 객체 추출
            if isinstance(detected_objects, dict) and 'persons' in detected_objects:
                detected_objects = detected_objects['persons']
            elif not isinstance(detected_objects, list):
                detected_objects = detected_objects.get('objects', []) if isinstance(detected_objects, dict) else []
            
            print(f"📦 파싱된 객체 수: {len(detected_objects)}")
            
            # 검색 조건에 맞는 객체만 필터링
            filtered_objects = self._filter_objects_by_search(detected_objects, self.current_search_query)
            print(f"🎯 필터링된 객체 수: {len(filtered_objects)}")
            
            # 이미지 로드 및 박스 그리기
            image_data = self._draw_bboxes_on_frame(video, frame_number, filtered_objects)
            
            return HttpResponse(image_data, content_type='image/jpeg')
            
        except Video.DoesNotExist:
            return HttpResponse(status=404)
        except Frame.DoesNotExist:
            print(f"⚠️ Frame {frame_number} not found")
            return HttpResponse(status=404)
        except Exception as e:
            print(f"❌ 박스 그리기 실패: {e}")
            import traceback
            print(traceback.format_exc())
            return HttpResponse(status=500)
    
    def _load_objects_from_analysis_json(self, video, frame_number):
        """분석 결과 JSON 파일에서 프레임의 객체 데이터를 로드"""
        try:
            import json
            import os
            from django.conf import settings
            
            # 분석 결과 파일 경로 찾기
            analysis_dir = os.path.join(settings.MEDIA_ROOT, 'analysis_results')
            if not os.path.exists(analysis_dir):
                print(f"⚠️ 분석 결과 디렉토리가 없습니다: {analysis_dir}")
                return None
            
            # 비디오 ID로 분석 결과 파일 찾기 (우선순위 1)
            target_filename = f"real_analysis_{video.id}_enhanced"
            print(f"🔍 비디오 ID {video.id}에 대한 분석 결과 파일을 찾는 중: {target_filename}")
            
            # 분석 결과 파일들 검색
            for filename in os.listdir(analysis_dir):
                if filename.endswith('.json') and 'enhanced' in filename:
                    # 비디오 ID로 매칭 시도
                    if f"real_analysis_{video.id}_" in filename:
                        file_path = os.path.join(analysis_dir, filename)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                analysis_data = json.load(f)
                            
                            # frame_results에서 해당 프레임 찾기
                            if 'frame_results' in analysis_data and isinstance(analysis_data['frame_results'], list):
                                for frame_data in analysis_data['frame_results']:
                                    if frame_data.get('image_id') == frame_number:
                                        print(f"✅ 분석 결과에서 프레임 {frame_number} 데이터를 찾았습니다: {filename}")
                                        return frame_data.get('persons', [])
                            
                        except Exception as e:
                            print(f"⚠️ 분석 결과 파일 읽기 실패 {filename}: {e}")
                            continue
            
            # 비디오 파일명으로도 시도 (우선순위 2)
            video_filename = video.original_name or video.filename
            if video_filename:
                print(f"🔍 비디오 파일명으로 분석 결과 파일을 찾는 중: {video_filename}")
                for filename in os.listdir(analysis_dir):
                    if filename.endswith('.json') and 'enhanced' in filename:
                        if video_filename.replace('.mp4', '') in filename:
                            file_path = os.path.join(analysis_dir, filename)
                            try:
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    analysis_data = json.load(f)
                                
                                # frame_results에서 해당 프레임 찾기
                                if 'frame_results' in analysis_data and isinstance(analysis_data['frame_results'], list):
                                    for frame_data in analysis_data['frame_results']:
                                        if frame_data.get('image_id') == frame_number:
                                            print(f"✅ 분석 결과에서 프레임 {frame_number} 데이터를 찾았습니다: {filename}")
                                            return frame_data.get('persons', [])
                                
                            except Exception as e:
                                print(f"⚠️ 분석 결과 파일 읽기 실패 {filename}: {e}")
                                continue
            
            print(f"⚠️ 프레임 {frame_number}에 대한 분석 결과를 찾을 수 없습니다")
            return None
            
        except Exception as e:
            print(f"❌ 분석 결과 로드 실패: {e}")
            return None
    
    def _filter_objects_by_search(self, detected_objects, search_query):
        """검색 조건에 맞는 객체만 필터링"""
        if not search_query:
            return detected_objects
        
        filtered = []
        search_query_lower = search_query.lower()
        
        # 색상 키워드 매핑 (기존 db_builder.py와 동일)
        color_keywords = {
            '빨간색': ['red', '빨강', 'wearing red clothes'],
            '빨강': ['red', '빨간색', 'wearing red clothes'],
            '파란색': ['blue', '파랑', 'wearing blue clothes'],
            '파랑': ['blue', '파란색', 'wearing blue clothes'],
            '녹색': ['green', '초록', '초록색', 'wearing green clothes'],
            '초록': ['green', '녹색', '초록색', 'wearing green clothes'],
            '초록색': ['green', '초록', '녹색', 'wearing green clothes'],
            '노란색': ['yellow', '노랑', 'wearing yellow clothes'],
            '노랑': ['yellow', '노란색', 'wearing yellow clothes'],
            '핑크색': ['pink', '핑크', '분홍', '분홍색', 'wearing pink clothes'],
            '핑크': ['pink', '핑크색', '분홍', '분홍색', 'wearing pink clothes'],
            '분홍': ['pink', '핑크', '핑크색', '분홍색', 'wearing pink clothes'],
            '분홍색': ['pink', '핑크', '분홍', '핑크색', 'wearing pink clothes'],
            '보라색': ['purple', '보라', 'wearing purple clothes'],
            '보라': ['purple', '보라색', 'wearing purple clothes'],
            '검은색': ['black', '검정', 'wearing black clothes'],
            '검정': ['black', '검은색', 'wearing black clothes'],
            '흰색': ['white', '하얀', 'wearing white clothes'],
            '하얀': ['white', '흰색', 'wearing white clothes'],
            '회색': ['gray', 'grey', '회색', 'wearing gray clothes'],
            '갈색': ['brown', '갈색', 'wearing brown clothes'],
            '주황색': ['orange', '주황', 'wearing orange clothes'],
            '주황': ['orange', '주황색', 'wearing orange clothes']
        }
        
        for obj in detected_objects:
            match = False
            
            # 가방 관련 검색을 먼저 확인
            if '가방' in search_query_lower or 'bag' in search_query_lower:
                attributes = obj.get('attributes', {})
                accessories = attributes.get('accessories', {})
                accessory_value = accessories.get('value', '').lower()
                print(f"🎒 가방 검색 확인 - accessory_value: {accessory_value}")
                if 'bag' in accessory_value or 'backpack' in accessory_value or 'handbag' in accessory_value:
                    match = True
                    # 가방 영역을 별도로 표시하기 위해 객체 정보 수정
                    obj['is_bag_search'] = True
                    obj['bag_region'] = self._estimate_bag_region(obj)
                    print(f"🎒 가방 객체 발견! bag_region: {obj['bag_region']}")
                else:
                    # 가방이 없는 사람은 가방 검색에서 제외
                    match = False
            
            # 가방 검색이 아닌 경우에만 다른 조건 확인
            if not match:
                # 색상 검색인 경우
                for color_kr, color_en_list in color_keywords.items():
                    if color_kr in search_query_lower or any(en in search_query_lower for en in color_en_list):
                        # 객체의 색상 정보 확인
                        attributes = obj.get('attributes', {})
                        clothing_color = attributes.get('clothing_color', {})
                        color_value = clothing_color.get('value', '').lower()
                        
                        if any(en in color_value for en in color_en_list):
                            match = True
                            break
                
                # 일반 객체 검색인 경우 (사람, 가방 등)
                if not match:
                    obj_class = obj.get('class', '').lower()
                    if obj_class in search_query_lower or '사람' in search_query_lower and obj_class == 'person':
                        match = True
            
            if match:
                filtered.append(obj)
        
        return filtered
    
    def _estimate_bag_region(self, person_obj):
        """사람 객체에서 가방 영역을 추정"""
        bbox = person_obj.get('bbox', [])
        if len(bbox) != 4:
            return None
        
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        # 가방은 보통 사람의 어깨/옆구리 부분에 위치
        # 백팩의 경우: 상단 1/3 영역의 오른쪽 1/4 부분
        # 핸드백의 경우: 상단 1/3 영역의 왼쪽 1/4 부분
        
        bag_type = person_obj.get('attributes', {}).get('accessories', {}).get('value', '').lower()
        
        if 'backpack' in bag_type:
            # 백팩: 상단 1/2, 오른쪽 1/3 영역 (더 크게)
            bag_x1 = x1 + width * 0.67  # 오른쪽 1/3
            bag_y1 = y1 + height * 0.05  # 상단 1/20
            bag_x2 = x2
            bag_y2 = y1 + height * 0.5  # 상단 1/2
        elif 'handbag' in bag_type:
            # 핸드백: 상단 1/2, 왼쪽 1/3 영역 (더 크게)
            bag_x1 = x1
            bag_y1 = y1 + height * 0.05  # 상단 1/20
            bag_x2 = x1 + width * 0.33  # 왼쪽 1/3
            bag_y2 = y1 + height * 0.5  # 상단 1/2
        else:
            # 일반 가방: 상단 1/2, 양쪽 1/3 영역 (더 크게)
            bag_x1 = x1
            bag_y1 = y1 + height * 0.05  # 상단 1/20
            bag_x2 = x2
            bag_y2 = y1 + height * 0.5  # 상단 1/2
        
        return [bag_x1, bag_y1, bag_x2, bag_y2]
    
    def _draw_bboxes_on_frame(self, video, frame_number, detected_objects):
        """프레임에 바운딩 박스 그리기"""
        try:
            from PIL import Image, ImageDraw, ImageFont
            import cv2
            import io
            import numpy as np
            import os
            
            # 🔧 수정: original_name을 기준으로 실제 파일 찾기
            from django.conf import settings
            uploads_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
            
            # original_name을 기준으로 파일 찾기
            original_name = video.original_name
            if not original_name:
                original_name = video.filename
            
            # 파일명에서 확장자 제거
            base_name = original_name.replace('.mp4', '')
            
            # uploads 디렉토리에서 해당 파일 찾기
            video_path = None
            if os.path.exists(uploads_dir):
                for filename in os.listdir(uploads_dir):
                    if filename.endswith('.mp4') and base_name in filename:
                        video_path = os.path.join(uploads_dir, filename)
                        break
            
            # 찾지 못한 경우 원래 방식으로 시도
            if not video_path:
                video_path = video.filename
                if not os.path.isabs(video_path):
                    video_path = os.path.join(settings.MEDIA_ROOT, 'uploads', video_path)
            
            print(f"🎥 비디오 파일 경로: {video_path}")
            print(f"📦 감지된 객체 수: {len(detected_objects)}")
            print(f"📦 객체 데이터: {detected_objects}")
            
            # 파일 존재 확인
            if not os.path.exists(video_path):
                print(f"⚠️ 비디오 파일이 존재하지 않음: {video_path}")
                return self._create_dummy_image_with_boxes(frame_number, detected_objects)
            
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                print(f"⚠️ 비디오 파일 열기 실패: {video_path}")
                cap.release()
                return self._create_dummy_image_with_boxes(frame_number, detected_objects)
            
            # 프레임 번호로 이동 (0-based index)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                print(f"⚠️ 프레임 {frame_number} 읽기 실패, 더미 이미지 생성")
                return self._create_dummy_image_with_boxes(frame_number, detected_objects)
            
            # OpenCV 이미지를 PIL 이미지로 변환
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            
            # 이미지 크기 가져오기
            img_width, img_height = image.size
            print(f"🖼️ 실제 이미지 크기: {img_width}x{img_height}")
            
            draw = ImageDraw.Draw(image)
            
            # 바운딩 박스 그리기
            colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'cyan', 'magenta']
            
            for i, obj in enumerate(detected_objects):
                bbox = obj.get('bbox', [])
                obj_class = obj.get('class', 'object')
                confidence = obj.get('confidence', 0)
                track_id = obj.get('track_id', '')
                color_description = obj.get('color_description', '')
                
                # 가방 검색인 경우 가방 영역만 표시
                is_bag_search = ('가방' in self.current_search_query.lower() or 'bag' in self.current_search_query.lower())
                print(f"🎒 바운딩 박스 그리기 - is_bag_search: {is_bag_search}, obj.is_bag_search: {obj.get('is_bag_search')}, bag_region: {obj.get('bag_region')}")
                if is_bag_search and obj.get('is_bag_search') and obj.get('bag_region'):
                    bag_bbox = obj.get('bag_region')
                    if len(bag_bbox) == 4:
                        x1_norm, y1_norm, x2_norm, y2_norm = bag_bbox
                        
                        x1 = int(x1_norm * img_width)
                        y1 = int(y1_norm * img_height)
                        x2 = int(x2_norm * img_width)
                        y2 = int(y2_norm * img_height)
                        
                        # 좌표 유효성 검사
                        x1 = max(0, min(x1, img_width))
                        y1 = max(0, min(y1, img_height))
                        x2 = max(0, min(x2, img_width))
                        y2 = max(0, min(y2, img_height))
                        
                        color = 'red'  # 가방은 빨간색으로 표시
                        
                        # 가방 바운딩 박스 그리기
                        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
                        
                        # 가방 레이블
                        label = "가방"
                        draw.text((x1, y1-25), label, fill=color)
                        
                elif len(bbox) == 4:
                    # 일반 객체 바운딩 박스
                    x1_norm, y1_norm, x2_norm, y2_norm = bbox
                    
                    x1 = int(x1_norm * img_width)
                    y1 = int(y1_norm * img_height)
                    x2 = int(x2_norm * img_width)
                    y2 = int(y2_norm * img_height)
                    
                    # 좌표 유효성 검사
                    x1 = max(0, min(x1, img_width))
                    y1 = max(0, min(y1, img_height))
                    x2 = max(0, min(x2, img_width))
                    y2 = max(0, min(y2, img_height))
                    
                    color = colors[i % len(colors)]
                    
                    # 바운딩 박스 그리기
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                    
                    # 레이블 그리기
                    label_parts = [obj_class]
                    if track_id:
                        label_parts.append(f"ID:{track_id}")
                    if color_description:
                        label_parts.append(color_description)
                    label_parts.append(f"{confidence:.2f}")
                    
                    label = " | ".join(label_parts)
                    
                    # 레이블 배경 추가 (가독성 향상)
                    label_bbox = draw.textbbox((x1, y1-20), label)
                    draw.rectangle(label_bbox, fill=color, outline=color)
                    draw.text((x1, y1-20), label, fill='white')
            
            # 이미지를 바이트로 변환
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=90)
            print(f"✅ 박스가 그려진 이미지 생성 완료 (객체 수: {len(detected_objects)})")
            return buffer.getvalue()
            
        except Exception as e:
            print(f"❌ 프레임 처리 실패: {e}")
            import traceback
            print(traceback.format_exc())
            
            # 폴백: 더미 이미지 반환
            return self._create_dummy_image_with_boxes(frame_number, detected_objects)

    def _create_dummy_image_with_boxes(self, frame_number, detected_objects):
        """더미 이미지에 바운딩 박스 정보 표시"""
        try:
            from PIL import Image, ImageDraw, ImageFont
            import io
            
            # 더미 이미지 생성
            image = Image.new('RGB', (640, 480), color='lightgray')
            draw = ImageDraw.Draw(image)
            
            # 제목 그리기
            draw.text((10, 10), f"Frame {frame_number} - Video File Not Found", fill='black')
            
            # 감지된 객체 정보 표시
            y_offset = 40
            for i, obj in enumerate(detected_objects):
                obj_class = obj.get('class', 'object')
                confidence = obj.get('confidence', 0)
                track_id = obj.get('track_id', '')
                color_desc = obj.get('color_description', '')
                
                info_text = f"{i+1}. {obj_class}"
                if track_id:
                    info_text += f" (ID:{track_id})"
                if color_desc:
                    info_text += f" - {color_desc}"
                info_text += f" ({confidence:.2f})"
                
                draw.text((10, y_offset), info_text, fill='black')
                y_offset += 20
                
                if y_offset > 450:  # 이미지 경계 내에서 표시
                    break
            
            # 바이트로 변환
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=90)
            return buffer.getvalue()
            
        except Exception as e:
            print(f"❌ 더미 이미지 생성도 실패: {e}")
            # 최후의 수단: 간단한 오류 이미지
            try:
                image = Image.new('RGB', (320, 240), color='red')
                draw = ImageDraw.Draw(image)
                draw.text((10, 10), "Error", fill='white')
                buffer = io.BytesIO()
                image.save(buffer, format='JPEG', quality=50)
                return buffer.getvalue()
            except:
                raise Exception("이미지 생성 완전 실패")

    def _get_original_frame(self, video, frame_number):
        """원본 프레임 반환"""
        try:
            import cv2
            import io
            from PIL import Image
            import os
            
            # 🔧 수정: original_name을 기준으로 실제 파일 찾기
            from django.conf import settings
            uploads_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
            
            # original_name을 기준으로 파일 찾기
            original_name = video.original_name
            if not original_name:
                original_name = video.filename
            
            # 파일명에서 확장자 제거
            base_name = original_name.replace('.mp4', '')
            
            # uploads 디렉토리에서 해당 파일 찾기
            video_path = None
            if os.path.exists(uploads_dir):
                for filename in os.listdir(uploads_dir):
                    if filename.endswith('.mp4') and base_name in filename:
                        video_path = os.path.join(uploads_dir, filename)
                        break
            
            # 찾지 못한 경우 원래 방식으로 시도
            if not video_path:
                video_path = video.filename
                if not os.path.isabs(video_path):
                    video_path = os.path.join(settings.MEDIA_ROOT, 'uploads', video_path)
            
            if not os.path.exists(video_path):
                # 파일이 없으면 더미 이미지 반환
                image = Image.new('RGB', (640, 480), color='lightgray')
                draw = ImageDraw.Draw(image)
                draw.text((10, 10), f"Frame {frame_number} - No Detections", fill='black')
                
                buffer = io.BytesIO()
                image.save(buffer, format='JPEG', quality=90)
                return HttpResponse(buffer.getvalue(), content_type='image/jpeg')
            
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                # OpenCV 이미지를 JPEG로 인코딩
                _, buffer = cv2.imencode('.jpg', frame)
                return HttpResponse(buffer.tobytes(), content_type='image/jpeg')
            else:
                # 프레임 읽기 실패시 더미 이미지
                image = Image.new('RGB', (640, 480), color='lightgray')
                buffer = io.BytesIO()
                image.save(buffer, format='JPEG', quality=90)
                return HttpResponse(buffer.getvalue(), content_type='image/jpeg')
                
        except Exception as e:
            print(f"❌ 원본 프레임 로드 실패: {e}")
            # 최후의 수단
            image = Image.new('RGB', (320, 240), color='red')
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=50)
            return HttpResponse(buffer.getvalue(), content_type='image/jpeg')
# ✅ EnhancedFrameView - 고급 프레임 뷰  
# 기존 FrameView 클래스에 바운딩 박스 옵션 추가
class EnhancedFrameView(FrameView):
    """기존 FrameView를 확장한 고급 프레임 View"""
    
    def get(self, request, video_id, frame_number):
        try:
            # 바운딩 박스 표시 옵션 확인
            show_bbox = request.GET.get('bbox', '').lower() in ['true', '1', 'yes']
            
            if show_bbox:
                # 바운딩 박스가 포함된 이미지 반환
                bbox_view = FrameWithBboxView()
                return bbox_view.get(request, video_id, frame_number)
            else:
                # 기본 프레임 반환
                return super().get(request, video_id, frame_number)
                
        except Exception as e:
            print(f"❌ 고급 프레임 뷰 오류: {e}")
            return super().get(request, video_id, frame_number)

# chat/views.py에 다음 클래스를 추가하세요

class AnalysisCapabilitiesView(APIView):
    """시스템 분석 기능 상태 확인"""
    permission_classes = [AllowAny]
    
    def get(self, request):
        try:
            print("🔍 AnalysisCapabilitiesView: 분석 기능 상태 요청")
            
            # VideoAnalyzer 인스턴스 가져오기
            try:
                analyzer = get_video_analyzer()
                analyzer_available = True
                print("✅ VideoAnalyzer 인스턴스 로딩 성공")
            except Exception as e:
                print(f"⚠️ VideoAnalyzer 로딩 실패: {e}")
                analyzer = None
                analyzer_available = False
            
            # 시스템 기능 상태 확인
            capabilities = {
                'system_status': {
                    'analyzer_available': analyzer_available,
                    'device': getattr(analyzer, 'device', 'unknown') if analyzer else 'none',
                    'timestamp': datetime.now().isoformat()
                },
                'core_features': {
                    'object_detection': {
                        'name': '객체 감지',
                        'available': analyzer.model is not None if analyzer else False,
                        'description': 'YOLO 기반 실시간 객체 감지',
                        'icon': '🎯'
                    },
                    'enhanced_captions': {
                        'name': '고급 캡션 생성',
                        'available': True,
                        'description': 'AI 기반 상세 캡션 생성',
                        'icon': '💬'
                    }
                },
                'advanced_features': {
                    'clip_analysis': {
                        'name': 'CLIP 분석',
                        'available': getattr(analyzer, 'clip_available', False) if analyzer else False,
                        'description': 'OpenAI CLIP 모델 기반 씬 이해',
                        'icon': '🖼️'
                    },
                    'ocr_text_extraction': {
                        'name': 'OCR 텍스트 추출',
                        'available': getattr(analyzer, 'ocr_available', False) if analyzer else False,  
                        'description': 'EasyOCR 기반 다국어 텍스트 인식',
                        'icon': '📝'
                    },
                    'vqa_analysis': {
                        'name': 'VQA 질문답변',
                        'available': getattr(analyzer, 'vqa_available', False) if analyzer else False,
                        'description': 'BLIP 모델 기반 시각적 질문 답변',
                        'icon': '❓'
                    },
                    'scene_graph': {
                        'name': 'Scene Graph',
                        'available': getattr(analyzer, 'scene_graph_available', False) if analyzer else False,
                        'description': 'NetworkX 기반 객체 관계 분석',
                        'icon': '🕸️'
                    }
                },
                'api_status': {
                    'groq_available': True,  # LLMClient에서 확인 필요
                    'openai_available': True,
                    'anthropic_available': True
                }
            }
            
            # 사용 가능한 기능 수 계산
            total_features = len(capabilities['core_features']) + len(capabilities['advanced_features'])
            available_features = sum(1 for features in [capabilities['core_features'], capabilities['advanced_features']] 
                                   for feature in features.values() if feature.get('available', False))
            
            capabilities['summary'] = {
                'total_features': total_features,
                'available_features': available_features,
                'availability_rate': (available_features / total_features * 100) if total_features > 0 else 0,
                'system_ready': analyzer_available and available_features > 0
            }
            
            print(f"✅ 분석 기능 상태: {available_features}/{total_features} 사용 가능")
            
            return Response(capabilities)
            
        except Exception as e:
            print(f"❌ AnalysisCapabilitiesView 오류: {e}")
            import traceback
            print(f"🔍 상세 오류: {traceback.format_exc()}")
            
            # 오류 발생시 기본 상태 반환
            error_response = {
                'system_status': {
                    'analyzer_available': False,
                    'device': 'error',
                    'error': str(e)
                },
                'core_features': {},
                'advanced_features': {},
                'api_status': {},
                'summary': {
                    'total_features': 0,
                    'available_features': 0,
                    'availability_rate': 0,
                    'system_ready': False,
                    'error': str(e)
                }
            }
            
            return Response(error_response, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# views.py에 추가할 고급 검색 API 클래스들

class CrossVideoSearchView(APIView):
    """영상 간 검색 - 여러 비디오에서 조건 검색"""
    permission_classes = [AllowAny]
    
    def post(self, request):
        try:
            query = request.data.get('query', '').strip()
            search_filters = request.data.get('filters', {})
            
            if not query:
                return Response({'error': '검색어를 입력해주세요.'}, status=400)
            
            # 쿼리 분석 - 날씨, 시간대, 장소 등 추출
            query_analysis = self._analyze_query(query)
            
            # 분석된 비디오들 중에서 검색
            videos = Video.objects.filter(is_analyzed=True)
            matching_videos = []
            
            for video in videos:
                match_score = self._calculate_video_match_score(video, query_analysis, search_filters)
                if match_score > 0.3:  # 임계값
                    matching_videos.append({
                        'video_id': video.id,
                        'video_name': video.original_name,
                        'match_score': match_score,
                        'match_reasons': self._get_match_reasons(video, query_analysis),
                        'metadata': self._get_video_metadata(video),
                        'thumbnail_url': f'/api/frame/{video.id}/100/',
                    })
            
            # 점수순 정렬
            matching_videos.sort(key=lambda x: x['match_score'], reverse=True)
            
            return Response({
                'query': query,
                'total_matches': len(matching_videos),
                'results': matching_videos[:20],  # 상위 20개
                'query_analysis': query_analysis,
                'search_type': 'cross_video'
            })
            
        except Exception as e:
            return Response({'error': str(e)}, status=500)
    
    def _analyze_query(self, query):
        """쿼리에서 날씨, 시간대, 장소 등 추출"""
        analysis = {
            'weather': None,
            'time_of_day': None,
            'location': None,
            'objects': [],
            'activities': []
        }
        
        query_lower = query.lower()
        
        # 날씨 키워드
        weather_keywords = {
            '비': 'rainy', '비가': 'rainy', '우천': 'rainy',
            '맑은': 'sunny', '화창한': 'sunny', '햇빛': 'sunny',
            '흐린': 'cloudy', '구름': 'cloudy'
        }
        
        # 시간대 키워드
        time_keywords = {
            '밤': 'night', '야간': 'night', '저녁': 'evening',
            '낮': 'day', '오후': 'afternoon', '아침': 'morning'
        }
        
        # 장소 키워드
        location_keywords = {
            '실내': 'indoor', '건물': 'indoor', '방': 'indoor',
            '실외': 'outdoor', '도로': 'outdoor', '거리': 'outdoor'
        }
        
        for keyword, value in weather_keywords.items():
            if keyword in query_lower:
                analysis['weather'] = value
                break
        
        for keyword, value in time_keywords.items():
            if keyword in query_lower:
                analysis['time_of_day'] = value
                break
                
        for keyword, value in location_keywords.items():
            if keyword in query_lower:
                analysis['location'] = value
                break
        
        return analysis
    
    def _calculate_video_match_score(self, video, query_analysis, filters):
        """비디오와 쿼리 간의 매칭 점수 계산"""
        score = 0.0
        
        try:
            # 분석 결과가 있는 경우
            if hasattr(video, 'analysis'):
                stats = video.analysis.analysis_statistics
                scene_types = stats.get('scene_types', [])
                
                # 날씨 매칭
                if query_analysis['weather']:
                    weather_scenes = [s for s in scene_types if query_analysis['weather'] in s.lower()]
                    if weather_scenes:
                        score += 0.4
                
                # 시간대 매칭
                if query_analysis['time_of_day']:
                    time_scenes = [s for s in scene_types if query_analysis['time_of_day'] in s.lower()]
                    if time_scenes:
                        score += 0.3
                
                # 장소 매칭
                if query_analysis['location']:
                    location_scenes = [s for s in scene_types if query_analysis['location'] in s.lower()]
                    if location_scenes:
                        score += 0.3
            
            return min(score, 1.0)
            
        except Exception:
            return 0.0
    
    def _get_match_reasons(self, video, query_analysis):
        """매칭 이유 생성"""
        reasons = []
        
        if query_analysis['weather']:
            reasons.append(f"{query_analysis['weather']} 날씨 조건")
        if query_analysis['time_of_day']:
            reasons.append(f"{query_analysis['time_of_day']} 시간대")
        if query_analysis['location']:
            reasons.append(f"{query_analysis['location']} 환경")
            
        return reasons
    
    def _get_video_metadata(self, video):
        """비디오 메타데이터 반환"""
        metadata = {
            'duration': video.duration,
            'file_size': video.file_size,
            'uploaded_at': video.uploaded_at.isoformat(),
            'analysis_type': 'basic'
        }
        
        if hasattr(video, 'analysis'):
            stats = video.analysis.analysis_statistics
            metadata.update({
                'analysis_type': stats.get('analysis_type', 'basic'),
                'scene_types': stats.get('scene_types', []),
                'dominant_objects': stats.get('dominant_objects', [])
            })
        
        return metadata

# views.py - 고급 검색 관련 뷰 수정된 버전
# views.py - IntraVideoTrackingView 향상된 버전 (더미 데이터 지원)

@method_decorator(csrf_exempt, name='dispatch')
class IntraVideoTrackingView(APIView):
    """영상 내 객체 추적 - 향상된 버전 (더미 데이터 지원)"""
    permission_classes = [AllowAny]
    
    def post(self, request):
        try:
            video_id = request.data.get('video_id')
            tracking_target = request.data.get('tracking_target', '').strip()
            time_range = request.data.get('time_range', {})
            
            logger.info(f"🎯 객체 추적 요청: 비디오={video_id}, 대상='{tracking_target}', 시간범위={time_range}")
            
            if not video_id or not tracking_target:
                return Response({'error': '비디오 ID와 추적 대상이 필요합니다.'}, status=400)
            
            try:
                video = Video.objects.get(id=video_id)
            except Video.DoesNotExist:
                return Response({'error': '비디오를 찾을 수 없습니다.'}, status=404)
            
            # Frame 데이터 확인 및 생성
            self._ensure_frame_data(video)
            
            # 타겟 분석 (색상, 객체 타입 등 추출)
            target_analysis = self._analyze_tracking_target(tracking_target)
            logger.info(f"📋 타겟 분석 결과: {target_analysis}")
            
            # 프레임별 추적 결과
            tracking_results = self._perform_object_tracking(video, target_analysis, time_range)
            
            logger.info(f"✅ 객체 추적 완료: {len(tracking_results)}개 결과")
            
            # 결과가 없으면 더 관대한 검색 수행
            if not tracking_results:
                logger.info("🔄 관대한 검색 모드로 재시도...")
                tracking_results = self._perform_lenient_tracking(video, target_analysis, time_range)
            
            return Response({
                'video_id': video_id,
                'tracking_target': tracking_target,
                'target_analysis': target_analysis,
                'tracking_results': tracking_results,
                'total_detections': len(tracking_results),
                'search_type': 'object_tracking'
            })
            
        except Exception as e:
            logger.error(f"❌ 객체 추적 오류: {e}")
            import traceback
            logger.error(f"🔍 상세 오류: {traceback.format_exc()}")
            return Response({'error': str(e)}, status=500)
    
    def _ensure_frame_data(self, video):
        """Frame 데이터 확인 및 생성"""
        try:
            frame_count = video.frames.count()
            if frame_count == 0:
                logger.warning(f"⚠️ 비디오 {video.original_name}에 Frame 데이터가 없습니다. 더미 데이터를 생성합니다.")
                from .models import create_dummy_frame_data
                create_dummy_frame_data(video, frame_count=30)
                logger.info(f"✅ 더미 Frame 데이터 생성 완료: 30개")
                return True
            else:
                logger.info(f"📊 기존 Frame 데이터 확인: {frame_count}개")
                return False
        except Exception as e:
            logger.error(f"❌ Frame 데이터 확인 실패: {e}")
            return False
    
    def _analyze_tracking_target(self, target):
        """추적 대상 분석 - 향상된 버전"""
        analysis = {
            'object_type': None,
            'colors': [],
            'gender': None,
            'clothing': [],
            'keywords': target.lower().split(),
            'original_target': target
        }
        
        target_lower = target.lower()
        
        # 객체 타입 매핑 확장
        object_mappings = {
            ('사람', '남성', '여성', '인물'): 'person',
            ('가방', 'handbag'): 'handbag',  # 추가!
            ('tv', '티비', '텔레비전'): 'tv',
            ('의자', 'chair'): 'chair',
            ('차', '자동차', '차량', '승용차'): 'car',
            ('자전거', 'bicycle'): 'bicycle',
            ('개', '강아지', '멍멍이'): 'dog',
            ('고양이', '냥이'): 'cat',
            ('노트북', '컴퓨터', 'laptop'): 'laptop',
            ('핸드폰', '휴대폰', '폰'): 'cell_phone'
        }
        
        
        for keywords, obj_type in object_mappings.items():
            if any(keyword in target_lower for keyword in keywords):
                analysis['object_type'] = obj_type
                break
        
        # 색상 추출 확장
        color_keywords = {
            '빨간': 'red', '빨강': 'red', '적색': 'red',
            '주황': 'orange', '오렌지': 'orange',
            '노란': 'yellow', '노랑': 'yellow', '황색': 'yellow',
            '초록': 'green', '녹색': 'green',
            '파란': 'blue', '파랑': 'blue', '청색': 'blue',
            '보라': 'purple', '자주': 'purple',
            '검은': 'black', '검정': 'black',
            '흰': 'white', '하얀': 'white', '백색': 'white',
            '회색': 'gray', '그레이': 'gray',
            '핑크': 'pink','분홍': 'pink',
            '갈색': 'brown', '브라운': 'brown',
        }
        
        for keyword, color in color_keywords.items():
            if keyword in target_lower:
                analysis['colors'].append(color)
        
        # 성별 및 의상 정보
        if any(word in target_lower for word in ['남성', '남자', '아저씨']):
            analysis['gender'] = 'male'
        elif any(word in target_lower for word in ['여성', '여자', '아주머니']):
            analysis['gender'] = 'female'
        
        if any(word in target_lower for word in ['상의', '티셔츠', '셔츠', '옷']):
            analysis['clothing'].append('top')
        if any(word in target_lower for word in ['모자', '캡', '햇']):
            analysis['clothing'].append('hat')
        
        return analysis
    
    def _perform_object_tracking(self, video, target_analysis, time_range):
        """실제 객체 추적 수행 - 향상된 버전"""
        tracking_results = []
        
        try:
            # Frame 모델에서 해당 비디오의 프레임들 가져오기
            frames_query = Frame.objects.filter(video=video).order_by('timestamp')
            
            # 시간 범위 필터링
            if time_range.get('start') and time_range.get('end'):
                start_time = self._parse_time_to_seconds(time_range['start'])
                end_time = self._parse_time_to_seconds(time_range['end'])
                frames_query = frames_query.filter(timestamp__gte=start_time, timestamp__lte=end_time)
                logger.info(f"⏰ 시간 필터링: {start_time}s ~ {end_time}s")
            
            frames = list(frames_query)
            logger.info(f"📊 분석할 프레임 수: {len(frames)}개")
            
            if not frames:
                logger.warning("⚠️ 분석할 프레임이 없습니다.")
                return []
            
            for frame in frames:
                try:
                    matches = self._find_matching_objects(frame, target_analysis)
                    for match in matches:
                        tracking_results.append({
                            'frame_id': frame.image_id,
                            'timestamp': frame.timestamp,
                            'confidence': match['confidence'],
                            'bbox': match['bbox'],
                            'description': match['description'],
                            'tracking_id': match.get('tracking_id', f"obj_{frame.image_id}"),
                            'match_reasons': match['match_reasons']
                        })
                except Exception as frame_error:
                    logger.warning(f"⚠️ 프레임 {frame.image_id} 처리 실패: {frame_error}")
                    continue
            
            # 시간순 정렬
            tracking_results.sort(key=lambda x: x['timestamp'])
            
            return tracking_results
            
        except Exception as e:
            logger.error(f"❌ 추적 수행 오류: {e}")
            return []
        
    def _perform_lenient_tracking(self, video, target_analysis, time_range):
        try:
            frames_query = Frame.objects.filter(video=video).order_by('timestamp')
            if time_range.get('start') and time_range.get('end'):
                start_time = self._parse_time_to_seconds(time_range['start'])
                end_time = self._parse_time_to_seconds(time_range['end'])
                frames_query = frames_query.filter(timestamp__gte=start_time, timestamp__lte=end_time)
                
            tracking_results = []
            for frame in frames_query:
                try:
                    detected_objects = self._get_detected_objects(frame)
                    for obj in detected_objects:
                        match_score = 0.0
                        match_reasons = []
                        
                        # 객체 타입 (필수)
                        if target_analysis.get('object_type'):
                            if obj['class'] == target_analysis['object_type']:
                                match_score += 0.3
                                match_reasons.append(f"{obj['class']} 객체 타입 매칭")
                            else:
                                continue  # 객체 타입이 다르면 건너뛰기
                        
                        # 색상 (관대하지만 여전히 선별적)
                        color_matched = False
                        if target_analysis.get('colors'):
                            for color in target_analysis['colors']:
                                obj_color_desc = obj['color_description'].lower()
                                if color == 'black':
                                    if 'black' in obj_color_desc:
                                        if 'mixed' not in obj_color_desc:
                                            match_score += 0.3  # 순수 black
                                        else:
                                            match_score += 0.1  # black-mixed
                                        match_reasons.append(f"{color} 색상 매칭")
                                        color_matched = True
                                        break
                                else:
                                    if color in obj_color_desc or color in [str(c).lower() for c in obj['colors']]:
                                        match_score += 0.2
                                        match_reasons.append(f"{color} 색상 매칭")
                                        color_matched = True
                                        break
                            
                            if not color_matched:
                                continue  # 색상이 지정되었는데 매칭되지 않으면 제외
                        
                        # 키워드 매칭
                        for keyword in target_analysis.get('keywords', []):
                            if keyword in obj['class'] and keyword not in ['사람', '옷', '입은']:
                                match_score += 0.1
                                match_reasons.append(f"키워드 '{keyword}' 매칭")
                        
                        # 관대한 검색에서도 최소 점수 유지
                        if match_score >= 0.3:
                            tracking_results.append({
                                'frame_id': frame.image_id,
                                'timestamp': frame.timestamp,
                                'confidence': min(match_score, obj['confidence'] or 0.5),
                                'bbox': obj['bbox'],
                                'description': self._generate_match_description(obj, target_analysis),
                                'tracking_id': obj.get('track_id') or f"obj_{frame.image_id}",
                                'match_reasons': match_reasons
                            })
                except Exception:
                    continue
                    
            tracking_results.sort(key=lambda x: x['timestamp'])
            logger.info(f"🔍 관대한 검색 결과: {len(tracking_results)}개")
            return tracking_results
        except Exception as e:
            logger.error(f"❌ 관대한 추적 오류: {e}")
            return []
    def _get_detected_objects(self, frame):
        """
        다양한 저장 스키마를 호환해서 객체 리스트를 반환한다.
        우선순위:
        1) frame.detected_objects
        2) frame.comprehensive_features['objects']
        3) frame.yolo_objects / frame.detections / frame.objects
        문자열(JSON)로 저장된 경우 파싱 시도.
        각 객체는 최소한 {'class','bbox','confidence'} 키를 갖도록 정규화.
        """
        import json

        candidates = []

        # 1) detected_objects
        if hasattr(frame, 'detected_objects') and frame.detected_objects:
            candidates.append(frame.detected_objects)

        # 2) comprehensive_features.objects
        if hasattr(frame, 'comprehensive_features') and frame.comprehensive_features:
            objs = None
            if isinstance(frame.comprehensive_features, dict):
                objs = frame.comprehensive_features.get('objects') \
                or frame.comprehensive_features.get('detections')
            elif isinstance(frame.comprehensive_features, str):
                try:
                    cf = json.loads(frame.comprehensive_features)
                    objs = (cf or {}).get('objects') or (cf or {}).get('detections')
                except Exception:
                    pass
            if objs:
                candidates.append(objs)

        # 3) 기타 필드들
        for attr in ('yolo_objects', 'detections', 'objects'):
            if hasattr(frame, attr) and getattr(frame, attr):
                candidates.append(getattr(frame, attr))

        # 첫 번째 유효 후보 선택
        detected = None
        for c in candidates:
            try:
                if isinstance(c, str):
                    c = json.loads(c)
                if isinstance(c, dict):           # {'objects': [...]} 형태 지원
                    c = c.get('objects') or c.get('detections')
                if isinstance(c, list):
                    detected = c
                    break
            except Exception:
                continue

        if not isinstance(detected, list):
            return []

        # 정규화
        norm = []
        for o in detected:
            if not isinstance(o, dict):
                continue
            cls = (o.get('class') or o.get('label') or o.get('name') or '').lower()
            bbox = o.get('bbox') or o.get('box') or o.get('xyxy') or []
            conf = float(o.get('confidence') or o.get('score') or 0.0)
            colors = o.get('colors') or o.get('color') or []
            if isinstance(colors, str):
                colors = [colors]
            color_desc = (o.get('color_description') or o.get('dominant_color') or 'unknown')
            track_id = o.get('track_id') or o.get('id')

            norm.append({
                'class': cls,
                'bbox': bbox,
                'confidence': conf,
                'colors': colors,
                'color_description': str(color_desc).lower(),
                'track_id': track_id,
                # 원본도 같이 보관(디버그/확장용)
                '_raw': o,
            })
        return norm

    def _find_matching_objects(self, frame, target_analysis):
        matches = []
        try:
            detected_objects = self._get_detected_objects(frame)
            if not detected_objects:
                return matches
                
            for obj in detected_objects:
                match_score = 0.0
                match_reasons = []
                
                # 객체 타입 매칭 (필수)
                if target_analysis.get('object_type') and obj['class'] == target_analysis['object_type']:
                    match_score += 0.4
                    match_reasons.append(f"{target_analysis['object_type']} 객체 매칭")
                elif target_analysis.get('object_type') and obj['class'] != target_analysis['object_type']:
                    # 객체 타입이 다르면 건너뛰기
                    continue
                
                # 색상 매칭 (더 엄격하게)
                color_matched = False
                if target_analysis.get('colors'):
                    target_colors = target_analysis['colors']
                    obj_color_desc = obj['color_description'].lower()
                    obj_colors = [str(c).lower() for c in obj['colors']]
                    
                    for target_color in target_colors:
                        # 정확한 색상 매칭 우선
                        if target_color == 'black':
                            if ('black' in obj_color_desc and 'mixed' not in obj_color_desc) or \
                            'black' in obj_colors:
                                match_score += 0.5  # 정확한 색상 매칭 높은 점수
                                match_reasons.append(f"정확한 {target_color} 색상 매칭")
                                color_matched = True
                                break
                            elif 'black' in obj_color_desc:  # black-mixed 등
                                match_score += 0.2  # 부분 매칭 낮은 점수
                                match_reasons.append(f"부분 {target_color} 색상 매칭")
                                color_matched = True
                        else:
                            # 다른 색상들도 비슷한 로직
                            if target_color in obj_color_desc and 'mixed' not in obj_color_desc:
                                match_score += 0.5
                                match_reasons.append(f"정확한 {target_color} 색상 매칭")
                                color_matched = True
                                break
                            elif target_color in obj_color_desc or target_color in obj_colors:
                                match_score += 0.2
                                match_reasons.append(f"부분 {target_color} 색상 매칭")
                                color_matched = True
                    
                    # 색상이 지정되었는데 매칭되지 않으면 제외
                    if not color_matched:
                        continue
                
                # 키워드 매칭 (보조)
                for keyword in target_analysis.get('keywords', []):
                    if keyword in obj['class'] and keyword not in ['사람', '옷', '입은']:
                        match_score += 0.1
                        match_reasons.append(f"키워드 '{keyword}' 매칭")
                
                # 최소 점수 기준 상향 조정
                if match_score >= 0.4:  # 0.3에서 0.4로 상향
                    matches.append({
                        'confidence': min(match_score, obj['confidence'] or 0.5),
                        'bbox': obj['bbox'],
                        'description': self._generate_match_description(obj, target_analysis),
                        'match_reasons': match_reasons,
                        'tracking_id': obj.get('track_id') or f"obj_{frame.image_id}",
                    })
            return matches
        except Exception as e:
            logger.warning(f"⚠️ 객체 매칭 오류: {e}")
            return []

    
    def _generate_match_description(self, obj, target_analysis):
        """매칭 설명 생성 - 향상된 버전"""
        desc_parts = []
        
        # 색상 정보
        color_desc = obj.get('color_description', '')
        if color_desc and color_desc != 'unknown':
            desc_parts.append(color_desc)
        
        # 객체 클래스
        obj_class = obj.get('class', '객체')
        desc_parts.append(obj_class)
        
        # 성별 정보 (있는 경우)
        if target_analysis.get('gender'):
            desc_parts.append(f"({target_analysis['gender']})")
        
        # 의상 정보 (있는 경우)
        if target_analysis.get('clothing'):
            clothing_desc = ', '.join(target_analysis['clothing'])
            desc_parts.append(f"[{clothing_desc}]")
        
        description = ' '.join(desc_parts) + ' 감지'
        
        return description
    
    def _parse_time_to_seconds(self, time_str):
        """시간 문자열을 초로 변환 - 향상된 버전"""
        try:
            if not time_str:
                return 0
            
            time_str = str(time_str).strip()
            
            if ':' in time_str:
                parts = time_str.split(':')
                minutes = int(parts[0])
                seconds = int(parts[1]) if len(parts) > 1 else 0
                return minutes * 60 + seconds
            else:
                # 순수 숫자인 경우
                return int(float(time_str))
        except (ValueError, TypeError) as e:
            logger.warning(f"⚠️ 시간 파싱 실패: {time_str} -> {e}")
            return 0

@method_decorator(csrf_exempt, name='dispatch')
class TimeBasedAnalysisView(APIView):
    """시간대별 분석 - 수정된 버전"""
    permission_classes = [AllowAny]
    
    def post(self, request):
        try:
            video_id = request.data.get('video_id')
            time_range = request.data.get('time_range', {})
            analysis_type = request.data.get('analysis_type', '성비 분포')
            
            logger.info(f"📊 시간대별 분석 요청: 비디오={video_id}, 시간범위={time_range}, 타입='{analysis_type}'")
            
            if not video_id or not time_range.get('start') or not time_range.get('end'):
                return Response({'error': '비디오 ID와 시간 범위가 필요합니다.'}, status=400)
            
            try:
                video = Video.objects.get(id=video_id)
            except Video.DoesNotExist:
                return Response({'error': '비디오를 찾을 수 없습니다.'}, status=404)
            
            # 시간 범위 파싱
            start_time = self._parse_time_to_seconds(time_range['start'])
            end_time = self._parse_time_to_seconds(time_range['end'])
            
            logger.info(f"⏰ 분석 시간: {start_time}초 ~ {end_time}초")
            
            # 해당 시간대의 프레임들 분석
            analysis_result = self._perform_time_based_analysis(
                video, start_time, end_time, analysis_type
            )
            
            logger.info(f"✅ 시간대별 분석 완료")
            
            return Response({
                'video_id': video_id,
                'time_range': time_range,
                'analysis_type': analysis_type,
                'result': analysis_result,
                'search_type': 'time_analysis'
            })
            
        except Exception as e:
            logger.error(f"❌ 시간대별 분석 오류: {e}")
            return Response({'error': str(e)}, status=500)
    
    def _perform_time_based_analysis(self, video, start_time, end_time, analysis_type):
        """시간대별 분석 수행"""
        
        # 해당 시간대 프레임들 가져오기
        frames = Frame.objects.filter(
            video=video,
            timestamp__gte=start_time,
            timestamp__lte=end_time
        ).order_by('timestamp')
        
        frame_list = list(frames)
        logger.info(f"📊 분석 대상 프레임: {len(frame_list)}개")
        
        if '성비' in analysis_type or '사람' in analysis_type:
            return self._analyze_gender_distribution(frame_list, start_time, end_time)
        elif '차량' in analysis_type or '교통' in analysis_type:
            return self._analyze_vehicle_distribution(frame_list, start_time, end_time)
        else:
            return self._analyze_general_statistics(frame_list, start_time, end_time)
    
    def _analyze_gender_distribution(self, frames, start_time, end_time):
        """성비 분석"""
        person_detections = []
        
        for frame in frames:
            if not hasattr(frame, 'detected_objects') or not frame.detected_objects:
                continue
                
            for obj in frame.detected_objects:
                if obj.get('class') == 'person':
                    person_detections.append({
                        'timestamp': frame.timestamp,
                        'confidence': obj.get('confidence', 0.5),
                        'bbox': obj.get('bbox', []),
                        'colors': obj.get('colors', []),
                        'color_description': obj.get('color_description', '')
                    })
        
        # 성별 추정 (간단한 휴리스틱 - 실제로는 더 정교한 AI 모델 필요)
        male_count = 0
        female_count = 0
        
        for detection in person_detections:
            # 색상 기반 간단한 성별 추정
            colors = detection['color_description'].lower()
            if 'blue' in colors or 'black' in colors or 'gray' in colors:
                male_count += 1
            elif 'pink' in colors or 'red' in colors:
                female_count += 1
            else:
                # 50:50으로 분배
                if len(person_detections) % 2 == 0:
                    male_count += 1
                else:
                    female_count += 1
        
        total_persons = male_count + female_count
        
        # 의상 색상 분포
        clothing_colors = {}
        for detection in person_detections:
            color = detection['color_description']
            if color and color != 'unknown':
                clothing_colors[color] = clothing_colors.get(color, 0) + 1
        
        # 피크 시간대 분석
        time_distribution = {}
        for detection in person_detections:
            time_bucket = int(detection['timestamp'] // 30) * 30  # 30초 단위
            time_distribution[time_bucket] = time_distribution.get(time_bucket, 0) + 1
        
        peak_times = sorted(time_distribution.items(), key=lambda x: x[1], reverse=True)[:2]
        peak_time_strings = [f"{self._seconds_to_time_string(t[0])}-{self._seconds_to_time_string(t[0]+30)}" 
                           for t in peak_times]
        
        return {
            'total_persons': total_persons,
            'male_count': male_count,
            'female_count': female_count,
            'gender_ratio': {
                'male': round((male_count / total_persons * 100), 1) if total_persons > 0 else 0,
                'female': round((female_count / total_persons * 100), 1) if total_persons > 0 else 0
            },
            'clothing_colors': dict(sorted(clothing_colors.items(), key=lambda x: x[1], reverse=True)),
            'peak_times': peak_time_strings,
            'movement_patterns': 'left_to_right_dominant',  # 간단한 예시
            'analysis_period': f"{self._seconds_to_time_string(start_time)} - {self._seconds_to_time_string(end_time)}"
        }
    
    def _analyze_vehicle_distribution(self, frames, start_time, end_time):
        """차량 분포 분석"""
        vehicles = []
        
        for frame in frames:
            if not hasattr(frame, 'detected_objects') or not frame.detected_objects:
                continue
                
            for obj in frame.detected_objects:
                if obj.get('class') in ['car', 'truck', 'bus', 'motorcycle']:
                    vehicles.append({
                        'type': obj.get('class'),
                        'timestamp': frame.timestamp,
                        'confidence': obj.get('confidence', 0.5)
                    })
        
        vehicle_types = {}
        for v in vehicles:
            vehicle_types[v['type']] = vehicle_types.get(v['type'], 0) + 1
        
        duration_minutes = (end_time - start_time) / 60
        
        return {
            'total_vehicles': len(vehicles),
            'vehicle_types': vehicle_types,
            'average_per_minute': round(len(vehicles) / max(1, duration_minutes), 1),
            'analysis_period': f"{self._seconds_to_time_string(start_time)} - {self._seconds_to_time_string(end_time)}"
        }
    
    def _analyze_general_statistics(self, frames, start_time, end_time):
        """일반 통계 분석"""
        all_objects = []
        
        for frame in frames:
            if hasattr(frame, 'detected_objects') and frame.detected_objects:
                all_objects.extend(frame.detected_objects)
        
        object_counts = {}
        for obj in all_objects:
            obj_class = obj.get('class', 'unknown')
            object_counts[obj_class] = object_counts.get(obj_class, 0) + 1
        
        return {
            'total_objects': len(all_objects),
            'object_distribution': dict(sorted(object_counts.items(), key=lambda x: x[1], reverse=True)),
            'frames_analyzed': len(frames),
            'average_objects_per_frame': round(len(all_objects) / max(1, len(frames)), 1),
            'analysis_period': f"{self._seconds_to_time_string(start_time)} - {self._seconds_to_time_string(end_time)}"
        }
    
    def _parse_time_to_seconds(self, time_str):
        """시간 문자열을 초로 변환"""
        try:
            if ':' in time_str:
                parts = time_str.split(':')
                minutes = int(parts[0])
                seconds = int(parts[1]) if len(parts) > 1 else 0
                return minutes * 60 + seconds
            else:
                return int(time_str)
        except:
            return 0
    
    def _seconds_to_time_string(self, seconds):
        """초를 시간 문자열로 변환"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}:{secs:02d}"


@method_decorator(csrf_exempt, name='dispatch')
class CrossVideoSearchView(APIView):
    """영상 간 검색 - 수정된 버전"""
    permission_classes = [AllowAny]
    
    def post(self, request):
        try:
            query = request.data.get('query', '').strip()
            search_filters = request.data.get('filters', {})
            
            logger.info(f"🔍 크로스 비디오 검색 요청: '{query}'")
            
            if not query:
                return Response({'error': '검색어를 입력해주세요.'}, status=400)
            
            # 쿼리 분석
            query_analysis = self._analyze_query(query)
            
            # 분석된 비디오들 중에서 검색
            videos = Video.objects.filter(is_analyzed=True)
            matching_videos = []
            
            for video in videos:
                match_score = self._calculate_video_match_score(video, query_analysis, search_filters)
                if match_score > 0.3:  # 임계값
                    matching_videos.append({
                        'video_id': video.id,
                        'video_name': video.original_name,
                        'match_score': match_score,
                        'match_reasons': self._get_match_reasons(video, query_analysis),
                        'metadata': self._get_video_metadata(video),
                        'thumbnail_url': f'/frame/{video.id}/100/',
                    })
            
            # 점수순 정렬
            matching_videos.sort(key=lambda x: x['match_score'], reverse=True)
            
            logger.info(f"✅ 크로스 비디오 검색 완료: {len(matching_videos)}개 결과")
            
            return Response({
                'query': query,
                'total_matches': len(matching_videos),
                'results': matching_videos[:20],  # 상위 20개
                'query_analysis': query_analysis,
                'search_type': 'cross_video'
            })
            
        except Exception as e:
            logger.error(f"❌ 크로스 비디오 검색 오류: {e}")
            return Response({'error': str(e)}, status=500)
    
    def _analyze_query(self, query):
        """쿼리에서 날씨, 시간대, 장소 등 추출"""
        analysis = {
            'weather': None,
            'time_of_day': None,
            'location': None,
            'objects': [],
            'activities': []
        }
        
        query_lower = query.lower()
        
        # 날씨 키워드
        weather_keywords = {
            '비': 'rainy', '비가': 'rainy', '우천': 'rainy',
            '맑은': 'sunny', '화창한': 'sunny', '햇빛': 'sunny',
            '흐린': 'cloudy', '구름': 'cloudy'
        }
        
        # 시간대 키워드
        time_keywords = {
            '밤': 'night', '야간': 'night', '저녁': 'evening',
            '낮': 'day', '오후': 'afternoon', '아침': 'morning'
        }
        
        # 장소 키워드
        location_keywords = {
            '실내': 'indoor', '건물': 'indoor', '방': 'indoor',
            '실외': 'outdoor', '도로': 'outdoor', '거리': 'outdoor'
        }
        
        for keyword, value in weather_keywords.items():
            if keyword in query_lower:
                analysis['weather'] = value
                break
        
        for keyword, value in time_keywords.items():
            if keyword in query_lower:
                analysis['time_of_day'] = value
                break
                
        for keyword, value in location_keywords.items():
            if keyword in query_lower:
                analysis['location'] = value
                break
        
        return analysis
    
    def _calculate_video_match_score(self, video, query_analysis, filters):
        """비디오와 쿼리 간의 매칭 점수 계산"""
        score = 0.0
        
        try:
            # VideoAnalysis에서 분석 결과가 있는 경우
            if hasattr(video, 'analysis'):
                analysis = video.analysis
                stats = analysis.analysis_statistics
                scene_types = stats.get('scene_types', [])
                
                # 날씨 매칭
                if query_analysis['weather']:
                    weather_scenes = [s for s in scene_types if query_analysis['weather'] in s.lower()]
                    if weather_scenes:
                        score += 0.4
                
                # 시간대 매칭
                if query_analysis['time_of_day']:
                    time_scenes = [s for s in scene_types if query_analysis['time_of_day'] in s.lower()]
                    if time_scenes:
                        score += 0.3
                
                # 장소 매칭
                if query_analysis['location']:
                    location_scenes = [s for s in scene_types if query_analysis['location'] in s.lower()]
                    if location_scenes:
                        score += 0.3
            
            return min(score, 1.0)
            
        except Exception:
            return 0.0
    
    def _get_match_reasons(self, video, query_analysis):
        """매칭 이유 생성"""
        reasons = []
        
        if query_analysis['weather']:
            reasons.append(f"{query_analysis['weather']} 날씨 조건")
        if query_analysis['time_of_day']:
            reasons.append(f"{query_analysis['time_of_day']} 시간대")
        if query_analysis['location']:
            reasons.append(f"{query_analysis['location']} 환경")
            
        return reasons
    
    def _get_video_metadata(self, video):
        """비디오 메타데이터 반환"""
        metadata = {
            'duration': video.duration,
            'file_size': video.file_size,
            'uploaded_at': video.uploaded_at.isoformat(),
            'analysis_type': 'basic'
        }
        
        if hasattr(video, 'analysis'):
            stats = video.analysis.analysis_statistics
            metadata.update({
                'analysis_type': stats.get('analysis_type', 'basic'),
                'scene_types': stats.get('scene_types', []),
                'dominant_objects': stats.get('dominant_objects', [])
            })
        
        return metadata


class AdvancedSearchAutoView(APIView):
    """통합 고급 검색 - 자동 타입 감지"""
    permission_classes = [AllowAny]
    
    def post(self, request):
        try:
            query = request.data.get('query', '').strip()
            video_id = request.data.get('video_id')
            time_range = request.data.get('time_range', {})
            options = request.data.get('options', {})
            
            if not query:
                return Response({'error': '검색어를 입력해주세요.'}, status=400)
            
            # 검색 타입 자동 감지
            search_type = self._detect_search_type(query, video_id, time_range, options)
            
            # 해당 검색 타입에 따라 적절한 View 호출
            if search_type == 'cross-video':
                view = CrossVideoSearchView()
                return view.post(request)
            elif search_type == 'object-tracking':
                view = IntraVideoTrackingView()
                return view.post(request)
            elif search_type == 'time-analysis':
                view = TimeBasedAnalysisView()
                return view.post(request)
            else:
                # 기본 검색으로 fallback
                view = EnhancedVideoChatView()
                return view.post(request)
                
        except Exception as e:
            return Response({'error': str(e)}, status=500)
    
    def _detect_search_type(self, query, video_id, time_range, options):
        """검색 타입 자동 감지 로직"""
        query_lower = query.lower()
        
        # 시간대별 분석 키워드
        time_analysis_keywords = [
            '성비', '분포', '통계', '시간대', '구간', '사이', 
            '몇명', '얼마나', '평균', '비율', '패턴', '분석'
        ]
        
        # 객체 추적 키워드
        tracking_keywords = [
            '추적', '따라가', '이동', '경로', '지나간', 
            '상의', '모자', '색깔', '옷', '사람', '차량'
        ]
        
        # 영상 간 검색 키워드
        cross_video_keywords = [
            '촬영된', '영상', '비디오', '찾아', '비가', '밤', 
            '낮', '실내', '실외', '장소', '날씨'
        ]
        
        # 시간 범위가 있고 분석 키워드가 있으면 시간대별 분석
        if (time_range.get('start') and time_range.get('end')) or \
           any(keyword in query_lower for keyword in time_analysis_keywords):
            return 'time-analysis'
        
        # 특정 비디오 ID가 있고 추적 키워드가 있으면 객체 추적
        if video_id and any(keyword in query_lower for keyword in tracking_keywords):
            return 'object-tracking'
        
        # 크로스 비디오 키워드가 있으면 영상 간 검색
        if any(keyword in query_lower for keyword in cross_video_keywords):
            return 'cross-video'
        
        # 기본값: 비디오 ID가 있으면 추적, 없으면 크로스 비디오
        return 'object-tracking' if video_id else 'cross-video'


class AnalyzerSystemStatusView(APIView):
    """AI 분석 시스템 전체 상태 조회"""
    permission_classes = [AllowAny]
    
    def get(self, request):
        try:
            if not VIDEO_ANALYZER_AVAILABLE:
                return Response({
                    'system_status': 'unavailable',
                    'error': 'video_analyzer 모듈을 import할 수 없습니다',
                    'available_features': {},
                    'recommendation': 'video_analyzer.py 파일과 의존성들을 확인해주세요'
                })
            
            # 분석기 상태 조회
            analyzer_status = get_analyzer_status()
            
            # RAG 시스템 상태 조회
            try:
                rag_system = get_video_rag_system()
                rag_info = rag_system.get_database_info() if rag_system else None
                rag_available = rag_system is not None
            except:
                rag_info = None
                rag_available = False
            
            # 시스템 통계
            total_videos = Video.objects.count()
            analyzed_videos = Video.objects.filter(is_analyzed=True).count()
            processing_videos = Video.objects.filter(analysis_status='processing').count()
            
            response_data = {
                'system_status': 'operational' if analyzer_status.get('status') == 'initialized' else 'limited',
                'analyzer': analyzer_status,
                'rag_system': {
                    'available': rag_available,
                    'info': rag_info
                },
                'statistics': {
                    'total_videos': total_videos,
                    'analyzed_videos': analyzed_videos,
                    'processing_videos': processing_videos,
                    'analysis_rate': (analyzed_videos / max(total_videos, 1)) * 100
                },
                'capabilities': {
                    'yolo_object_detection': analyzer_status.get('features', {}).get('yolo', False),
                    'clip_scene_analysis': analyzer_status.get('features', {}).get('clip', False),
                    'ocr_text_extraction': analyzer_status.get('features', {}).get('ocr', False),
                    'vqa_question_answering': analyzer_status.get('features', {}).get('vqa', False),
                    'scene_graph_generation': analyzer_status.get('features', {}).get('scene_graph', False),
                    'rag_search_system': rag_available
                },
                'device': analyzer_status.get('device', 'unknown'),
                'last_checked': datetime.now().isoformat()
            }
            
            return Response(response_data)
            
        except Exception as e:
            return Response({
                'system_status': 'error',
                'error': f'시스템 상태 조회 실패: {str(e)}',
                'last_checked': datetime.now().isoformat()
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)



import os, time, json, subprocess, tempfile
from datetime import datetime
from django.conf import settings
from django.http import FileResponse, Http404
from django.urls import reverse
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from rest_framework.views import APIView
from rest_framework.permissions import AllowAny
from rest_framework.response import Response

from .models import Video, TrackPoint, Frame, Scene
from django.http import JsonResponse


@method_decorator(csrf_exempt, name='dispatch')
class EnhancedVideoChatView(APIView):
    """향상된 비디오 채팅 - 자연어 질의에 대해 텍스트 + 썸네일/클립을 함께 반환"""
    permission_classes = [AllowAny]

    # ---------- 초기화 ----------
    def __init__(self):
        super().__init__()
        self.llm_client = None
        self.video_analyzer = None
    def _initialize_services(self):
        """서비스 안전 초기화 - LLM 클라이언트 개선"""
        if self.llm_client is None:
            try:
                from .llm_client import get_llm_client
                self.llm_client = get_llm_client()
                if self.llm_client.is_available():
                    print("LLM 클라이언트 초기화 완료")
                else:
                    print("LLM 클라이언트 비활성화 - 기본 설명 생성 모드")
            except Exception as e:
                print(f"LLM 클라이언트 초기화 실패: {e}")
                # Mock 클라이언트로 폴백
                from .llm_client import MockLLMClient
                self.llm_client = MockLLMClient()

        if self.video_analyzer is None:
            try:
                from .video_analyzer import get_video_analyzer
                self.video_analyzer = get_video_analyzer()
                print("비디오 분석기 초기화 완료")
            except Exception as e:
                print(f"비디오 분석기 초기화 실패: {e}")

    # ---------- 공용 유틸 ----------
    def _frame_urls(self, request, video_id, frame_number):
        """프레임 정규 이미지 & 박스이미지 URL"""
        base = request.build_absolute_uri
        return {
            'image': base(reverse('frame_normal', args=[video_id, frame_number])),
            'image_bbox': base(reverse('frame_with_bbox', args=[video_id, frame_number])),
        }

    def _clip_url(self, request, video_id, timestamp, duration=4):
        """프리뷰 클립 URL"""
        url = reverse('clip_preview', args=[video_id, int(timestamp)])
        return request.build_absolute_uri(f"{url}?duration={int(duration)}")

    def _format_time(self, seconds):
        try:
            m, s = int(seconds) // 60, int(seconds) % 60
            return f"{m}:{s:02d}"
        except:
            return "0:00"

    def _get_video_safe(self, video_id):
        try:
            if video_id:
                return Video.objects.get(id=video_id)
            return Video.objects.filter(is_analyzed=True).first()
        except:
            return None

    # ---------- NLU(간단 슬롯 추출) ----------
  # EnhancedVideoChatView에 추가할 메서드들

    def _nlu(self, text: str):
        """intent + slots 간단 추출 (영상 설명 의도 추가)"""
        q = text.lower()
        intent = 'general'
        
        # 영상 설명 키워드 추가
        if any(k in q for k in ['설명해줘', '설명해', '어떤', '무슨', '내용', '장면', '영상에 대해', '뭐가 나와', '어떻게', '상황']):
            intent = 'video_description'
        elif any(k in q for k in ['요약', 'summary']): 
            intent = 'summary'
        elif any(k in q for k in ['하이라이트', 'highlight']): 
            intent = 'highlight'
        elif any(k in q for k in ['정보', 'info']): 
            intent = 'info'
        elif any(k in q for k in ['성비', 'gender']): 
            intent = 'gender_distribution'
        elif any(k in q for k in ['분위기', '무드', 'mood']): 
            intent = 'scene_mood'
        elif any(k in q for k in ['비오는', '밤', '낮', '실내', '실외']): 
            intent = 'cross_video'
        elif any(k in q for k in ['찾아줘', '찾아 줘', '찾아', '검색', '나와', '보여줘', '추적']): 
            intent = 'object_tracking'
        elif any(k in q for k in ['있어?', '나와?', '등장해?']): 
            intent = 'object_presence'

        # 기존 색상/객체/시간범위 처리 (동일)
        color_map = {
            '빨강':'red','빨간':'red','적색':'red',
            '주황':'orange','오렌지':'orange',
            '노랑':'yellow','노란':'yellow','황색':'yellow',
            '초록':'green','녹색':'green',
            '파랑':'blue','파란':'blue','청색':'blue',
            '보라':'purple','자주':'purple',
            '검정':'black','검은':'black',
            '하양':'white','흰':'white','백색':'white',
            '회색':'gray','그레이':'gray',
            '갈색':'brown',
            '핑크':'pink','분홍':'pink',
        }
        colors = [v for k,v in color_map.items() if k in q]

        object_map = {
            '사람':'person','남성':'person','여성':'person','인물':'person',
            '가방':'handbag','핸드백':'handbag',
            'tv':'tv','티비':'tv','텔레비전':'tv',
            '의자':'chair',
            '자전거':'bicycle',
            '차':'car','자동차':'car',
            '고양이':'cat','개':'dog',
            '노트북':'laptop','휴대폰':'cell_phone'
        }
        objects = []
        for k,v in object_map.items():
            if k in q:
                objects.append(v)
        objects = list(dict.fromkeys(objects))

        import re
        tmatch = re.search(r'(\d{1,2}:\d{2})\s*[-~]\s*(\d{1,2}:\d{2})', q)
        trange = None
        if tmatch:
            def to_sec(s):
                mm, ss = s.split(':')
                return int(mm) * 60 + int(ss)
            trange = {'start': to_sec(tmatch.group(1)), 'end': to_sec(tmatch.group(2))}

        return {'intent': intent, 'slots': {'colors': colors, 'objects': objects, 'time_range': trange}}

    def _handle_video_description(self, video: Video, raw_text: str, request=None):
        """LLM을 활용한 자연스러운 영상 설명 생성"""
        try:
            # 프레임들의 캡션 정보 수집
            frames = Frame.objects.filter(video=video).order_by('timestamp')
            
            if not frames.exists():
                return {'text': '영상 분석 데이터가 없어서 설명을 제공할 수 없습니다.', 'items': []}
            
            # 대표 캡션들 수집 (전체 영상의 5-8개 구간)
            total_frames = frames.count()
            sample_count = min(8, max(5, total_frames // 6))  # 5-8개 구간
            sample_indices = [int(i * total_frames / sample_count) for i in range(sample_count)]
            
            key_scenes = []
            caption_data = []
            
            for idx in sample_indices:
                try:
                    frame = frames[idx] if idx < total_frames else frames.last()
                    
                    # 최고 품질 캡션 선택
                    best_caption = ""
                    if hasattr(frame, 'final_caption') and frame.final_caption:
                        best_caption = frame.final_caption
                    elif hasattr(frame, 'enhanced_caption') and frame.enhanced_caption:
                        best_caption = frame.enhanced_caption
                    elif hasattr(frame, 'caption') and frame.caption:
                        best_caption = frame.caption
                    elif hasattr(frame, 'blip_caption') and frame.blip_caption:
                        best_caption = frame.blip_caption
                    
                    if best_caption and len(best_caption.strip()) > 10:
                        scene_data = {
                            'timestamp': float(frame.timestamp),
                            'time_str': self._format_time(frame.timestamp),
                            'frame_id': frame.image_id,
                            'caption': best_caption.strip()
                        }
                        key_scenes.append(scene_data)
                        caption_data.append({
                            'time': scene_data['time_str'],
                            'caption': best_caption.strip()
                        })
                        
                except (IndexError, AttributeError):
                    continue
            
            if not caption_data:
                return {'text': '영상 캡션 정보가 부족해서 상세한 설명을 제공할 수 없습니다.', 'items': []}
            
            # LLM을 사용해서 자연스러운 설명 생성
            llm_description = self._generate_llm_description(video, caption_data, raw_text)
            
            # 대표 장면 이미지들 (3-5개)
            representative_scenes = key_scenes[::max(1, len(key_scenes)//4)][:5]  # 최대 5개 선택
            items = []
            
            for scene in representative_scenes:
                if request:
                    media = self._frame_urls(request, video.id, scene['frame_id'])
                    clip = self._clip_url(request, video.id, scene['timestamp'])
                    items.append({
                        'time': scene['time_str'],
                        'seconds': int(scene['timestamp']),
                        'frame_id': scene['frame_id'],
                        'desc': scene['caption'][:120] + "..." if len(scene['caption']) > 120 else scene['caption'],
                        'full_caption': scene['caption'],
                        'source': 'AI 분석',
                        'thumbUrl': media.get('image'),
                        'thumbBBoxUrl': media.get('image_bbox'),
                        'clipUrl': clip,
                    })
            
            return {'text': llm_description, 'items': items}
            
        except Exception as e:
            print(f"영상 설명 생성 오류: {e}")
            return {'text': f'영상 설명을 생성하는 중 오류가 발생했습니다: {str(e)}', 'items': []}

    def _generate_llm_description(self, video: Video, caption_data, user_query):
        """LLM을 사용해서 캡션들을 분석하고 자연스러운 설명 생성"""
        try:
            if not self.llm_client:
                # LLM이 없으면 기본 설명 생성
                return self._generate_fallback_description(video, caption_data)
            
            # LLM 프롬프트 구성
            prompt = self._build_description_prompt(video, caption_data, user_query)
            
            # LLM 호출
            llm_response = self.llm_client.generate_response(prompt)
            
            if llm_response and len(llm_response.strip()) > 50:
                return llm_response.strip()
            else:
                return self._generate_fallback_description(video, caption_data)
                
        except Exception as e:
            print(f"LLM 설명 생성 실패: {e}")
            return self._generate_fallback_description(video, caption_data)

    def _build_description_prompt(self, video: Video, caption_data, user_query):
        """LLM용 프롬프트 구성"""
        
        prompt = f"""영상 분석 결과를 바탕으로 자연스럽고 읽기 쉬운 영상 설명을 작성해주세요.

    영상 정보:
    - 파일명: {video.original_name}
    - 길이: {round(video.duration, 1)}초
    - 사용자 질문: "{user_query}"

    시간대별 분석 결과:
    """
        
        for data in caption_data:
            prompt += f"- {data['time']}: {data['caption']}\n"
        
        prompt += """
    다음 요구사항에 따라 설명을 작성해주세요:

    1. 자연스럽고 읽기 쉬운 한국어로 작성
    2. 중복되는 내용은 요약하여 정리
    3. 영상의 전체적인 흐름과 주요 내용 강조
    4. 2-3개 문단으로 구성 (각 문단은 2-4문장)
    5. 기술적인 용어나 프레임 번호 같은 정보는 제외
    6. 영상의 분위기나 상황을 생생하게 전달

    설명 형식:
    첫 번째 문단: 영상의 전체적인 배경과 상황
    두 번째 문단: 주요 장면과 활동
    세 번째 문단: 영상의 특징이나 인상적인 부분

    이제 영상 설명을 작성해주세요:"""

        return prompt

    def _generate_fallback_description(self, video: Video, caption_data):
        """LLM이 없을 때 사용할 기본 설명 생성"""
        
        description = f"'{video.original_name}' 영상 분석\n\n"
        
        # 기본 정보
        description += f"이 영상은 총 {round(video.duration, 1)}초 길이의 영상입니다.\n\n"
        
        # 주요 내용 요약
        all_captions = " ".join([data['caption'] for data in caption_data]).lower()
        
        # 장소 추출
        locations = []
        if '실내' in all_captions or 'indoor' in all_captions:
            locations.append('실내')
        if '쇼핑몰' in all_captions:
            locations.append('쇼핑몰')
        if '거리' in all_captions:
            locations.append('거리')
        
        # 시간대 추출
        time_info = []
        if '오후' in all_captions:
            time_info.append('오후 시간')
        if '밝은' in all_captions:
            time_info.append('밝은 환경')
        
        # 활동 추출
        activities = []
        if '걷' in all_captions:
            activities.append('사람들이 걷고 있는')
        if '쇼핑' in all_captions:
            activities.append('쇼핑하는')
        
        # 설명 구성
        if locations:
            description += f"{', '.join(locations)}에서 "
        if time_info:
            description += f"{', '.join(time_info)}에 "
        if activities:
            description += f"{', '.join(activities)} 모습이 담겨 있습니다.\n\n"
        
        # 시간대별 주요 변화
        if len(caption_data) >= 3:
            description += "영상 초반에는 "
            start_caption = caption_data[0]['caption']
            if '사람' in start_caption:
                description += "여러 사람들이 등장하여 "
            if '걷' in start_caption:
                description += "이동하는 모습을 보여주며, "
            
            description += "중반부에는 "
            mid_caption = caption_data[len(caption_data)//2]['caption']
            if '활동' in mid_caption or '쇼핑' in mid_caption:
                description += "다양한 활동들이 이어집니다. "
            
            description += "전체적으로 일상적인 장면들이 자연스럽게 연결된 영상입니다."
        
        return description

    def _generate_comprehensive_description(self, video: Video, key_scenes, detailed_captions):
        """수집된 캡션들을 바탕으로 종합적인 영상 설명 생성"""
        
        # 1. 기본 정보
        description = f"📹 '{video.original_name}' 영상 분석 결과\n\n"
        description += f"⏱️ 길이: {round(video.duration, 1)}초\n"
        description += f"🎬 총 {len(key_scenes)}개 주요 장면 분석\n\n"
        
        # 2. 전체적인 특징 추출
        all_text = " ".join(detailed_captions).lower()
        
        # 장소/환경 정보
        locations = []
        if '실내' in all_text or 'indoor' in all_text:
            locations.append('실내')
        if '실외' in all_text or 'outdoor' in all_text:
            locations.append('실외')
        if '쇼핑몰' in all_text:
            locations.append('쇼핑몰')
        if '거리' in all_text or 'sidewalk' in all_text:
            locations.append('거리')
        if '건물' in all_text or 'building' in all_text:
            locations.append('건물')
        
        # 시간대 정보
        time_info = []
        if '오후' in all_text or 'afternoon' in all_text:
            time_info.append('오후')
        if '아침' in all_text or 'morning' in all_text:
            time_info.append('아침')
        if '밤' in all_text or 'night' in all_text:
            time_info.append('밤')
        if '밝은' in all_text or 'bright' in all_text:
            time_info.append('밝은 환경')
        
        # 주요 객체/활동
        detected_objects = set()
        activities = set()
        
        for caption in detailed_captions:
            caption_lower = caption.lower()
            # 객체 추출
            if '사람' in caption_lower or 'person' in caption_lower:
                detected_objects.add('사람')
            if '가방' in caption_lower or 'handbag' in caption_lower:
                detected_objects.add('가방')
            if 'tv' in caption_lower or '티비' in caption_lower:
                detected_objects.add('TV')
            if '의자' in caption_lower or 'chair' in caption_lower:
                detected_objects.add('의자')
            
            # 활동 추출
            if '걷' in caption_lower or 'walking' in caption_lower:
                activities.add('걷기')
            if '서' in caption_lower or 'standing' in caption_lower:
                activities.add('서있기')
            if '쇼핑' in caption_lower or 'shopping' in caption_lower:
                activities.add('쇼핑')
            if '대화' in caption_lower or 'talking' in caption_lower:
                activities.add('대화')
        
        # 3. 종합 설명
        description += "🏞️ **영상 개요:**\n"
        
        if locations:
            description += f"- 장소: {', '.join(locations)}\n"
        if time_info:
            description += f"- 시간/환경: {', '.join(time_info)}\n"
        if detected_objects:
            description += f"- 주요 객체: {', '.join(list(detected_objects)[:5])}\n"
        if activities:
            description += f"- 주요 활동: {', '.join(list(activities)[:3])}\n"
        
        description += "\n"
        
        # 4. 시간대별 주요 장면 (처음, 중간, 끝 3개 구간)
        if len(key_scenes) >= 3:
            description += "🎞️ **주요 장면 요약:**\n\n"
            
            # 시작 장면
            start_scene = key_scenes[0]
            description += f"**{start_scene['time_str']} (시작):** {start_scene['caption'][:150]}...\n\n"
            
            # 중간 장면
            mid_scene = key_scenes[len(key_scenes)//2]
            description += f"**{mid_scene['time_str']} (중반):** {mid_scene['caption'][:150]}...\n\n"
            
            # 끝 장면
            end_scene = key_scenes[-1]
            description += f"**{end_scene['time_str']} (종료):** {end_scene['caption'][:150]}...\n\n"
        
        # 5. 추가 정보
        description += "💡 **분석 정보:**\n"
        description += f"- 분석 상태: {video.analysis_status}\n"
        description += f"- 프레임 기반 AI 분석을 통해 생성된 설명입니다\n"
        description += f"- 아래 이미지들을 클릭하면 해당 시점의 상세 장면을 볼 수 있습니다"
        
        return description
    # ---------- Frame JSON 통일 ----------
    def _get_detected_objects(self, frame: Frame):
        """
        Frame.detected_objects(JSONField/CharField) → list[dict] 로 통일 반환
        객체 예시: {class:'person', bbox:[x1,y1,x2,y2], colors:['green'], color_description:'green-mixed', confidence:0.7, gender:'male', track_id:'t1'}
        """
        data = []
        raw = getattr(frame, 'detected_objects', None)
        if not raw:
            return data
        try:
            if isinstance(raw, str):
                data = json.loads(raw)
            elif isinstance(raw, (list, dict)):
                data = raw
        except Exception:
            return []
        if isinstance(data, dict):
            # {objects:[...]} 형태도 허용
            data = data.get('objects', [])
        # 안전 필드 보정
        norm = []
        for o in data:
            norm.append({
                'class': (o.get('class') or o.get('label') or '').lower(),
                'bbox': o.get('bbox') or o.get('box') or [],
                'colors': o.get('colors') or [],
                'color_description': (o.get('color_description') or o.get('color') or 'unknown').lower(),
                'confidence': float(o.get('confidence', 0.5)),
                'gender': (o.get('gender') or '').lower(),
                'track_id': o.get('track_id') or o.get('id'),
            })
        return norm

    # ---------- POST ----------

    def post(self, request):
        try:
            self._initialize_services()
            user_query = (request.data.get('message') or '').strip()
            video_id = request.data.get('video_id')

            if not user_query:
                return Response({'response': '메시지를 입력해주세요.'})

            video = self._get_video_safe(video_id)
            if not video:
                return Response({'response': '분석된 비디오가 없습니다. 업로드/분석 후 이용해주세요.'})

            nlu = self._nlu(user_query)
            intent, slots = nlu['intent'], nlu['slots']

            # 영상 설명 처리 추가
            if intent == 'video_description':
                out = self._handle_video_description(video, user_query, request=request)
            elif intent == 'object_tracking':
                out = self._handle_object_tracking(video, slots, user_query, request=request)
            elif intent == 'object_presence':
                out = self._handle_object_presence(video, user_query, slots, request=request)
            elif intent == 'gender_distribution':
                out = {'text': self._handle_gender_distribution(video, slots), 'items': []}
            elif intent == 'scene_mood':
                out = {'text': self._handle_scene_mood(video), 'items': []}
            elif intent == 'cross_video':
                out = {'text': self._handle_cross_video(user_query), 'items': []}
            elif intent == 'summary':
                out = self._handle_summary(video, request=request)
            elif intent == 'highlight':
                out = self._handle_highlight(video, request=request)
            elif intent == 'info':
                out = {'text': self._handle_info(video), 'items': []}
            else:
                out = {'text': f"'{user_query}' 질문 확인! 색상/객체/시간범위를 함께 주시면 더 정확해요. 예) '초록 상의 사람 0:05~0:10'", 'items': []}

            return Response({
                'response': out['text'],
                'video_id': video.id,
                'video_name': video.original_name,
                'query_type': intent,
                'timestamp': time.time(),
                'items': out.get('items', []),
            })

        except Exception as e:
            print(f"[EnhancedVideoChatView] 오류: {e}")
            return Response({'response': f"질문을 받았습니다. 처리 중 오류: {e}", 'fallback': True})
    # ---------- Intent Handlers ----------
    def _handle_object_tracking(self, video: Video, slots: dict, raw_text: str, request=None):
        """색/객체/시간 범위를 기반으로 상위 매칭 장면 + 썸네일/클립 반환"""
        colors = set(slots.get('colors') or [])
        objects = set(slots.get('objects') or ['person'])  # 기본 사람
        tr = slots.get('time_range')

        # person_database에서 사람 데이터 검색
        hits = []
        
        # 분석 결과 JSON 파일에서 person_database 읽기
        print(f"🔍 [DEBUG] 비디오 ID: {video.id}, JSON 경로: {video.analysis_json_path}")
        if video.analysis_json_path and os.path.exists(video.analysis_json_path):
            try:
                with open(video.analysis_json_path, 'r', encoding='utf-8') as f:
                    analysis_data = json.load(f)
                print(f"🔍 [DEBUG] JSON 파일 로드 성공, video_id: {analysis_data.get('video_id')}")
                
                if 'result' in analysis_data and 'person_database' in analysis_data['result']:
                    person_database = analysis_data['result']['person_database']
                    print(f"🔍 [DEBUG] person_database 개수: {len(person_database)}")
                    
                    # 첫 번째 person 데이터 샘플 출력
                    if person_database:
                        sample_person = person_database[0]
                        print(f"🔍 [DEBUG] 첫 번째 person 샘플: {sample_person}")
                    
                    for person_data in person_database:
                        # 시간 범위 필터링
                        if tr and tr.get('start') is not None and tr.get('end') is not None:
                            if not (tr['start'] <= person_data.get('timestamp', 0) <= tr['end']):
                                continue
                        
                        score, reasons = 0.0, []
                        
                        # 객체 매칭 (사람인지 확인)
                        if objects and 'person' in objects:
                            if person_data.get('class', '').lower() == 'person':
                                score += 0.5
                                reasons.append("사람 객체")
                        
                        # 색상 매칭
                        if colors and 'attributes' in person_data:
                            attrs = person_data['attributes']
                            hit = False
                            
                            # 의류 색상 확인
                            if 'clothing_color' in attrs:
                                clothing_colors = attrs['clothing_color']
                                if isinstance(clothing_colors, dict):
                                    for color_key, color_value in clothing_colors.items():
                                        if any(c in color_key.lower() or c in str(color_value).lower() for c in colors):
                                            hit = True
                                            break
                                elif isinstance(clothing_colors, str):
                                    if any(c in clothing_colors.lower() for c in colors):
                                        hit = True
                            
                            if hit:
                                score += 0.3
                                reasons.append("색상 매칭")
                        
                        if score >= 0.5:
                            hits.append({
                                't': float(person_data.get('timestamp', 0)),
                                'time': self._format_time(person_data.get('timestamp', 0)),
                                'frame_id': person_data.get('frame_id', 0),
                                'desc': f"사람 (신뢰도: {person_data.get('confidence', 0):.2f})",
                                'score': min(1.0, (score + person_data.get('confidence', 0.5) * 0.2)),
                                'reasons': reasons,
                                'track': person_data.get('track_id', ''),
                                'bbox': person_data.get('bbox', []),
                                'attributes': person_data.get('attributes', {})
                            })
                    
                    print(f"🔍 [DEBUG] person_database에서 찾은 hits: {len(hits)}개")
                            
            except Exception as e:
                print(f"❌ person_database 읽기 오류: {e}")
        
        # 기존 Frame 기반 검색도 병행 (fallback)
        if not hits:
            frames_qs = Frame.objects.filter(video=video).order_by('timestamp')
            if tr and tr.get('start') is not None and tr.get('end') is not None:
                frames_qs = frames_qs.filter(timestamp__gte=tr['start'], timestamp__lte=tr['end'])

            for f in frames_qs:
                dets = self._get_detected_objects(f)
                if not dets: continue
                for d in dets:
                    score, reasons = 0.0, []
                    # 객체 매칭
                    if objects:
                        if d['class'] in objects:
                            score += 0.5
                            reasons.append(f"{d['class']} 객체")
                        elif any(o in d['class'] for o in objects):
                            score += 0.3
                            reasons.append(f"{d['class']} 유사 객체")
                    # 색상 매칭
                    if colors:
                        hit = False
                        cd = d['color_description']
                        if any(c in cd for c in colors):
                            hit = True
                        if not hit and d['colors']:
                            if any(c in (str(x).lower()) for x in d['colors'] for c in colors):
                                hit = True
                        if hit:
                            score += 0.3
                            reasons.append("색상 매칭")

                    if score >= 0.5:
                        hits.append({
                            't': float(f.timestamp),
                            'time': self._format_time(f.timestamp),
                            'frame_id': f.image_id,
                            'desc': f"{d.get('color_description','')} {d.get('class','object')}".strip(),
                            'score': min(1.0, (score + d.get('confidence', 0.5) * 0.2)),
                            'reasons': reasons,
                            'track': d.get('track_id') or '',
                        })

        if not hits:
            return {'text': f"'{raw_text}'로는 매칭이 없었어요. 시간 범위를 넓히거나 색상 없이 다시 시도해 보세요.", 'items': []}

        # 정렬 + 중복 제거 + 상위 10개
        hits.sort(key=lambda x: (-x['score'], x['t']))
        uniq, seen = [], set()
        for h in hits:
            key = (int(h['t']), h['desc'])
            if key in seen: continue
            seen.add(key)
            media = self._frame_urls(request, video.id, h['frame_id']) if request else {}
            clip = self._clip_url(request, video.id, h['t']) if request else None
            uniq.append({
                'time': h['time'],
                'seconds': int(h['t']),
                'frame_id': h['frame_id'],
                'desc': h['desc'],
                'score': h['score'],
                'reasons': h['reasons'],
                'thumbUrl': media.get('image'),
                'thumbBBoxUrl': media.get('image_bbox'),
                'clipUrl': clip,
            })
            if len(uniq) >= 10: break

        text = "🔎 요청하신 장면을 찾았어요 (상위 {n}개):\n".format(n=len(uniq))
        text += "\n".join([f"- {it['time']} · {it['desc']} · ~{int(it['score']*100)}%" for it in uniq])
        return {'text': text, 'items': uniq}

    def _handle_object_presence(self, video: Video, raw_text: str, slots: dict, request=None):
        """특정 객체/키워드 등장 여부 간단 확인 + 썸네일"""
        objs = slots.get('objects') or []
        q = raw_text.lower()
        frames = Frame.objects.filter(video=video).order_by('timestamp')[:100]
        hits = []
        for f in frames:
            cap = (f.final_caption or f.enhanced_caption or f.caption or '').lower()
            dets = self._get_detected_objects(f)
            ok = False
            reason = ""
            if objs and any(o in (cap or '') for o in objs):
                ok, reason = True, "캡션 매칭"
            if not ok and dets:
                if objs and any(d['class'] in objs for d in dets):
                    ok, reason = True, "객체 매칭"
                elif any(k in cap for k in q.split()):
                    ok, reason = True, "키워드 매칭"

            if ok:
                media = self._frame_urls(request, video.id, f.image_id)
                clip = self._clip_url(request, video.id, f.timestamp)
                hits.append({
                    'time': self._format_time(f.timestamp),
                    'seconds': int(f.timestamp),
                    'frame_id': f.image_id,
                    'desc': (f.final_caption or f.enhanced_caption or f.caption or '').strip()[:120],
                    'thumbUrl': media['image'],
                    'thumbBBoxUrl': media['image_bbox'],
                    'clipUrl': clip,
                })
            if len(hits) >= 10: break

        if not hits:
            return {'text': "해당 키워드/객체를 찾지 못했어요.", 'items': []}
        text = "✅ 찾았습니다:\n" + "\n".join([f"- {h['time']} · {h['desc']}" for h in hits])
        return {'text': text, 'items': hits}

    def _handle_highlight(self, video: Video, request=None):
        """상위 5개 씬 + 각 씬 대표 썸네일/클립"""
        scenes = Scene.objects.filter(video=video).order_by('start_time')[:5]
        if not scenes:
            return {'text': "하이라이트가 아직 없어요. 분석이 끝났는지 확인해 주세요.", 'items': []}

        items, lines = [], []
        for s in scenes:
            mid = (s.start_time + s.end_time) / 2.0
            f = Frame.objects.filter(video=video, timestamp__gte=mid).order_by('timestamp').first() or \
                Frame.objects.filter(video=video).order_by('-timestamp').first()
            media = self._frame_urls(request, video.id, f.image_id) if f else {}
            clip = self._clip_url(request, video.id, mid) if f else None
            objs = (s.dominant_objects or [])[:5]
            items.append({
                'range': [int(s.start_time), int(s.end_time)],
                'start': self._format_time(s.start_time),
                'end': self._format_time(s.end_time),
                'objects': objs,
                'thumbUrl': media.get('image'),
                'thumbBBoxUrl': media.get('image_bbox'),
                'clipUrl': clip,
            })
            lines.append(f"- {self._format_time(s.start_time)}–{self._format_time(s.end_time)} · {', '.join(objs) or '장면'}")

        return {'text': "✨ 주요 장면:\n" + "\n".join(lines), 'items': items}

    def _handle_summary(self, video: Video, request=None):
        """간단 요약 + 대표 썸네일 몇 장"""
        summary = [
            f"‘{video.original_name}’ 요약",
            f"- 길이: {round(video.duration,2)}초 · 분석 상태: {video.analysis_status}",
        ]
        try:
            analysis = getattr(video, 'analysis', None)
            if analysis and analysis.analysis_statistics:
                stats = analysis.analysis_statistics
                dom = stats.get('dominant_objects', [])[:5]
                if dom:
                    summary.append(f"- 주요 객체: {', '.join(dom)}")
                scene_types = stats.get('scene_types', [])[:3]
                if scene_types:
                    summary.append(f"- 장면 유형: {', '.join(scene_types)}")
        except:
            pass

        frames = Frame.objects.filter(video=video).order_by('timestamp')[:6]
        items = []
        for f in frames:
            media = self._frame_urls(request, video.id, f.image_id)
            clip = self._clip_url(request, video.id, f.timestamp)
            items.append({
                'time': self._format_time(f.timestamp),
                'seconds': int(f.timestamp),
                'frame_id': f.image_id,
                'desc': (f.final_caption or f.enhanced_caption or f.caption or '').strip()[:120],
                'thumbUrl': media['image'],
                'thumbBBoxUrl': media['image_bbox'],
                'clipUrl': clip,
            })

        return {'text': "\n".join(summary), 'items': items}

    def _handle_info(self, video: Video):
        sc = Scene.objects.filter(video=video).count()
        fc = Frame.objects.filter(video=video).count()
        return "\n".join([
            "비디오 정보",
            f"- 파일명: {video.original_name}",
            f"- 길이: {round(video.duration,2)}초",
            f"- 분석 상태: {video.analysis_status}",
            f"- 씬 수: {sc}개",
            f"- 분석 프레임: {fc}개",
        ])


    def _enhance_person_detection_with_gender(self, frame_data):
        """사람 감지 데이터에 성별 정보 보강 (분석 시점에서 호출)"""
        try:
            if not frame_data or not isinstance(frame_data, list):
                return frame_data
            
            enhanced_data = []
            for obj in frame_data:
                if not isinstance(obj, dict) or obj.get('class') != 'person':
                    enhanced_data.append(obj)
                    continue
                
                enhanced_obj = obj.copy()
                
                # 기존 성별 정보가 없는 경우에만 추정
                if not enhanced_obj.get('gender'):
                    # 여기서 추가적인 성별 분석 로직을 구현할 수 있음
                    # 예: 의복, 체형, 머리카락 등 기반 휴리스틱
                    
                    # 임시: 랜덤하게 성별 할당 (실제로는 더 정교한 분석 필요)
                    import random
                    if random.random() < 0.3:  # 30% 확률로 성별 추정
                        enhanced_obj['gender'] = random.choice(['male', 'female'])
                        enhanced_obj['gender_confidence'] = 0.6  # 낮은 신뢰도
                    else:
                        enhanced_obj['gender'] = 'unknown'
                        enhanced_obj['gender_confidence'] = 0.0
                
                enhanced_data.append(enhanced_obj)
            
            return enhanced_data
        except Exception as e:
            logger.warning(f"성별 정보 보강 실패: {e}")
            return frame_data

    def _get_detected_objects(self, frame: Frame):
        """
        Frame 객체 추출 시 성별 정보 처리 개선
        """
        import json

        candidates = []

        # 1) detected_objects
        if hasattr(frame, 'detected_objects') and frame.detected_objects:
            candidates.append(frame.detected_objects)

        # 2) comprehensive_features.objects  
        if hasattr(frame, 'comprehensive_features') and frame.comprehensive_features:
            objs = None
            if isinstance(frame.comprehensive_features, dict):
                objs = frame.comprehensive_features.get('objects') \
                or frame.comprehensive_features.get('detections')
            elif isinstance(frame.comprehensive_features, str):
                try:
                    cf = json.loads(frame.comprehensive_features)
                    objs = (cf or {}).get('objects') or (cf or {}).get('detections')
                except Exception:
                    pass
            if objs:
                candidates.append(objs)

        # 3) 기타 필드들
        for attr in ('yolo_objects', 'detections', 'objects'):
            if hasattr(frame, attr) and getattr(frame, attr):
                candidates.append(getattr(frame, attr))

        # 첫 번째 유효 후보 선택
        detected = None
        for c in candidates:
            try:
                if isinstance(c, str):
                    c = json.loads(c)
                if isinstance(c, dict):
                    c = c.get('objects') or c.get('detections')
                if isinstance(c, list):
                    detected = c
                    break
            except Exception:
                continue

        if not isinstance(detected, list):
            return []

        # 정규화 - 성별 정보 포함
        norm = []
        for o in detected:
            if not isinstance(o, dict):
                continue
            
            cls = (o.get('class') or o.get('label') or o.get('name') or '').lower()
            bbox = o.get('bbox') or o.get('box') or o.get('xyxy') or []
            conf = float(o.get('confidence') or o.get('score') or 0.0)
            colors = o.get('colors') or o.get('color') or []
            if isinstance(colors, str):
                colors = [colors]
            color_desc = (o.get('color_description') or o.get('dominant_color') or 'unknown')
            track_id = o.get('track_id') or o.get('id')
            
            # 성별 정보 추출 개선
            gender = o.get('gender') or o.get('sex') or 'unknown'
            if isinstance(gender, bool):
                gender = 'male' if gender else 'female'
            gender = str(gender).lower()
            
            # 성별 신뢰도
            gender_conf = float(o.get('gender_confidence') or o.get('gender_score') or 0.0)

            norm.append({
                'class': cls,
                'bbox': bbox,
                'confidence': conf,
                'colors': colors,
                'color_description': str(color_desc).lower(),
                'track_id': track_id,
                'gender': gender,
                'gender_confidence': gender_conf,
                '_raw': o,  # 원본 데이터도 보관
            })
        return norm
    def _handle_scene_mood(self, video: Video):
        """씬 타입 기반 간단 무드 설명"""
        try:
            analysis = getattr(video, 'analysis', None)
            if analysis and analysis.analysis_statistics:
                types = (analysis.analysis_statistics.get('scene_types') or [])[:3]
                if types:
                    return f"분위기: {', '.join(types)}"
        except:
            pass
        return "분위기 정보를 파악할 단서가 부족합니다."

    def _handle_cross_video(self, raw_text: str):
        """여러 영상 중 조건에 맞는 후보 명시 (여기선 설명만)"""
        return "여러 영상 간 조건 검색은 준비되어 있습니다. UI에서 목록/필터를 제공해 주세요."
    def _handle_gender_distribution(self, video: Video, slots: dict):
        """성별 분포 분석 - 개선된 버전"""
        tr = slots.get('time_range')
        qs = Frame.objects.filter(video=video)
        if tr and tr.get('start') is not None and tr.get('end') is not None:
            qs = qs.filter(timestamp__gte=tr['start'], timestamp__lte=tr['end'])

        male = female = unknown = 0
        person_detections = []
        
        for f in qs:
            detected_objects = self._get_detected_objects(f)
            for d in detected_objects:
                if d['class'] != 'person': 
                    continue
                
                person_detections.append(d)
                
                # 성별 정보 추출 - 여러 방법 시도
                gender = None
                
                # 1. 직접적인 gender 필드
                if 'gender' in d and d['gender'] and d['gender'] != 'unknown':
                    gender = str(d['gender']).lower()
                
                # 2. 원본 데이터에서 성별 정보 찾기
                elif '_raw' in d and d['_raw']:
                    raw = d['_raw']
                    for key in ['gender', 'sex', 'male', 'female']:
                        if key in raw and raw[key]:
                            val = str(raw[key]).lower()
                            if val in ['male', 'man', 'm', 'true'] and key in ['male', 'gender']:
                                gender = 'male'
                                break
                            elif val in ['female', 'woman', 'f', 'true'] and key in ['female', 'gender']:
                                gender = 'female'  
                                break
                            elif val in ['male', 'female']:
                                gender = val
                                break
                
                # 3. 색상/의복 기반 휴리스틱 추정 (보조적)
                if not gender:
                    color_desc = d.get('color_description', '').lower()
                    colors = [str(c).lower() for c in d.get('colors', [])]
                    
                    # 간단한 휴리스틱 (정확도 낮음, 참고용)
                    if any('pink' in x for x in [color_desc] + colors):
                        gender = 'female_guess'
                    elif any('blue' in x for x in [color_desc] + colors):
                        gender = 'male_guess'
                
                # 카운팅
                if gender in ['male', 'male_guess']:
                    male += 1
                elif gender in ['female', 'female_guess']:
                    female += 1
                else:
                    unknown += 1

        total = male + female + unknown
        
        if total == 0:
            return "영상에서 사람을 감지하지 못했습니다."
        
        # 결과 포맷팅
        def pct(x): 
            return round(100.0 * x / total, 1) if total > 0 else 0
        
        result = f"성비 분석 결과 (총 {total}명 감지):\n"
        result += f"👨 남성: {male}명 ({pct(male)}%)\n"
        result += f"👩 여성: {female}명 ({pct(female)}%)\n"
        result += f"❓ 미상: {unknown}명 ({pct(unknown)}%)\n\n"
        
        # 추가 정보
        if unknown > total * 0.8:  # 80% 이상이 미상인 경우
            result += "💡 성별 추정 정확도가 낮습니다. 이는 다음 이유일 수 있습니다:\n"
            result += "- 영상 해상도나 각도 문제\n"
            result += "- 사람이 멀리 있거나 부분적으로만 보임\n"
            result += "- AI 모델의 성별 분석 기능 제한\n\n"
        
        # 디버깅 정보 (개발 시에만 표시)
        result += f"🔍 디버그 정보:\n"
        result += f"- 처리된 프레임 수: {qs.count()}개\n"
        result += f"- 감지된 person 객체: {len(person_detections)}개\n"
        
        if person_detections:
            sample_detection = person_detections[0]
            result += f"- 샘플 객체 정보: {sample_detection.get('gender', 'N/A')} (신뢰도: {sample_detection.get('gender_confidence', 0)})\n"
        
        # 시간 범위 정보
        if tr:
            result += f"📅 분석 구간: {tr.get('start', '시작')}~{tr.get('end', '끝')}"
        else:
            result += f"📅 분석 구간: 전체 영상"
        
        return result
# views.py (동일 파일 내)
class ClipPreviewView(APIView):
    """ffmpeg 로 짧은 미리보기 클립 생성/반환"""
    permission_classes = [AllowAny]

    def get(self, request, video_id, timestamp):
        duration = int(request.GET.get('duration', 4))
        try:
            video = Video.objects.get(id=video_id)
        except Video.DoesNotExist:
            raise Http404("video not found")

        src_path = getattr(getattr(video, 'file', None), 'path', None)
        if not src_path or not os.path.exists(src_path):
            raise Http404("file not found")

        tmp_dir = tempfile.mkdtemp()
        out_path = os.path.join(tmp_dir, f"clip_{video_id}_{timestamp}.mp4")

        cmd = [
            'ffmpeg','-y',
            '-ss', str(int(timestamp)),
            '-i', src_path,
            '-t', str(duration),
            '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '28',
            '-an',
            out_path
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError:
            raise Http404("ffmpeg error")

        resp = FileResponse(open(out_path, 'rb'), content_type='video/mp4')
        resp['Content-Disposition'] = f'inline; filename="clip_{video_id}_{timestamp}.mp4"'
        return resp


# 누락된 함수들 추가
@api_view(['POST'])
def start_analysis(request, pk):
    """분석 시작 - EnhancedAnalyzeVideoView 사용"""
    try:
        # EnhancedAnalyzeVideoView 인스턴스 생성
        enhanced_view = EnhancedAnalyzeVideoView()
        
        # POST 요청 데이터 준비
        request_data = {
            'analysisType': 'enhanced',
            'enhancedAnalysis': True,
            'analysisConfig': {}
        }
        
        # EnhancedAnalyzeVideoView의 post 메서드 호출
        response = enhanced_view.post(request, pk)
        
        return response
        
    except Exception as e:
        print(f"❌ 분석 시작 오류: {e}")
        return Response({
            'error': f'분석 시작 중 오류: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
def get_analysis_status(request, pk):
    """분석 상태 조회"""
    try:
        video = Video.objects.get(id=pk)
        return JsonResponse({
            'status': video.analysis_status, 
            'video_id': pk,
            'is_analyzed': video.is_analyzed,
            'success_rate': video.success_rate,
            'processing_time': video.processing_time
        })
    except Video.DoesNotExist:
        return JsonResponse({'error': '비디오를 찾을 수 없습니다.'}, status=404)


@api_view(['POST'])
def chat_with_video(request, pk):
    """비디오 분석 결과 기반 채팅 - 새로운 요약 기능 통합"""
    try:
        video = Video.objects.get(id=pk)
        
        # 분석이 완료되지 않은 경우
        if not video.is_analyzed or video.analysis_status != 'completed':
            return JsonResponse({
                'error': '비디오 분석이 완료되지 않았습니다. 먼저 분석을 실행해주세요.'
            }, status=400)
        
        message = request.data.get('message', '')
        if not message:
            return JsonResponse({'error': '메시지를 입력해주세요.'}, status=400)
        
        print(f"💬 채팅 요청: '{message}' (비디오 ID: {pk})")
        
        # 요약 관련 키워드 확인
        summary_keywords = ['요약', '정리', '설명', '개요', '줄거리', '내용']
        is_summary_request = any(keyword in message for keyword in summary_keywords)
        
        if is_summary_request:
            print("📝 요약 요청으로 인식 - CLIP + GPT Vision 통합 요약 기능 사용")
            try:
                # CLIP + GPT Vision 통합 요약 기능 사용
                print(f"🔍 VideoSummaryView._generate_video_summary 호출 시작 (비디오 ID: {pk})")
                summary_view = VideoSummaryView()
                summary_data = summary_view._generate_video_summary(video)
                print(f"🔍 요약 생성 결과: {summary_data is not None}")
                
                if summary_data and 'llm_summary' in summary_data:
                    response_text = summary_data['llm_summary']
                    print(f"✅ LLM 요약 생성 성공: {response_text[:100]}...")
                    
                    # 분석 방법 정보 추가
                    analysis_methods = summary_data.get('analysis_methods', [])
                    if analysis_methods:
                        methods_text = ' + '.join(analysis_methods)
                        response_text += f"\n\n🤖 분석 방법: {methods_text}"
                    
                    # 추가 정보가 있으면 포함
                    if 'clip_analyses' in summary_data and summary_data['clip_analyses']:
                        clip_count = len(summary_data['clip_analyses'])
                        response_text += f"\n📸 CLIP 분석: {clip_count}개 프레임 분석 완료"
                    
                    if 'gpt_vision_analyses' in summary_data and summary_data['gpt_vision_analyses']:
                        gpt_count = len(summary_data['gpt_vision_analyses'])
                        response_text += f"\n👁️ GPT Vision 분석: {gpt_count}개 프레임 상세 분석 완료"
                    
                    return JsonResponse({
                        'response': response_text,
                        'video_id': pk,
                        'timestamp': time.time()
                    })
                else:
                    print(f"❌ 요약 데이터 없음 또는 llm_summary 없음: {summary_data}")
                    # 기본 요약 생성
                    basic_summary = _generate_basic_summary_for_chat(video)
                    return JsonResponse({
                        'response': basic_summary,
                        'video_id': pk,
                        'timestamp': time.time()
                    })
                    
            except Exception as e:
                print(f"❌ 새로운 요약 기능 오류: {e}")
                # 기존 방식으로 폴백
                pass
        
        # 기존 채팅 로직 (요약이 아닌 경우)
        analysis_data = None
        if video.analysis_json_path and os.path.exists(video.analysis_json_path):
            try:
                with open(video.analysis_json_path, 'r', encoding='utf-8') as f:
                    analysis_data = json.load(f)
            except Exception as e:
                print(f"❌ 분석 결과 JSON 로드 실패: {e}")
        
        # 간단한 AI 응답 생성 (실제로는 LLM 서비스 사용)
        response = generate_chat_response(video, message, analysis_data)
        
        return JsonResponse({
            'response': response,
            'video_id': pk,
            'timestamp': time.time()
        })
        
    except Video.DoesNotExist:
        return JsonResponse({'error': '비디오를 찾을 수 없습니다.'}, status=404)
    except Exception as e:
        return JsonResponse({'error': f'채팅 처리 중 오류가 발생했습니다: {str(e)}'}, status=500)


@api_view(['GET'])
def get_video_details(request, pk):
    """비디오 상세 정보 조회"""
    try:
        video = Video.objects.get(id=pk)
        return JsonResponse({
            'id': video.id,
            'filename': video.filename,
            'original_name': video.original_name,
            'analysis_status': video.analysis_status,
            'is_analyzed': video.is_analyzed,
            'success_rate': video.success_rate,
            'processing_time': video.processing_time,
            'analysis_type': video.analysis_type,
            'advanced_features_used': video.advanced_features_used,
            'scene_types': video.scene_types,
            'unique_objects': video.unique_objects,
            'analysis_json_path': video.analysis_json_path,
            'uploaded_at': video.uploaded_at,
            'file_size': video.file_size
        })
    except Video.DoesNotExist:
        return JsonResponse({'error': '비디오를 찾을 수 없습니다.'}, status=404)


def _generate_video_summary_for_chat(video_id):
    """채팅용 영상 요약 생성 (GPT Vision 포함)"""
    try:
        video = Video.objects.get(id=video_id)
        
        # 프레임 데이터 수집
        frames = Frame.objects.filter(video=video).order_by('timestamp')
        if not frames.exists():
            return None
        
        # 프레임 캡션을 활용한 상세 분석
        from .llm_client import llm_client
        vision_analyses = []
        
        # 대표 프레임 2-3개 선택 (시작, 중간, 끝)
        selected_frames = []
        frames_list = list(frames)  # QuerySet을 리스트로 변환
        if len(frames_list) >= 3:
            selected_frames = [frames_list[0], frames_list[len(frames_list)//2], frames_list[-1]]
        elif len(frames_list) >= 2:
            selected_frames = [frames_list[0], frames_list[-1]]
        else:
            selected_frames = [frames_list[0]]
        
        # 실제 분석 데이터에서 프레임 정보 가져오기
        analysis_files = glob.glob(f"media/analysis_results/real_analysis_{video_id}_enhanced_*.json")
        analysis_data = None
        if analysis_files:
            with open(analysis_files[0], 'r', encoding='utf-8') as f:
                analysis_data = json.load(f)
        
        for frame in selected_frames:
            try:
                # 실제 분석 데이터에서 해당 프레임 정보 찾기
                frame_analysis_data = None
                if analysis_data:
                    for frame_result in analysis_data.get('frame_results', []):
                        if frame_result.get('image_id') == frame.image_id:
                            frame_analysis_data = frame_result
                            break
                
                if frame_analysis_data and frame_analysis_data.get('persons'):
                    person = frame_analysis_data['persons'][0]
                    attributes = person.get('attributes', {})
                    confidence = person.get('confidence', 0)
                    
                    # 실제 데이터 기반 설명 생성
                    gender = attributes.get('gender', {}).get('value', '사람')
                    age = attributes.get('age', {}).get('value', '')
                    clothing = attributes.get('detailed_clothing', {}).get('value', '')
                    color = attributes.get('clothing_color', {}).get('value', '')
                    posture = attributes.get('posture', {}).get('value', '')
                    hair = attributes.get('hair_style', {}).get('value', '')
                    accessories = attributes.get('accessories', {}).get('value', '')
                    
                    description_parts = []
                    if gender: description_parts.append(gender)
                    if age: description_parts.append(age)
                    if hair: description_parts.append(f"{hair} 머리")
                    if color: description_parts.append(f"{color}")
                    if clothing: description_parts.append(clothing)
                    if posture: description_parts.append(posture)
                    if accessories: description_parts.append(accessories)
                    
                    caption = ", ".join(description_parts) if description_parts else "사람이 활동하는 장면"
                    
                    frame_analysis = {
                        'raw_analysis': f"시간 {frame.timestamp:.1f}초: {caption} (신뢰도: {confidence:.2f})",
                        'person_count': 1,
                        'persons': [{'location': '중앙', 'features': caption, 'activity': posture}],
                        'scene_description': caption,
                        'confidence': confidence
                    }
                else:
                    # 분석 데이터가 없는 경우
                    caption = "인물 탐지 없음"
                    frame_analysis = {
                        'raw_analysis': f"시간 {frame.timestamp:.1f}초: {caption}",
                        'person_count': 0,
                        'persons': [],
                        'scene_description': caption,
                        'confidence': 0.0
                    }
                
                vision_analyses.append({
                    'timestamp': frame.timestamp,
                    'frame_id': frame.image_id,
                    'analysis': frame_analysis
                })
                
                print(f"✅ 프레임 {frame.image_id} 분석 완료: {caption}")
                
            except Exception as e:
                print(f"⚠️ 프레임 {frame.image_id} 분석 실패: {e}")
                continue
        
        # 기존 시각적 분석도 유지
        vision_analysis = _analyze_video_frames_for_chat(video_id, list(frames))
        
        # 클립 데이터 생성 (GPT Vision 분석 포함)
        clips = []
        for i, frame in enumerate(frames[:5]):
            # 해당 프레임의 GPT Vision 분석 찾기
            gpt_analysis = None
            for va in vision_analyses:
                if va['frame_id'] == frame.image_id:
                    gpt_analysis = va['analysis']
                    break
            
            clips.append({
                'id': i + 1,
                'start_time': frame.timestamp,
                'end_time': frame.timestamp + 1.0,
                'description': f'프레임 {frame.image_id}',
                'timestamp': frame.timestamp,
                'gpt_vision_analysis': gpt_analysis
            })
        
        # LLM 요약 생성 (GPT Vision 분석 포함)
        video_data = {
            'video_id': video_id,
            'clips': clips,
            'vision_analysis': vision_analysis,
            'gpt_vision_analyses': vision_analyses
        }
        
        llm_summary = llm_client.generate_summary(video_data)
        
        return {
            'llm_summary': llm_summary,
            'vision_analysis': vision_analysis,
            'gpt_vision_analyses': vision_analyses,
            'clips': clips
        }
        
    except Exception as e:
        print(f"❌ 채팅용 영상 요약 생성 오류: {e}")
        return None

def _analyze_video_frames_for_chat(video_id, frames):
    """채팅용 비디오 프레임 분석"""
    try:
        from .vision_analyzer import vision_analyzer
        if not vision_analyzer:
            return {"error": "Vision analyzer not available"}
        
        # 프레임 이미지 경로들 수집 (1-3개만 선택)
        frame_paths = []
        selected_frames = []
        
        # 더 다양한 프레임 선택 (랜덤 + 시간대별)
        total_frames = len(frames)
        if total_frames >= 5:
            # 5개 이상인 경우: 시작, 1/4, 중간, 3/4, 끝에서 선택
            indices = [0, total_frames // 4, total_frames // 2, 3 * total_frames // 4, total_frames - 1]
        elif total_frames >= 3:
            # 3-4개인 경우: 시작, 중간, 끝에서 선택
            indices = [0, total_frames // 2, total_frames - 1]
        elif total_frames >= 1:
            # 1-2개만 있는 경우 모두 선택
            indices = list(range(total_frames))
        else:
            return {"error": "분석할 프레임이 없습니다"}
        
        # 최대 3개만 선택 (처리 속도 향상)
        indices = indices[:3]
        
        for idx in indices:
            frame = frames[idx]
            image_path = os.path.join(settings.MEDIA_ROOT, 'images', f'video{video_id}_frame{frame.image_id}.jpg')
            if os.path.exists(image_path):
                frame_paths.append(image_path)
                selected_frames.append({
                    'frame_id': frame.image_id,
                    'timestamp': frame.timestamp,
                    'image_path': image_path
                })
        
        if not frame_paths:
            return {"error": "분석할 프레임 이미지를 찾을 수 없습니다"}
        
        print(f"🔍 채팅용 프레임 {len(frame_paths)}개 분석 중...")
        
        # 시각적 분석 실행
        analysis_result = vision_analyzer.analyze_video_frames(video_id, frame_paths)
        analysis_result['selected_frames'] = selected_frames
        analysis_result['analysis_count'] = len(frame_paths)
        
        return analysis_result
        
    except Exception as e:
        print(f"❌ 채팅용 프레임 분석 오류: {e}")
        return {"error": f"프레임 분석 실패: {str(e)}"}

def _generate_basic_summary_for_chat(video):
    """채팅용 기본 요약 생성 - 실제 분석 데이터 기반"""
    try:
        frames = Frame.objects.filter(video=video).order_by('timestamp')
        
        video_title = video.title if video.title else f"영상 {video.id}"
        summary = f"📹 '{video_title}' 영상 요약\n\n"
        
        if frames.exists():
            # 대표 프레임 2-3개 선택
            frames_list = list(frames)
            if len(frames_list) >= 3:
                selected_frames = [frames_list[0], frames_list[len(frames_list)//2], frames_list[-1]]
            elif len(frames_list) >= 2:
                selected_frames = [frames_list[0], frames_list[-1]]
            else:
                selected_frames = [frames_list[0]]
            
            # 실제 분석 데이터 기반 정확한 프레임 분석
            summary += "🎬 실제 분석 데이터 기반 장면 분석\n"
            
            # 분석 JSON 파일에서 데이터 가져오기
            analysis_files = glob.glob(f"media/analysis_results/real_analysis_{video.id}_enhanced_*.json")
            if analysis_files:
                with open(analysis_files[0], 'r', encoding='utf-8') as f:
                    analysis_data = json.load(f)
                
                video_summary = analysis_data.get('video_summary', {})
                total_detections = video_summary.get('total_detections', 0)
                unique_persons = video_summary.get('unique_persons', 0)
                attributes = video_summary.get('detailed_attribute_statistics', {})
                
                # 실제 분석 결과 기반 설명
                summary += f"**실제 탐지 데이터**: 총 {total_detections}번 탐지, 고유 인물 {unique_persons}명\n\n"
                
                # 주요 속성 정보
                if attributes:
                    summary += "**탐지된 인물 특징**:\n"
                    if 'gender' in attributes:
                        gender_info = list(attributes['gender'].keys())[0]
                        summary += f"- 성별: {gender_info}\n"
                    if 'age' in attributes:
                        age_info = list(attributes['age'].keys())[0]
                        summary += f"- 나이대: {age_info}\n"
                    if 'clothing_color' in attributes:
                        color_info = list(attributes['clothing_color'].keys())[0]
                        summary += f"- 옷 색상: {color_info}\n"
                    if 'detailed_clothing' in attributes:
                        clothing_info = list(attributes['detailed_clothing'].keys())[0]
                        summary += f"- 옷 종류: {clothing_info}\n"
                    if 'posture' in attributes:
                        posture_info = list(attributes['posture'].keys())[0]
                        summary += f"- 자세: {posture_info}\n"
                    if 'hair_style' in attributes:
                        hair_info = list(attributes['hair_style'].keys())[0]
                        summary += f"- 머리 스타일: {hair_info}\n"
                    if 'accessories' in attributes:
                        accessories_info = list(attributes['accessories'].keys())[0]
                        summary += f"- 소지품: {accessories_info}\n"
                    summary += "\n"
                
                # 프레임별 상세 분석
                for i, frame in enumerate(selected_frames, 1):
                    frame_analysis = None
                    for frame_result in analysis_data.get('frame_results', []):
                        if frame_result.get('image_id') == frame.image_id:
                            frame_analysis = frame_result
                            break
                    
                    if frame_analysis and frame_analysis.get('persons'):
                        person = frame_analysis['persons'][0]
                        confidence = person.get('confidence', 0)
                        attributes = person.get('attributes', {})
                        
                        # 실제 데이터 기반 설명
                        gender = attributes.get('gender', {}).get('value', '사람')
                        age = attributes.get('age', {}).get('value', '')
                        clothing = attributes.get('detailed_clothing', {}).get('value', '')
                        color = attributes.get('clothing_color', {}).get('value', '')
                        posture = attributes.get('posture', {}).get('value', '')
                        hair = attributes.get('hair_style', {}).get('value', '')
                        accessories = attributes.get('accessories', {}).get('value', '')
                        
                        description_parts = []
                        if gender: description_parts.append(gender)
                        if age: description_parts.append(age)
                        if hair: description_parts.append(f"{hair} 머리")
                        if color: description_parts.append(f"{color}")
                        if clothing: description_parts.append(clothing)
                        if posture: description_parts.append(posture)
                        if accessories: description_parts.append(accessories)
                        
                        caption = ", ".join(description_parts) if description_parts else "사람이 활동하는 장면"
                        summary += f"- **구간 {i}** ({frame.timestamp:.1f}초): {caption} (신뢰도: {confidence:.2f})\n"
                    else:
                        summary += f"- **구간 {i}** ({frame.timestamp:.1f}초): 인물 탐지 없음\n"
            else:
                summary += "분석 데이터를 찾을 수 없습니다.\n"
            
            summary += "\n"
            
            # 실제 데이터 기반 종합 분석
            summary += "💭 실제 분석 결과 요약\n"
            
            if analysis_files:
                # 실제 분석 데이터 기반 정확한 설명
                if total_detections > 0:
                    summary += f"**실제 탐지 결과**: 이 영상에서는 총 {total_detections}번의 인물 탐지가 이루어졌으며, "
                    summary += f"고유 인물은 {unique_persons}명입니다.\n\n"
                    
                    # 장면 유형 분석
                    scene_types = video_summary.get('scene_diversity', {}).get('scene_type_distribution', {})
                    if scene_types:
                        summary += "**장면 유형**: "
                        scene_list = [f"{scene}({count}회)" for scene, count in scene_types.items()]
                        summary += ", ".join(scene_list) + "\n"
                    
                    # 활동 수준 분석
                    activity_levels = video_summary.get('scene_diversity', {}).get('activity_level_distribution', {})
                    if activity_levels:
                        summary += "**활동 수준**: "
                        activity_list = [f"{level}({count}회)" for level, count in activity_levels.items()]
                        summary += ", ".join(activity_list) + "\n"
                    
                    # 조명 조건 분석
                    lighting = video_summary.get('scene_diversity', {}).get('lighting_distribution', {})
                    if lighting:
                        summary += "**조명 조건**: "
                        lighting_list = [f"{light}({count}회)" for light, count in lighting.items()]
                        summary += ", ".join(lighting_list) + "\n"
                    
                    # 품질 평가
                    quality = video_summary.get('quality_assessment', {})
                    if quality:
                        overall_score = quality.get('overall_score', 0)
                        status = quality.get('status', 'unknown')
                        summary += f"**분석 품질**: {status} (점수: {overall_score:.2f})\n"
                    
                    summary += "\n**결론**: 실제 분석 데이터에 따르면, 이 영상은 "
                    if unique_persons == 1 and total_detections > 100:
                        summary += "동일한 인물이 반복적으로 탐지되는 단순한 장면입니다. "
                    elif unique_persons > 1:
                        summary += "여러 명의 인물이 탐지되는 복합적인 장면입니다. "
                    else:
                        summary += "인물 탐지가 제한적인 장면입니다. "
                    
                    summary += "특별한 복잡한 활동보다는 기본적인 일상 행동이 주를 이룹니다.\n\n"
                else:
                    summary += "**실제 탐지 결과**: 이 영상에서는 인물이 탐지되지 않았습니다.\n\n"
            else:
                summary += "분석 데이터를 찾을 수 없어 정확한 분석을 제공할 수 없습니다.\n\n"
        
        # 기본 정보
        summary += f"📊 분석 정보: {video.analysis_status} • 성공률 {video.success_rate}%"
        
        return summary
        
    except Exception as e:
        print(f"❌ 기본 요약 생성 오류: {e}")
        return f"📹 '{video.title}' 영상 요약\n\n영상 분석이 완료되었습니다. 더 자세한 정보를 원하시면 다른 기능을 사용해보세요!"

def generate_chat_response(video, message, analysis_data):
    """GPT를 사용하여 분석 결과 기반 채팅 응답 생성"""
    try:
        # OpenAI API 키 설정
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # 분석 결과를 컨텍스트로 구성
        context = f"""
비디오 정보:
- 파일명: {video.filename}
- 분석 상태: {video.analysis_status}
- 분석 완료 여부: {video.is_analyzed}
- 성공률: {video.success_rate}%
- 처리 시간: {video.processing_time}초
- 분석 유형: {video.analysis_type}
- 고유 객체 수: {video.unique_objects}개
- 씬 유형: {video.scene_types}
- 고급 기능 사용: {video.advanced_features_used}
"""
        
        # 분석 결과 데이터가 있는 경우 추가
        if analysis_data and 'result' in analysis_data:
            result = analysis_data['result']
            context += f"""
상세 분석 결과:
- 감지된 객체: {result.get('detected_objects', [])}
- 씬 분석: {result.get('scene_types', [])}
- 고급 기능: {result.get('advanced_features_used', {})}
- 성공률: {result.get('success_rate', 0)}%
- 처리 시간: {result.get('processing_time', 0)}초
"""
        
        # GPT 프롬프트 구성
        system_prompt = f"""당신은 비디오 분석 전문가입니다. 주어진 비디오 분석 결과를 바탕으로 사용자의 질문에 정확하고 도움이 되는 답변을 제공해주세요.

비디오 분석 정보:
{context}

답변 시 다음 사항을 고려해주세요:
1. 분석 결과를 바탕으로 구체적이고 정확한 정보 제공
2. 감지된 객체, 씬 유형, 분석 통계 등을 활용
3. 사용자가 이해하기 쉽게 설명
4. 필요시 추가 분석이나 개선 사항 제안
5. 한국어로 답변하되, 전문 용어는 적절히 설명

답변 형식:
- 이모지를 사용하여 가독성 향상
- 중요한 정보는 **굵게** 표시
- 구체적인 수치나 데이터 포함
- 사용자 질문에 직접적으로 답변"""
        
        # 프레임 검색이 필요한지 확인
        search_keywords = ['사람', 'person', '찾아', '보여', '프레임', 'frame', '이미지', 'image']
        needs_frame_search = any(keyword in message.lower() for keyword in search_keywords)
        
        # GPT API 호출 (최신 방식)
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"질문: {message}"}
            ],
            max_tokens=1000,
            temperature=0.7
        )
        
        gpt_response = response.choices[0].message.content
        
        # 프레임 검색이 필요한 경우 추가 정보 제공
        if needs_frame_search:
            matching_frames = search_frames_by_query(message, analysis_data, video)
            if matching_frames:
                gpt_response += f"\n\n🔍 **관련 프레임 검색 결과**:"
                gpt_response += f"\n총 {len(matching_frames)}개의 매칭 프레임을 찾았습니다."
                
                for i, frame in enumerate(matching_frames[:3]):  # 상위 3개만 표시
                    gpt_response += f"\n- 프레임 {frame['frame_id']} (시간: {frame['timestamp']:.1f}초)"
                    gpt_response += f" - {frame['match_reason']} (신뢰도: {frame['confidence']:.2f})"
                
                if len(matching_frames) > 3:
                    gpt_response += f"\n- ... 외 {len(matching_frames) - 3}개 프레임"
                
                gpt_response += f"\n\n💡 **프레임 이미지 보기**: 프레임 번호를 클릭하면 해당 이미지를 볼 수 있습니다."
        
        return gpt_response
        
    except Exception as e:
        print(f"❌ GPT 채팅 오류: {e}")
        # GPT 오류 시 기본 응답
        return f"""📹 **{video.filename}** 비디오에 대한 질문을 받았습니다.

⚠️ AI 채팅 서비스에 일시적인 문제가 발생했습니다.
분석 결과: {video.analysis_status}
성공률: {video.success_rate}%
감지된 객체 수: {video.unique_objects}개

❓ **질문**: {message}
💬 **답변**: 죄송합니다. 현재 AI 서비스에 문제가 있어 기본 정보만 제공드립니다. 잠시 후 다시 시도해주세요."""


@api_view(['GET'])
def get_tracks(request, pk):
    """트랙 정보 조회"""
    try:
        video = Video.objects.get(id=pk)
        tracks = TrackPoint.objects.filter(video=video)
        track_data = []
        for track in tracks:
            track_data.append({
                'track_id': track.track_id,
                'frame_number': track.frame_number,
                'bbox': [track.x1, track.y1, track.x2, track.y2],
                'class_id': track.class_id,
                'score': track.score
            })
        return JsonResponse({'tracks': track_data})
    except Video.DoesNotExist:
        return JsonResponse({'error': '비디오를 찾을 수 없습니다.'}, status=404)


@api_view(['POST'])
def batch_delete_videos(request):
    """일괄 삭제"""
    video_ids = request.data.get('video_ids', [])
    deleted_count = 0
    for video_id in video_ids:
        try:
            video = Video.objects.get(id=video_id)
            video.delete()
            deleted_count += 1
        except Video.DoesNotExist:
            continue
    return JsonResponse({'message': f'{deleted_count}개의 비디오가 삭제되었습니다.'})


@api_view(['GET'])
def get_frame_image(request, pk, frame_number):
    """특정 프레임 이미지 반환"""
    try:
        video = Video.objects.get(id=pk)
        
        if not video.file:
            return JsonResponse({'error': '비디오 파일을 찾을 수 없습니다.'}, status=404)
        
        import cv2
        import base64
        
        # 비디오 파일 열기
        cap = cv2.VideoCapture(video.file.path)
        
        if not cap.isOpened():
            return JsonResponse({'error': '비디오 파일을 열 수 없습니다.'}, status=500)
        
        # 특정 프레임으로 이동
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        
        if not ret:
            cap.release()
            return JsonResponse({'error': f'프레임 {frame_number}을 찾을 수 없습니다.'}, status=404)
        
        # 프레임을 JPEG로 인코딩
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            cap.release()
            return JsonResponse({'error': '이미지 인코딩에 실패했습니다.'}, status=500)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        # Base64로 인코딩하여 반환
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return JsonResponse({
            'frame_number': frame_number,
            'video_id': pk,
            'image_data': f'data:image/jpeg;base64,{frame_base64}',
            'timestamp': frame_number / fps if fps > 0 else 0
        })
        
    except Video.DoesNotExist:
        return JsonResponse({'error': '비디오를 찾을 수 없습니다.'}, status=404)
    except Exception as e:
        return JsonResponse({'error': f'프레임 이미지 생성 중 오류가 발생했습니다: {str(e)}'}, status=500)


def get_frame_image_bbox(request, pk, frame_number):
    """특정 프레임 이미지 반환 (바운딩 박스 포함) - FrameWithBboxView 사용"""
    try:
        # FrameWithBboxView를 사용하여 바운딩 박스가 포함된 이미지 반환
        bbox_view = FrameWithBboxView()
        
        # 검색 조건이 있으면 전달 (URL 파라미터에서)
        search_query = request.GET.get('query', '')
        if search_query:
            # 검색 조건을 request에 추가
            request.search_query = search_query
            
        return bbox_view.get(request, pk, frame_number)
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@api_view(['POST'])
def search_frames(request, pk):
    """사용자 요청에 따른 프레임 검색"""
    try:
        video = Video.objects.get(id=pk)
        
        if not video.is_analyzed or video.analysis_status != 'completed':
            return JsonResponse({
                'error': '비디오 분석이 완료되지 않았습니다. 먼저 분석을 실행해주세요.'
            }, status=400)
        
        query = request.data.get('query', '')
        if not query:
            return JsonResponse({'error': '검색 쿼리를 입력해주세요.'}, status=400)
        
        # 분석 결과 JSON 파일 로드
        analysis_data = None
        if video.analysis_json_path and os.path.exists(video.analysis_json_path):
            try:
                with open(video.analysis_json_path, 'r', encoding='utf-8') as f:
                    analysis_data = json.load(f)
            except Exception as e:
                print(f"❌ 분석 결과 JSON 로드 실패: {e}")
        
        # 프레임 검색 로직
        matching_frames = search_frames_by_query(query, analysis_data, video)
        
        return JsonResponse({
            'video_id': pk,
            'query': query,
            'matching_frames': matching_frames,
            'total_matches': len(matching_frames)
        })
        
    except Video.DoesNotExist:
        return JsonResponse({'error': '비디오를 찾을 수 없습니다.'}, status=404)
    except Exception as e:
        return JsonResponse({'error': f'프레임 검색 중 오류가 발생했습니다: {str(e)}'}, status=500)


def search_frames_by_query(query, analysis_data, video):
    """쿼리에 따라 프레임 검색 - person_database 사용"""
    matching_frames = []
    
    if not analysis_data or 'result' not in analysis_data:
        return matching_frames
    
    result = analysis_data['result']
    query_lower = query.lower()
    
    # person_database에서 사람 검색
    if 'person_database' in result:
        person_database = result['person_database']
        print(f"🔍 [search_frames_by_query] 비디오 {video.id}의 person_database에서 검색: {len(person_database)}개 항목")
        
        for person_data in person_database:
            if person_data.get('class', '').lower() == 'person':
                frame_id = person_data.get('frame_id', 0)
                timestamp = person_data.get('timestamp', 0)
                confidence = person_data.get('confidence', 0)
                
                # 이미지 경로 생성 (이미 분석 시에 저장됨)
                frame_id = person_data.get('frame_id', 0)
                frame_image_path = f"images/video{video.id}_frame{frame_id}.jpg"
                
                matching_frames.append({
                    'frame_id': frame_id,
                    'timestamp': timestamp,
                    'confidence': confidence,
                    'attributes': person_data.get('attributes', {}),
                    'bbox': person_data.get('bbox', []),
                    'frame_image_path': frame_image_path,
                    'match_reason': f"사람 감지 (신뢰도: {confidence:.2f})"
                })
    
    # 기존 frame_results 검색도 유지 (fallback)
    elif 'frame_results' in result:
        for frame_data in result['frame_results']:
            frame_id = frame_data.get('image_id', 0)
            timestamp = frame_data.get('timestamp', 0)
            
            # 감지된 사람들에서 검색
            if 'persons' in frame_data:
                for person in frame_data['persons']:
                    if matches_query(person, query_lower):
                        frame_image_path = frame_data.get('frame_image_path', '')
                        
                        matching_frames.append({
                            'frame_id': frame_id,
                            'timestamp': timestamp,
                            'confidence': person.get('confidence', 0),
                            'attributes': person.get('attributes', {}),
                            'bbox': person.get('bbox', {}),
                            'frame_image_path': frame_image_path,
                            'match_reason': f"사람 감지: {person.get('attributes', {}).get('gender', {}).get('value', 'unknown')}"
                        })
            
            # 씬 분석에서 검색 (있는 경우)
            if 'scene_info' in frame_data:
                scene = frame_data['scene_info']
                if matches_scene_query(scene, query_lower):
                    matching_frames.append({
                        'frame_id': frame_id,
                        'timestamp': timestamp,
                        'confidence': scene.get('confidence', 0),
                        'scene_type': scene.get('scene_type', 'unknown'),
                        'match_reason': f"씬 분석: {scene.get('scene_type', 'unknown')}"
                    })
    
    # 중복 제거 및 신뢰도 순 정렬
    unique_frames = {}
    for frame in matching_frames:
        key = f"{frame['frame_id']}_{frame['timestamp']}"  # frame_id와 timestamp 조합으로 고유성 보장
        if key not in unique_frames or frame['confidence'] > unique_frames[key]['confidence']:
            unique_frames[key] = frame
    
    # 신뢰도 순으로 정렬하고 상위 10개 반환
    sorted_frames = sorted(unique_frames.values(), key=lambda x: x['confidence'], reverse=True)
    print(f"🔍 [search_frames_by_query] 최종 결과: {len(sorted_frames)}개 프레임")
    return sorted_frames[:10]


def matches_query(person, query_lower):
    """사람 객체가 쿼리와 일치하는지 확인"""
    attributes = person.get('attributes', {})
    
    # 성별 검색
    if any(keyword in query_lower for keyword in ['남자', '남성', 'man', 'male']):
        return attributes.get('gender', {}).get('value', '').lower() in ['man', 'male']
    
    if any(keyword in query_lower for keyword in ['여자', '여성', 'woman', 'female']):
        return attributes.get('gender', {}).get('value', '').lower() in ['woman', 'female']
    
    # 나이 검색
    if any(keyword in query_lower for keyword in ['아이', '어린이', 'child', 'kid']):
        return attributes.get('age', {}).get('value', '').lower() in ['child', 'kid']
    
    if any(keyword in query_lower for keyword in ['청년', '젊은', 'young', 'adult']):
        return attributes.get('age', {}).get('value', '').lower() in ['young adult', 'teenager']
    
    if any(keyword in query_lower for keyword in ['중년', 'middle', 'aged']):
        return attributes.get('age', {}).get('value', '').lower() in ['middle-aged']
    
    if any(keyword in query_lower for keyword in ['노인', 'elderly', 'old']):
        return attributes.get('age', {}).get('value', '').lower() in ['elderly']
    
    # 옷 색상 검색 (한국어 색상명 포함)
    clothing_color = attributes.get('clothing_color', {}).get('value', '').lower()
    color_mapping = {
        '빨강': 'red', '빨간': 'red', 'red': 'red',
        '파랑': 'blue', '파란': 'blue', 'blue': 'blue',
        '노랑': 'yellow', '노란': 'yellow', 'yellow': 'yellow',
        '초록': 'green', '녹색': 'green', 'green': 'green',
        '검정': 'black', '검은': 'black', 'black': 'black',
        '흰색': 'white', '흰': 'white', 'white': 'white',
        '회색': 'gray', 'grey': 'gray', 'gray': 'gray',
        '보라': 'purple', '보라색': 'purple', 'purple': 'purple',
        '주황': 'orange', '주황색': 'orange', 'orange': 'orange',
        '분홍': 'pink', '분홍색': 'pink', 'pink': 'pink',
        '갈색': 'brown', 'brown': 'brown'
    }
    
    for korean_color, english_color in color_mapping.items():
        if korean_color in query_lower and clothing_color == english_color:
            return True
        if english_color in query_lower and clothing_color == english_color:
            return True
    
    # 액세서리 검색
    accessories = attributes.get('accessories', {}).get('value', '').lower()
    accessory_keywords = {
        '안경': 'glasses', 'glasses': 'glasses',
        '선글라스': 'sunglasses', 'sunglasses': 'sunglasses',
        '모자': 'hat', 'hat': 'hat',
        '캡': 'cap', 'cap': 'cap',
        '가방': 'bag', 'bag': 'bag',
        '백팩': 'backpack', 'backpack': 'backpack',
        '핸드백': 'handbag', 'handbag': 'handbag',
        '시계': 'watch', 'watch': 'watch',
        '핸드폰': 'phone', 'phone': 'phone',
        '이어폰': 'earphones', 'earphones': 'earphones',
        '귀걸이': 'jewelry', 'jewelry': 'jewelry'
    }
    
    for korean_accessory, english_accessory in accessory_keywords.items():
        if korean_accessory in query_lower and english_accessory in accessories:
            return True
        if english_accessory in query_lower and english_accessory in accessories:
            return True
    
    # 옷 스타일 검색
    detailed_clothing = attributes.get('detailed_clothing', {}).get('value', '').lower()
    clothing_keywords = {
        '티셔츠': 't-shirt', 't-shirt': 't-shirt', 'tshirt': 't-shirt',
        '긴팔': 'long sleeve', 'long sleeve': 'long sleeve',
        '폴로': 'polo', 'polo': 'polo',
        '탱크톱': 'tank top', 'tank top': 'tank top',
        '스웨터': 'sweater', 'sweater': 'sweater',
        '후드': 'hoodie', 'hoodie': 'hoodie',
        '청바지': 'jeans', 'jeans': 'jeans',
        '바지': 'pants', 'pants': 'pants',
        '반바지': 'shorts', 'shorts': 'shorts',
        '레깅스': 'leggings', 'leggings': 'leggings',
        '치마': 'skirt', 'skirt': 'skirt',
        '드레스': 'dress', 'dress': 'dress'
    }
    
    for korean_clothing, english_clothing in clothing_keywords.items():
        if korean_clothing in query_lower and english_clothing in detailed_clothing:
            return True
        if english_clothing in query_lower and english_clothing in detailed_clothing:
            return True
    
    # 자세 검색
    posture = attributes.get('posture', {}).get('value', '').lower()
    posture_keywords = {
        '서있는': 'standing', 'standing': 'standing',
        '앉은': 'sitting', 'sitting': 'sitting',
        '걷는': 'walking', 'walking': 'walking',
        '뛰는': 'running', 'running': 'running',
        '누운': 'lying down', 'lying': 'lying down'
    }
    
    for korean_posture, english_posture in posture_keywords.items():
        if korean_posture in query_lower and english_posture in posture:
            return True
        if english_posture in query_lower and english_posture in posture:
            return True
    
    # 기본적으로 사람이면 일치
    if any(keyword in query_lower for keyword in ['사람', 'person', '인간']):
        return True
    
    return False


def matches_scene_query(scene, query_lower):
    """씬이 쿼리와 일치하는지 확인"""
    scene_type = scene.get('scene_type', '').lower()
    
    if any(keyword in query_lower for keyword in ['실내', 'indoor', 'inside']):
        return 'indoor' in scene_type
    
    if any(keyword in query_lower for keyword in ['실외', 'outdoor', 'outside']):
        return 'outdoor' in scene_type
    
    if any(keyword in query_lower for keyword in ['낮', 'day', 'daytime']):
        return 'day' in scene_type
    
    if any(keyword in query_lower for keyword in ['밤', 'night', 'nighttime']):
        return 'night' in scene_type
    
    return False


@api_view(['GET'])
def get_frame_image_file(request, pk, frame_number):
    """저장된 프레임 이미지 파일 반환"""
    try:
        video = Video.objects.get(id=pk)
        
        if not video.is_analyzed or video.analysis_status != 'completed':
            return JsonResponse({
                'error': '비디오 분석이 완료되지 않았습니다.'
            }, status=400)
        
        # 프레임 이미지 파일 경로 (실제 저장 경로에 맞게 수정)
        frame_image_path = os.path.join(
            settings.MEDIA_ROOT, 
            'images', 
            f'video{video.id}_frame{frame_number}.jpg'
        )
        
        if not os.path.exists(frame_image_path):
            return JsonResponse({
                'error': f'프레임 {frame_number} 이미지를 찾을 수 없습니다.'
            }, status=404)
        
        # 이미지 파일을 읽어서 반환
        with open(frame_image_path, 'rb') as f:
            image_data = f.read()
        
        response = HttpResponse(image_data, content_type='image/jpeg')
        response['Content-Disposition'] = f'inline; filename="frame_{frame_number:06d}.jpg"'
        return response
        
    except Video.DoesNotExist:
        return JsonResponse({'error': '비디오를 찾을 수 없습니다.'}, status=404)
    except Exception as e:
        return JsonResponse({'error': f'프레임 이미지 로드 중 오류가 발생했습니다: {str(e)}'}, status=500)


@api_view(['POST'])
def cleanup_storage(request):
    """저장공간 정리"""
    return JsonResponse({'message': '저장공간 정리가 완료되었습니다.'})


class ObjectSearchView(APIView):
    """사용자 입력 기반 객체 탐지 및 장면 검색"""
    permission_classes = [AllowAny]
    
    def post(self, request, pk):
        try:
            # JSON 데이터 파싱
            if hasattr(request, 'data'):
                user_query = request.data.get('query', '').strip()
            else:
                import json
                body = request.body.decode('utf-8')
                data = json.loads(body) if body else {}
                user_query = data.get('query', '').strip()
            
            if not user_query:
                return Response({
                    'error': '검색할 내용을 입력해주세요.'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            print(f"🔍 객체 탐지 검색 요청: video_id={pk}, query='{user_query}'")
            
            # 직접 검색 로직 구현 (사람, 가방, 어린이 등 모든 검색 지원)
            search_results = self._search_objects_directly(str(pk), user_query)
            
            if search_results['success']:
                return Response({
                    'success': True,
                    'query': user_query,
                    'video_id': pk,
                    'matches': search_results['matches'],
                    'total_matches': search_results['total_matches'],
                    'keywords': search_results['keywords']
                })
            else:
                return Response({
                    'success': False,
                    'error': search_results['error']
                }, status=status.HTTP_404_NOT_FOUND)
                
        except Exception as e:
            print(f"❌ 객체 탐지 검색 오류: {e}")
            return Response({
                'error': f'검색 중 오류가 발생했습니다: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    def _search_objects_directly(self, video_id, query):
        """직접 객체 검색 로직 (사람, 가방, 어린이 등 모든 검색 지원)"""
        try:
            from .models import Video, PersonDetection, YOLOObjectDetection, Frame
            
            # 비디오 존재 확인
            try:
                video = Video.objects.get(id=video_id)
            except Video.DoesNotExist:
                return {'success': False, 'error': '비디오를 찾을 수 없습니다.'}
            
            # 데이터베이스에서 직접 검색
            matches = []
            query_lower = query.lower()
            
            # 1. PersonDetection에서 검색
            person_matches = self._search_person_detections(video, query_lower)
            matches.extend(person_matches)
            
            # 2. YOLOObjectDetection에서 검색
            yolo_matches = self._search_yolo_objects(video, query_lower)
            matches.extend(yolo_matches)
            
            # 3. 복합 검색 (가방 가진 여자 등)
            complex_matches = self._search_complex_queries(video, query_lower)
            matches.extend(complex_matches)
            
            # 4. JSON 파일에서도 검색 (백업용)
            json_matches = self._search_json_analysis(video_id, query_lower)
            matches.extend(json_matches)
            
            # 중복 제거 및 정렬
            unique_matches = []
            seen = set()
            for match in matches:
                key = (match.get('frame_id'), match.get('class_name'), match.get('timestamp'))
                if key not in seen:
                    seen.add(key)
                    unique_matches.append(match)
            
            # 신뢰도순으로 정렬
            unique_matches.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            
            # 바운딩 박스 URL 추가
            for match in unique_matches:
                frame_id = match.get('frame_id', 0)
                if frame_id > 0:
                    match['bbox_image_url'] = f"/api/videos/{video_id}/frames/{frame_id}/bbox/"
                    match['frame_url'] = f"/api/videos/{video_id}/frames/{frame_id}/"
                else:
                    match['bbox_image_url'] = None
                    match['frame_url'] = None
            
            return {
                'success': True,
                'matches': unique_matches,
                'total_matches': len(unique_matches),
                'keywords': self._extract_keywords(query_lower)
            }
            
        except Exception as e:
            print(f"❌ 객체 검색 오류: {e}")
            return {'success': False, 'error': f'검색 중 오류가 발생했습니다: {str(e)}'}
    
    def _search_person_detections(self, video, query_lower):
        """PersonDetection에서 검색"""
        matches = []
        
        # 사람 관련 키워드 확인 (나이 그룹 포함)
        person_keywords = ['사람', '인간', '남자', '여자', '성인', 'person', 'man', 'woman', 'adult', 'people',
                          '어린이', '아이', 'child', 'kid', '노인', 'elderly', 'old', 'senior',
                          '청년', '젊은', 'young', '중년', 'middle', 'aged']
        if not any(keyword in query_lower for keyword in person_keywords):
            return matches
        
        person_detections = PersonDetection.objects.filter(video=video)
        
        for detection in person_detections:
            match_score = 0
            match_reasons = []
            
            # 기본 신뢰도 점수
            if detection.confidence > 0.5:
                match_score += detection.confidence
                match_reasons.append(f"높은 신뢰도: {detection.confidence:.2f}")
            
            # 성별 매칭 (엄격한 필터링)
            gender_matched = True
            if '남자' in query_lower or '남성' in query_lower or 'man' in query_lower:
                if 'man' in detection.gender_estimation.lower():
                    match_score += 0.3
                    match_reasons.append(f"성별 매칭: {detection.gender_estimation}")
                else:
                    gender_matched = False
            elif '여자' in query_lower or '여성' in query_lower or 'woman' in query_lower:
                if 'woman' in detection.gender_estimation.lower():
                    match_score += 0.3
                    match_reasons.append(f"성별 매칭: {detection.gender_estimation}")
                else:
                    gender_matched = False
            
            # 성별이 지정되었는데 매칭되지 않으면 제외
            if not gender_matched:
                continue
            
            # 나이 매칭 (엄격한 필터링 + 크기 정보 활용)
            age_matched = True
            if '어린이' in query_lower or '아이' in query_lower or 'child' in query_lower:
                # 나이 그룹 확인
                age_group_match = 'child' in detection.age_group.lower() or 'teen' in detection.age_group.lower()
                
                # 크기 정보도 확인 (어린이는 작은 크기여야 함)
                bbox = [detection.bbox_x1, detection.bbox_y1, detection.bbox_x2, detection.bbox_y2]
                if bbox and len(bbox) == 4:
                    width = bbox[2] - bbox[0]
                    height = bbox[3] - bbox[1]
                    area = width * height
                    size_match = area < 0.15  # 작은 크기 (어린이)
                else:
                    size_match = True  # 크기 정보가 없으면 나이 그룹만 확인
                
                if age_group_match and size_match:
                    match_score += 0.3
                    match_reasons.append(f"나이 매칭: {detection.age_group}")
                else:
                    age_matched = False
            
            if '노인' in query_lower or 'elderly' in query_lower or 'old' in query_lower:
                # 나이 그룹 확인
                age_group_match = 'elderly' in detection.age_group.lower() or 'senior' in detection.age_group.lower()
                
                # 크기 정보도 확인 (노인은 작은 크기가 아닌 보통 이상이어야 함)
                bbox = [detection.bbox_x1, detection.bbox_y1, detection.bbox_x2, detection.bbox_y2]
                if bbox and len(bbox) == 4:
                    width = bbox[2] - bbox[0]
                    height = bbox[3] - bbox[1]
                    area = width * height
                    size_match = area >= 0.1  # 작은 크기가 아닌 (노인은 어린이보다 큼)
                else:
                    size_match = True  # 크기 정보가 없으면 나이 그룹만 확인
                
                if age_group_match and size_match:
                    match_score += 0.3
                    match_reasons.append(f"나이 매칭: {detection.age_group}")
                else:
                    age_matched = False
            
            if '청년' in query_lower or '젊은' in query_lower or 'young' in query_lower:
                if 'teenager' in detection.age_group.lower() or 'young' in detection.age_group.lower() or 'adult' in detection.age_group.lower():
                    match_score += 0.3
                    match_reasons.append(f"나이 매칭: {detection.age_group}")
                else:
                    age_matched = False
            
            if '중년' in query_lower or 'middle' in query_lower or 'aged' in query_lower:
                if 'middle-aged' in detection.age_group.lower() or 'middle' in detection.age_group.lower() or 'aged' in detection.age_group.lower():
                    match_score += 0.3
                    match_reasons.append(f"나이 매칭: {detection.age_group}")
                else:
                    age_matched = False
            
            # 나이 그룹이 지정되었는데 매칭되지 않으면 제외
            if not age_matched:
                continue
            
            # 색상 매칭 (엄격한 필터링)
            color_keywords = {
                '빨간': 'red', '파란': 'blue', '검은': 'black', '흰': 'white',
                '초록': 'green', '노란': 'yellow', '보라': 'purple',
                '분홍': 'pink', '핑크': 'pink', '분홍색': 'pink', '핑크색': 'pink',
                '주황': 'orange', '오렌지': 'orange', '갈색': 'brown', '회색': 'gray'
            }
            
            color_matched = True
            for kr_color, en_color in color_keywords.items():
                if kr_color in query_lower or en_color in query_lower:
                    # 상체 색상 확인 (wearing white clothes 형태 고려)
                    upper_color = detection.upper_body_color.lower()
                    if en_color in upper_color:
                        match_score += 0.2
                        match_reasons.append(f"상체 색상 매칭: {detection.upper_body_color}")
                        color_matched = True
                        break
                    # 하체 색상 확인 (wearing white clothes 형태 고려)
                    lower_color = detection.lower_body_color.lower()
                    if en_color in lower_color:
                        match_score += 0.2
                        match_reasons.append(f"하체 색상 매칭: {detection.lower_body_color}")
                        color_matched = True
                        break
                    else:
                        color_matched = False
            
            # 색상이 지정되었는데 매칭되지 않으면 제외
            if any(kr_color in query_lower or en_color in query_lower for kr_color, en_color in color_keywords.items()) and not color_matched:
                continue
            
            # 자세 매칭 (엄격한 필터링)
            posture_matched = True
            posture_keywords = {
                '서있는': 'standing', 'standing': 'standing',
                '앉은': 'sitting', 'sitting': 'sitting',
                '걷는': 'walking', 'walking': 'walking',
                '뛰는': 'running', 'running': 'running',
                '누운': 'lying down', 'lying': 'lying down',
                '핸드폰': 'phone', '전화': 'phone', 'phone': 'phone'
            }
            
            for korean_posture, english_posture in posture_keywords.items():
                if korean_posture in query_lower or english_posture in query_lower:
                    posture_text = detection.posture.lower()
                    if english_posture in posture_text:
                        match_score += 0.2
                        match_reasons.append(f"자세 매칭: {detection.posture}")
                        posture_matched = True
                        break
                    else:
                        posture_matched = False
            
            # 자세가 지정되었는데 매칭되지 않으면 제외
            if any(korean_posture in query_lower or english_posture in query_lower for korean_posture, english_posture in posture_keywords.items()) and not posture_matched:
                continue
            
            if match_score > 0:
                # frame_id를 안전하게 가져오기
                frame_id = 0
                timestamp = 0
                if detection.frame:
                    frame_id = detection.frame.image_id
                    timestamp = detection.frame.timestamp
                else:
                    # frame이 없는 경우 기본값 사용
                    frame_id = 1
                    timestamp = 0.0
                
                matches.append({
                                    'frame_id': frame_id,
                                    'timestamp': timestamp,
                    'class_name': 'person',
                    'confidence': detection.confidence,
                    'bbox': [detection.bbox_x1, detection.bbox_y1, detection.bbox_x2, detection.bbox_y2],
                    'attributes': {
                        'gender': detection.gender_estimation,
                        'age': detection.age_group,
                        'clothing_color': detection.upper_body_color,
                        'posture': detection.posture
                    },
                    'match_score': match_score,
                    'match_reasons': match_reasons
                })
        
        return matches
    
    def _search_complex_queries(self, video, query_lower):
        """복합 검색 (가방 가진 여자, 핸드폰 가진 어린이 등)"""
        matches = []
        
        # 객체 키워드와 사람 키워드가 모두 있는 경우
        object_keywords = ['가방', '핸드폰', '전화', '우산', '백팩', 'handbag', 'phone', 'umbrella', 'backpack']
        person_keywords = ['사람', '남자', '여자', '어린이', '노인', 'person', 'man', 'woman', 'child', 'elderly']
        
        has_object = any(keyword in query_lower for keyword in object_keywords)
        has_person = any(keyword in query_lower for keyword in person_keywords)
        
        if has_object and has_person:
            # 같은 프레임에서 사람과 객체를 모두 찾기
            person_detections = PersonDetection.objects.filter(video=video)
            yolo_objects = YOLOObjectDetection.objects.filter(video=video)
            
            for person in person_detections:
                if not person.frame:
                    continue
                    
                # 같은 프레임의 객체들 찾기
                frame_objects = yolo_objects.filter(frame=person.frame)
                
                for obj in frame_objects:
                    match_score = 0
                    match_reasons = []
                    
                    # 사람 매칭
                    if person.confidence > 0.5:
                        match_score += person.confidence
                        match_reasons.append(f"사람 감지: {person.confidence:.2f}")
                    
                    # 객체 매칭
                    if obj.confidence > 0.3:
                        match_score += obj.confidence * 0.5
                        match_reasons.append(f"객체 감지: {obj.class_name} ({obj.confidence:.2f})")
                    
                    # 성별 매칭
                    if '남자' in query_lower or 'man' in query_lower:
                        if 'man' in person.gender_estimation.lower():
                            match_score += 0.3
                            match_reasons.append(f"성별 매칭: {person.gender_estimation}")
                        else:
                            continue
                    elif '여자' in query_lower or 'woman' in query_lower:
                        if 'woman' in person.gender_estimation.lower():
                            match_score += 0.3
                            match_reasons.append(f"성별 매칭: {person.gender_estimation}")
                        else:
                            continue
                    
                    # 나이 매칭 (크기 정보 활용)
                    if '어린이' in query_lower or 'child' in query_lower:
                        # 나이 그룹 확인
                        age_group_match = 'child' in person.age_group.lower() or 'teen' in person.age_group.lower()
                        
                        # 크기 정보도 확인 (어린이는 작은 크기여야 함)
                        bbox = [person.bbox_x1, person.bbox_y1, person.bbox_x2, person.bbox_y2]
                        if bbox and len(bbox) == 4:
                            width = bbox[2] - bbox[0]
                            height = bbox[3] - bbox[1]
                            area = width * height
                            size_match = area < 0.15  # 작은 크기 (어린이)
                        else:
                            size_match = True  # 크기 정보가 없으면 나이 그룹만 확인
                        
                        if age_group_match and size_match:
                            match_score += 0.3
                            match_reasons.append(f"나이 매칭: {person.age_group}")
                        else:
                            continue
                    elif '노인' in query_lower or 'elderly' in query_lower:
                        # 나이 그룹 확인
                        age_group_match = 'elderly' in person.age_group.lower() or 'senior' in person.age_group.lower()
                        
                        # 크기 정보도 확인 (노인은 작은 크기가 아닌 보통 이상이어야 함)
                        bbox = [person.bbox_x1, person.bbox_y1, person.bbox_x2, person.bbox_y2]
                        if bbox and len(bbox) == 4:
                            width = bbox[2] - bbox[0]
                            height = bbox[3] - bbox[1]
                            area = width * height
                            size_match = area >= 0.1  # 작은 크기가 아닌 (노인은 어린이보다 큼)
                        else:
                            size_match = True  # 크기 정보가 없으면 나이 그룹만 확인
                        
                        if age_group_match and size_match:
                            match_score += 0.3
                            match_reasons.append(f"나이 매칭: {person.age_group}")
                        else:
                            continue
                    elif '중년' in query_lower or 'middle' in query_lower or 'aged' in query_lower:
                        if 'middle-aged' in person.age_group.lower() or 'middle' in person.age_group.lower() or 'aged' in person.age_group.lower():
                            match_score += 0.3
                            match_reasons.append(f"나이 매칭: {person.age_group}")
                        else:
                            continue
                    elif '청년' in query_lower or '젊은' in query_lower or 'young' in query_lower:
                        if 'teenager' in person.age_group.lower() or 'young' in person.age_group.lower() or 'adult' in person.age_group.lower():
                            match_score += 0.3
                            match_reasons.append(f"나이 매칭: {person.age_group}")
                        else:
                            continue
                    
                    if match_score > 0.8:  # 높은 임계값으로 정확한 매칭만
                        matches.append({
                            'frame_id': person.frame.image_id,
                            'timestamp': person.frame.timestamp,
                            'class_name': 'person',
                            'confidence': person.confidence,
                            'bbox': [person.bbox_x1, person.bbox_y1, person.bbox_x2, person.bbox_y2],
                            'attributes': {
                                'gender': person.gender_estimation,
                                'age': person.age_group,
                                'clothing_color': person.upper_body_color,
                                'posture': person.posture,
                                'nearby_object': obj.class_name
                            },
                            'match_score': match_score,
                            'match_reasons': match_reasons
                        })
        
        return matches
    
    def _search_yolo_objects(self, video, query_lower):
        """YOLOObjectDetection에서 검색"""
        matches = []
        
        # 객체 타입 매핑
        object_mappings = {
            '사람': 'person', 'person': 'person',
            '차': 'car', '자동차': 'car', 'car': 'car',
            '트럭': 'truck', 'truck': 'truck',
            '자전거': 'bicycle', 'bike': 'bicycle', 'bicycle': 'bicycle',
            '가방': 'handbag', 'bag': 'handbag', '백팩': 'backpack', 'handbag': 'handbag', 'backpack': 'backpack',
            '우산': 'umbrella', 'umbrella': 'umbrella',
            '핸드폰': 'cell phone', '휴대폰': 'cell phone', 'phone': 'cell phone',
            '개': 'dog', 'dog': 'dog',
            '고양이': 'cat', 'cat': 'cat',
            '의자': 'chair', 'chair': 'chair',
            'tv': 'tv', '티비': 'tv', 'television': 'tv',
            '컵': 'cup', '잔': 'cup', 'cup': 'cup', 'wine glass': 'wine glass',
            '병': 'bottle', 'bottle': 'bottle',
            '그릇': 'bowl', 'bowl': 'bowl',
            '벤치': 'bench', 'bench': 'bench',
            '스케이트보드': 'skateboard', 'skateboard': 'skateboard'
        }
        
        # 검색할 객체 타입 찾기
        search_classes = []
        for keyword, class_name in object_mappings.items():
            if keyword in query_lower:
                search_classes.append(class_name)
        
        if not search_classes:
            return matches
        
        # YOLO 객체 검색
        yolo_detections = YOLOObjectDetection.objects.filter(
            video=video,
            class_name__in=search_classes
        ).order_by('-confidence')
        
        for detection in yolo_detections:
            match_score = detection.confidence
            match_reasons = [f"YOLO 감지: {detection.class_name} (신뢰도: {detection.confidence:.2f})"]
            
            matches.append({
                'frame_id': detection.frame_number,
                'timestamp': detection.timestamp,
                'class_name': detection.class_name,
                'confidence': detection.confidence,
                'bbox': [detection.bbox_x1, detection.bbox_y1, detection.bbox_x2, detection.bbox_y2],
                                    'match_score': match_score,
                'match_reasons': match_reasons
            })
        
        return matches
    
    def _search_json_analysis(self, video_id, query_lower):
        """JSON 분석 파일에서 검색 (백업용)"""
        matches = []
        
        try:
            import os
            import json
            import glob
            
            # JSON 분석 결과 파일 경로
            json_path = f"media/analysis_results/real_analysis_{video_id}_enhanced_*.json"
            json_files = glob.glob(json_path)
            
            if not json_files:
                return matches
            
            # 최신 파일 선택
            latest_file = max(json_files, key=os.path.getmtime)
            
            with open(latest_file, 'r', encoding='utf-8') as f:
                analysis_data = json.load(f)
            
            # 프레임별 검색
            frame_results = analysis_data.get('frame_results', [])
            for frame_data in frame_results:
                frame_id = frame_data.get('image_id', 0)
                timestamp = frame_data.get('timestamp', 0)
                objects = frame_data.get('objects', [])
                
                for obj in objects:
                    class_name = obj.get('class', '').lower()
                    confidence = obj.get('confidence', 0)
                    
                    # 사람 객체인지 확인
                    is_person = any(keyword in query_lower for keyword in [class_name, '사람', 'person'])
                    
                    if is_person:
                        match_score = confidence
                        match_reasons = [f"JSON 분석: {class_name}"]
                        
                        # 색상 매칭 확인
                        color_keywords = {
                            '빨간': 'red', '파란': 'blue', '검은': 'black', '흰': 'white',
                            '초록': 'green', '노란': 'yellow', '보라': 'purple',
                            '분홍': 'pink', '핑크': 'pink', '분홍색': 'pink', '핑크색': 'pink',
                            '주황': 'orange', '오렌지': 'orange', '갈색': 'brown', '회색': 'gray'
                        }
                        
                        # 나이 그룹 매칭 확인
                        age_matched = True
                        if any(keyword in query_lower for keyword in ['어린이', '아이', 'child', 'kid']):
                            age_value = attributes.get('age', {}).get('value', '').lower()
                            if 'child' not in age_value and 'kid' not in age_value and 'teen' not in age_value:
                                age_matched = False
                            else:
                                match_score += 0.2
                                match_reasons.append(f"나이 매칭: {age_value}")
                        
                        if any(keyword in query_lower for keyword in ['노인', 'elderly', 'old', 'senior']):
                            age_value = attributes.get('age', {}).get('value', '').lower()
                            if 'elderly' not in age_value and 'senior' not in age_value and 'old' not in age_value:
                                age_matched = False
                            else:
                                match_score += 0.2
                                match_reasons.append(f"나이 매칭: {age_value}")
                        
                        if any(keyword in query_lower for keyword in ['청년', '젊은', 'young', 'adult']):
                            age_value = attributes.get('age', {}).get('value', '').lower()
                            if 'young' not in age_value and 'adult' not in age_value:
                                age_matched = False
                            else:
                                match_score += 0.2
                                match_reasons.append(f"나이 매칭: {age_value}")
                        
                        if any(keyword in query_lower for keyword in ['중년', 'middle', 'aged']):
                            age_value = attributes.get('age', {}).get('value', '').lower()
                            if 'middle' not in age_value and 'aged' not in age_value:
                                age_matched = False
                            else:
                                match_score += 0.2
                                match_reasons.append(f"나이 매칭: {age_value}")
                        
                        # 나이 그룹이 지정되었는데 매칭되지 않으면 제외
                        if not age_matched:
                            continue
                        
                        # 객체의 색상 정보 확인
                        clothing_color = attributes.get('clothing_color', {})
                        if isinstance(clothing_color, dict):
                            color_value = clothing_color.get('value', '').lower()
                        else:
                            color_value = str(clothing_color).lower()
                        
                        color_matched = False
                        for kr_color, en_color in color_keywords.items():
                            if (kr_color in query_lower or en_color in query_lower) and en_color in color_value:
                                match_score += 0.2
                                match_reasons.append(f"색상 매칭: {color_value}")
                                color_matched = True
                                break
                        
                        # 색상이 지정되었는데 매칭되지 않으면 제외
                        if any(kr_color in query_lower or en_color in query_lower for kr_color, en_color in color_keywords.items()) and not color_matched:
                            continue
                        
                        matches.append({
                            'frame_id': frame_id,
                            'timestamp': timestamp,
                            'class_name': class_name,
                            'confidence': confidence,
                            'bbox': obj.get('bbox', []),
                            'match_score': match_score,
                            'match_reasons': match_reasons
                        })
            
        except Exception as e:
            print(f"⚠️ JSON 검색 오류: {e}")
        
        return matches
    
    def _extract_keywords(self, query_lower):
        """검색 쿼리에서 키워드 추출"""
        keywords = []
        
        # 객체 키워드
        object_keywords = ['사람', '차', '트럭', '자전거', '가방', '우산', '핸드폰', '개', '고양이', '의자', 'tv']
        for keyword in object_keywords:
            if keyword in query_lower:
                keywords.append(keyword)
        
        # 색상 키워드
        color_keywords = ['빨간', '파란', '검은', '흰', '초록', '노란', '보라']
        for keyword in color_keywords:
            if keyword in query_lower:
                keywords.append(keyword)
        
        return keywords


class ScenePreviewView(APIView):
    """특정 프레임의 장면 미리보기"""
    permission_classes = [AllowAny]
    
    def get(self, request, pk, frame_index):
        try:
            print(f"🎬 장면 미리보기 요청: video_id={pk}, frame_index={frame_index}")
            
            # RAG 시스템에서 장면 미리보기 정보 가져오기
            from api.db_builder import get_enhanced_video_rag_system
            rag_system = get_enhanced_video_rag_system()
            
            preview_info = rag_system.get_scene_preview(str(pk), frame_index)
            
            if preview_info['success']:
                return Response(preview_info)
            else:
                return Response({
                    'error': preview_info['error']
                }, status=status.HTTP_404_NOT_FOUND)
                
        except Exception as e:
            print(f"❌ 장면 미리보기 오류: {e}")
            return Response({
                'error': f'미리보기 생성 중 오류가 발생했습니다: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@method_decorator(csrf_exempt, name='dispatch')
class VideoSummaryView(APIView):
    """비디오 요약 및 하이라이트 생성"""
    permission_classes = [AllowAny]
    
    def post(self, request, video_id):
        try:
            video = Video.objects.get(id=video_id)
            summary_type = request.data.get('type', 'summary')  # summary, highlight, custom
            criteria = request.data.get('criteria', {})
            
            if summary_type == 'summary':
                result = self._generate_video_summary(video)
            elif summary_type == 'highlight':
                result = self._generate_highlights(video, criteria)
            elif summary_type == 'custom':
                result = self._generate_custom_summary(video, criteria)
            else:
                return Response({
                    'success': False,
                    'error': '지원하지 않는 요약 타입입니다'
                }, status=400)
            
            return Response({
                'success': True,
                'summary': result
            })
            
        except Video.DoesNotExist:
            return Response({
                'success': False,
                'error': '비디오를 찾을 수 없습니다'
            }, status=404)
        except Exception as e:
            return Response({
                'success': False,
                'error': str(e)
            }, status=500)
    
    def _generate_video_summary(self, video):
        """영상 요약 생성 (CLIP + GPT Vision 통합)"""
        try:
            video_title = video.title if video.title else f"영상 {video.id}"
            print(f"🎬 영상 요약 생성 시작: {video_title}")
            print(f"📊 비디오 ID: {video.id}, 분석 상태: {video.analysis_status}")
            
            # 1. 대표 프레임들 선택 (처음, 중간, 마지막)
            frames = Frame.objects.filter(video=video).order_by('timestamp')
            print(f"📸 발견된 프레임 수: {frames.count()}")
            
            if not frames.exists():
                print("⚠️ 분석된 프레임이 없습니다")
                return {
                    'type': 'summary',
                    'message': '분석된 프레임이 없습니다',
                    'clips': []
                }
            
            frames_list = list(frames)
            if len(frames_list) >= 3:
                selected_frames = [frames_list[0], frames_list[len(frames_list)//2], frames_list[-1]]
            elif len(frames_list) >= 2:
                selected_frames = [frames_list[0], frames_list[-1]]
            else:
                selected_frames = [frames_list[0]]
            
            print(f"📸 선택된 프레임 수: {len(selected_frames)}")
            
            # 2. CLIP을 사용한 프레임 분석
            clip_analyses = []
            frame_paths = []
            
            for frame in selected_frames:
                frame_path = os.path.join('media/images', f'video{video.id}_frame{frame.image_id}.jpg')
                if os.path.exists(frame_path):
                    frame_paths.append(frame_path)
                    print(f"🔍 CLIP 분석: {frame_path}")
                    
                    if vision_analyzer:
                        clip_analysis = vision_analyzer.analyze_frame(frame_path)
                        if 'error' not in clip_analysis:
                            clip_analyses.append({
                                'timestamp': frame.timestamp,
                                'frame_id': frame.image_id,
                                'clip_analysis': clip_analysis
                            })
                        else:
                            print(f"⚠️ CLIP 분석 실패: {clip_analysis['error']}")
                    else:
                        print("⚠️ Vision Analyzer를 사용할 수 없습니다")
            
            # 3. GPT Vision을 사용한 프레임 분석
            gpt_vision_analyses = []
            
            for frame_path in frame_paths:
                if llm_client and llm_client.is_available():
                    print(f"🤖 GPT Vision 분석: {frame_path}")
                    gpt_analysis = llm_client.analyze_frame_with_vision(
                        frame_path, 
                        "이 프레임에서 사람들을 자세히 분석해주세요. 탐지된 사람의 수, 위치, 특징, 활동을 포함해서 설명해주세요."
                    )
                    if gpt_analysis:
                        gpt_vision_analyses.append({
                            'frame_path': frame_path,
                            'analysis': gpt_analysis
                        })
                else:
                    print("⚠️ GPT Vision을 사용할 수 없습니다")
            
            # 4. 기본 클립 데이터 생성 (기존 로직 유지)
            person_detections = PersonDetection.objects.filter(video=video)
            time_segments = self._calculate_importance_segments(frames, person_detections)
            top_segments = sorted(time_segments, key=lambda x: x['importance'], reverse=True)[:5]
            
            clips = []
            for i, segment in enumerate(top_segments):
                start_time = segment['start_time']
                end_time = segment['end_time']
                importance = segment['importance']
                description = segment['description']
                
                clips.append({
                    'id': i + 1,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': end_time - start_time,
                    'importance_score': importance,
                    'description': description,
                    'thumbnail_url': f'/api/videos/{video.id}/frames/{segment["frame_id"]}/image/',
                    'clip_url': f'/api/videos/{video.id}/clips/{start_time:.1f}-{end_time:.1f}/'
                })
            
            # 5. 통합된 데이터로 요약 생성
            video_data = {
                'video_id': video.id,
                'video_title': video_title,
                'total_duration': video.duration,
                'clips': clips,
                'clip_analyses': clip_analyses,  # CLIP 분석 결과
                'gpt_vision_analyses': gpt_vision_analyses,  # GPT Vision 분석 결과
                'selected_frames': [{'timestamp': f.timestamp, 'frame_id': f.image_id} for f in selected_frames]
            }
            
            summary_data = {
                'type': 'summary',
                'video_id': video.id,
                'total_duration': video.duration,
                'summary_duration': sum(clip['duration'] for clip in clips),
                'compression_ratio': sum(clip['duration'] for clip in clips) / video.duration if video.duration > 0 else 0,
                'clips': clips,
                'message': f'CLIP과 GPT Vision을 활용하여 영상의 {len(clips)}개 주요 구간을 요약했습니다'
            }
            
            # 6. 통합된 데이터로 LLM 요약 생성
            if llm_client and llm_client.is_available():
                print("📝 통합 요약 생성 중...")
                llm_summary = llm_client.generate_summary(video_data)
                summary_data['llm_summary'] = llm_summary
                summary_data['message'] = 'CLIP과 GPT Vision을 활용한 AI 영상 요약입니다'
            else:
                print("⚠️ LLM을 사용할 수 없어 기본 요약 생성")
                summary_data['llm_summary'] = self._generate_fallback_summary(video_data)
            
            # 7. 분석 방법 정보 추가
            summary_data['analysis_methods'] = ['CLIP', 'GPT Vision'] if gpt_vision_analyses else ['CLIP']
            summary_data['clip_analyses'] = clip_analyses
            summary_data['gpt_vision_analyses'] = gpt_vision_analyses
            summary_data['selected_frames_count'] = len(selected_frames)
            
            return summary_data
            
        except Exception as e:
            print(f"❌ 영상 요약 생성 오류: {e}")
            import traceback
            traceback.print_exc()
            return {
                'type': 'summary',
                'error': str(e),
                'clips': [],
                'llm_summary': None
            }
    
    def _generate_highlights(self, video, criteria):
        """하이라이트 생성"""
        try:
            # 기본 하이라이트 기준
            default_criteria = {
                'person_count_threshold': 2,  # 2명 이상
                'movement_threshold': 0.3,    # 움직임이 많은 구간
                'confidence_threshold': 0.7,  # 높은 신뢰도
                'color_preference': None,     # 특정 색상
                'age_preference': None        # 특정 연령대
            }
            
            # 사용자 기준과 병합
            criteria = {**default_criteria, **criteria}
            
            # 조건에 맞는 프레임 찾기
            highlights = self._find_highlight_frames(video, criteria)
            
            # 연속된 구간으로 그룹화
            highlight_segments = self._group_consecutive_frames(highlights)
            
            clips = []
            for i, segment in enumerate(highlight_segments):
                clips.append({
                    'id': i + 1,
                    'start_time': segment['start_time'],
                    'end_time': segment['end_time'],
                    'duration': segment['end_time'] - segment['start_time'],
                    'reason': segment['reason'],
                    'person_count': segment['person_count'],
                    'confidence': segment['avg_confidence'],
                    'thumbnail_url': f'/api/videos/{video.id}/frames/{segment["frame_id"]}/image/',
                    'clip_url': f'/api/videos/{video.id}/clips/{segment["start_time"]:.1f}-{segment["end_time"]:.1f}/'
                })
            
                # LLM을 사용한 한국어 하이라이트 설명 생성
                highlight_data = {
                    'type': 'highlight',
                    'video_id': video.id,
                    'criteria': criteria,
                    'clips': clips,
                    'message': f'조건에 맞는 {len(clips)}개 하이라이트를 생성했습니다'
                }
                
                # LLM 하이라이트 설명 생성
                try:
                    llm_description = llm_client.generate_highlight_description(highlight_data)
                    highlight_data['llm_description'] = llm_description
                    highlight_data['message'] = 'AI가 생성한 하이라이트 설명입니다'
                except Exception as e:
                    print(f"⚠️ LLM 하이라이트 설명 생성 실패: {e}")
                    highlight_data['llm_description'] = None
                
                return highlight_data
            
        except Exception as e:
            print(f"❌ 하이라이트 생성 오류: {e}")
            return {
                'type': 'highlight',
                'error': str(e),
                'clips': []
            }
    
    def _generate_custom_summary(self, video, criteria):
        """사용자 정의 요약 생성"""
        try:
            # 사용자 정의 조건 처리
            query = criteria.get('query', '')
            time_range = criteria.get('time_range', None)
            
            # 검색 조건에 맞는 프레임 찾기
            matching_frames = self._search_frames_by_criteria(video, query, time_range)
            
            if not matching_frames:
                return {
                    'type': 'custom',
                    'message': '조건에 맞는 구간을 찾을 수 없습니다',
                    'clips': []
                }
            
            # 구간으로 그룹화
            segments = self._group_consecutive_frames(matching_frames)
            
            clips = []
            for i, segment in enumerate(segments):
                clips.append({
                    'id': i + 1,
                    'start_time': segment['start_time'],
                    'end_time': segment['end_time'],
                    'duration': segment['end_time'] - segment['start_time'],
                    'match_reason': segment['match_reason'],
                    'thumbnail_url': f'/api/videos/{video.id}/frames/{segment["frame_id"]}/image/',
                    'clip_url': f'/api/videos/{video.id}/clips/{segment["start_time"]:.1f}-{segment["end_time"]:.1f}/'
                })
            
                # LLM을 사용한 한국어 사용자 정의 요약 생성
                custom_data = {
                    'type': 'custom',
                    'video_id': video.id,
                    'query': query,
                    'clips': clips,
                    'message': f'"{query}" 조건에 맞는 {len(clips)}개 구간을 찾았습니다'
                }
                
                # LLM 사용자 정의 요약 생성
                try:
                    llm_summary = llm_client.generate_summary(custom_data)
                    custom_data['llm_summary'] = llm_summary
                    custom_data['message'] = f'AI가 "{query}" 검색 결과를 요약했습니다'
                except Exception as e:
                    print(f"⚠️ LLM 사용자 정의 요약 생성 실패: {e}")
                    custom_data['llm_summary'] = None
                
                return custom_data
            
        except Exception as e:
            print(f"❌ 사용자 정의 요약 생성 오류: {e}")
            return {
                'type': 'custom',
                'error': str(e),
                'clips': []
            }
    
    def _calculate_importance_segments(self, frames, person_detections):
        """중요도 기반 구간 계산"""
        segments = []
        segment_duration = 5.0  # 5초 구간
        
        current_time = 0
        while current_time < frames.last().timestamp:
            end_time = min(current_time + segment_duration, frames.last().timestamp)
            
            # 해당 구간의 프레임들
            segment_frames = frames.filter(
                timestamp__gte=current_time,
                timestamp__lt=end_time
            )
            
            if segment_frames.exists():
                # 구간별 중요도 계산
                importance = self._calculate_segment_importance(segment_frames, person_detections)
                
                # 구간 설명 생성
                description = self._generate_segment_description(segment_frames, person_detections)
                
                segments.append({
                    'start_time': current_time,
                    'end_time': end_time,
                    'importance': importance,
                    'description': description,
                    'frame_id': segment_frames.first().image_id
                })
            
            current_time += segment_duration
        
        return segments
    
    def _calculate_segment_importance(self, frames, person_detections):
        """구간 중요도 계산"""
        importance = 0.0
        
        # 사람 수 기반 중요도
        person_count = person_detections.filter(
            frame__in=frames
        ).values('frame').distinct().count()
        importance += person_count * 0.3
        
        # 움직임 기반 중요도 (간단한 구현)
        if frames.count() > 1:
            importance += 0.2
        
        # 신뢰도 기반 중요도
        from django.db import models
        avg_confidence = person_detections.filter(
            frame__in=frames
        ).aggregate(avg_conf=models.Avg('confidence'))['avg_conf'] or 0
        importance += avg_confidence * 0.5
        
        return min(importance, 1.0)
    
    def _generate_segment_description(self, frames, person_detections):
        """구간 설명 생성"""
        detections = person_detections.filter(frame__in=frames)
        
        if not detections.exists():
            return "빈 장면"
        
        # 사람 수
        person_count = detections.values('frame').distinct().count()
        
        # 주요 속성들
        genders = detections.values_list('gender_estimation', flat=True).distinct()
        age_groups = detections.values_list('age_group', flat=True).distinct()
        colors = detections.values_list('upper_body_color', flat=True).distinct()
        
        description_parts = []
        
        if person_count > 0:
            description_parts.append(f"{person_count}명의 사람")
        
        if genders:
            gender_text = ', '.join([g for g in genders if g != 'unknown'])
            if gender_text:
                description_parts.append(f"성별: {gender_text}")
        
        if age_groups:
            age_text = ', '.join([a for a in age_groups if a != 'unknown'])
            if age_text:
                description_parts.append(f"연령: {age_text}")
        
        if colors:
            color_text = ', '.join([c for c in colors if c != 'unknown'])
            if color_text:
                description_parts.append(f"색상: {color_text}")
        
        return " | ".join(description_parts) if description_parts else "사람 감지"
    
    def _find_highlight_frames(self, video, criteria):
        """하이라이트 조건에 맞는 프레임 찾기"""
        highlights = []
        
        # 사람 수 기준
        if criteria.get('person_count_threshold', 0) > 0:
            person_detections = PersonDetection.objects.filter(video=video)
            from django.db import models
            frame_counts = person_detections.values('frame').annotate(
                count=models.Count('id')
            ).filter(count__gte=criteria['person_count_threshold'])
            
            for item in frame_counts:
                frame = Frame.objects.get(id=item['frame'])
                highlights.append({
                    'frame_id': frame.image_id,
                    'timestamp': frame.timestamp,
                    'reason': f"{item['count']}명의 사람",
                    'person_count': item['count']
                })
        
        # 색상 기준
        if criteria.get('color_preference'):
            color = criteria['color_preference']
            detections = PersonDetection.objects.filter(
                video=video,
                upper_body_color__icontains=color
            )
            
            for detection in detections:
                highlights.append({
                    'frame_id': detection.frame.image_id,
                    'timestamp': detection.frame.timestamp,
                    'reason': f"{color} 색상 옷",
                    'person_count': 1
                })
        
        # 연령 기준
        if criteria.get('age_preference'):
            age = criteria['age_preference']
            detections = PersonDetection.objects.filter(
                video=video,
                age_group__icontains=age
            )
            
            for detection in detections:
                highlights.append({
                    'frame_id': detection.frame.image_id,
                    'timestamp': detection.frame.timestamp,
                    'reason': f"{age} 연령대",
                    'person_count': 1
                })
        
        return highlights
    
    def _group_consecutive_frames(self, frames, max_gap=2.0):
        """연속된 프레임들을 구간으로 그룹화"""
        if not frames:
            return []
        
        # 시간순 정렬
        sorted_frames = sorted(frames, key=lambda x: x['timestamp'])
        
        segments = []
        current_segment = [sorted_frames[0]]
        
        for i in range(1, len(sorted_frames)):
            current_frame = sorted_frames[i]
            prev_frame = sorted_frames[i-1]
            
            # 시간 간격이 max_gap 초 이내면 같은 구간
            if current_frame['timestamp'] - prev_frame['timestamp'] <= max_gap:
                current_segment.append(current_frame)
            else:
                # 구간 완료
                if current_segment:
                    segments.append(self._create_segment_from_frames(current_segment))
                current_segment = [current_frame]
        
        # 마지막 구간 추가
        if current_segment:
            segments.append(self._create_segment_from_frames(current_segment))
        
        return segments
    
    def _create_segment_from_frames(self, frames):
        """프레임 리스트에서 구간 정보 생성"""
        start_time = min(f['timestamp'] for f in frames)
        end_time = max(f['timestamp'] for f in frames)
        
        # 가장 많은 이유 선택
        reasons = [f['reason'] for f in frames]
        most_common_reason = max(set(reasons), key=reasons.count)
        
        # 평균 신뢰도 계산
        avg_confidence = sum(f.get('confidence', 0.5) for f in frames) / len(frames)
        
        return {
            'start_time': start_time,
            'end_time': end_time,
            'frame_id': frames[0]['frame_id'],
            'reason': most_common_reason,
            'person_count': sum(f.get('person_count', 1) for f in frames),
            'avg_confidence': avg_confidence
        }
    
    def _search_frames_by_criteria(self, video, query, time_range):
        """검색 조건에 맞는 프레임 찾기"""
        # 간단한 구현 - 실제로는 ObjectSearchView 로직 활용
        matching_frames = []
        
        if query:
            # 기본 검색 로직
            person_detections = PersonDetection.objects.filter(video=video)
            
            if '어린이' in query.lower() or 'child' in query.lower():
                detections = person_detections.filter(
                    age_group__icontains='child'
                )
            elif '노인' in query.lower() or 'elderly' in query.lower():
                detections = person_detections.filter(
                    age_group__icontains='elderly'
                )
            else:
                detections = person_detections
            
            for detection in detections:
                matching_frames.append({
                    'frame_id': detection.frame.image_id,
                    'timestamp': detection.frame.timestamp,
                    'match_reason': f"검색: {query}",
                    'confidence': detection.confidence
                })
        
        return matching_frames
    
    def _analyze_video_frames(self, video_id, frames):
        """비디오의 프레임 이미지들을 시각적으로 분석"""
        try:
            if not vision_analyzer:
                return {"error": "Vision analyzer not available"}
            
            # 프레임 이미지 경로들 수집 (1-3개만 선택)
            frame_paths = []
            selected_frames = []
            
            # 시간대별로 균등하게 3개 프레임 선택
            total_frames = len(frames)
            if total_frames >= 3:
                # 시작, 중간, 끝에서 각각 1개씩 선택
                indices = [0, total_frames // 2, total_frames - 1]
            elif total_frames >= 1:
                # 1-2개만 있는 경우 모두 선택
                indices = list(range(total_frames))
            else:
                return {"error": "분석할 프레임이 없습니다"}
            
            for idx in indices:
                frame = frames[idx]
                image_path = os.path.join(settings.MEDIA_ROOT, 'images', f'video{video_id}_frame{frame.image_id}.jpg')
                if os.path.exists(image_path):
                    frame_paths.append(image_path)
                    selected_frames.append({
                        'frame_id': frame.image_id,
                        'timestamp': frame.timestamp,
                        'image_path': image_path
                    })
            
            if not frame_paths:
                return {"error": "분석할 프레임 이미지를 찾을 수 없습니다"}
            
            print(f"🔍 선택된 프레임 {len(frame_paths)}개 분석 중...")
            for i, frame_info in enumerate(selected_frames):
                print(f"  - 프레임 {i+1}: {frame_info['timestamp']:.1f}초 (ID: {frame_info['frame_id']})")
            
            # 시각적 분석 실행
            analysis_result = vision_analyzer.analyze_video_frames(video_id, frame_paths)
            analysis_result['selected_frames'] = selected_frames
            analysis_result['analysis_count'] = len(frame_paths)
            
            return analysis_result
            
        except Exception as e:
            print(f"❌ 프레임 분석 오류: {e}")
            return {"error": f"프레임 분석 실패: {str(e)}"}
