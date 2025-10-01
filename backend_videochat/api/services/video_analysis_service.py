# api/services/video_analysis_service.py - 오류 수정 및 개선
import os
import json
import threading
import time
from django.conf import settings
from api.models import Video, AnalysisResult
from django.utils import timezone

# 분석 모듈들 import (오류 처리 개선)
ANALYSIS_MODULES_AVAILABLE = False
VIDEO_ANALYZER_AVAILABLE = False
DB_BUILDER_AVAILABLE = False

try:
    # 현재 디렉토리와 상위 디렉토리에서 모듈들을 찾기
    import sys
    
    # 현재 서비스 디렉토리에서 상위 api 디렉토리로 경로 추가
    current_dir = os.path.dirname(os.path.abspath(__file__))
    api_dir = os.path.dirname(current_dir)  # api 디렉토리
    sys.path.insert(0, api_dir)
    
    print(f"🔍 모듈 검색 경로 추가: {api_dir}")
    
    # 모듈별 개별 import 시도
    try:
        # api 디렉토리에서 video_analyzer 모듈 import
        from api import video_analyzer
        VIDEO_ANALYZER_AVAILABLE = True
        print("✅ video_analyzer 모듈 로드 성공")
    except ImportError as e:
        print(f"⚠️ video_analyzer 모듈 로드 실패: {e}")
        # 직접 import 시도
        try:
            import video_analyzer
            VIDEO_ANALYZER_AVAILABLE = True
            print("✅ video_analyzer 모듈 직접 로드 성공")
        except ImportError as e2:
            print(f"⚠️ video_analyzer 직접 로드도 실패: {e2}")
            VIDEO_ANALYZER_AVAILABLE = False
    
    try:
        # api 디렉토리에서 db_builder 모듈 import
        from api import db_builder
        DB_BUILDER_AVAILABLE = True
        print("✅ db_builder 모듈 로드 성공")
    except ImportError as e:
        print(f"⚠️ db_builder 모듈 로드 실패: {e}")
        # 직접 import 시도
        try:
            import db_builder
            DB_BUILDER_AVAILABLE = True
            print("✅ db_builder 모듈 직접 로드 성공")
        except ImportError as e2:
            print(f"⚠️ db_builder 직접 로드도 실패: {e2}")
            DB_BUILDER_AVAILABLE = False
    
    # 모든 모듈이 로드되었는지 확인
    ANALYSIS_MODULES_AVAILABLE = VIDEO_ANALYZER_AVAILABLE and DB_BUILDER_AVAILABLE
    
    if ANALYSIS_MODULES_AVAILABLE:
        print("✅ 모든 분석 모듈 로드 완료")
    else:
        print("⚠️ 일부 분석 모듈 로드 실패")
    
except Exception as e:
    print(f"⚠️ 분석 모듈 초기화 실패: {e}")
    VIDEO_ANALYZER_AVAILABLE = False
    DB_BUILDER_AVAILABLE = False
    ANALYSIS_MODULES_AVAILABLE = False

class VideoAnalysisService:
    def __init__(self):
        self.analyzer = None
        self.rag_system = None
        
        if ANALYSIS_MODULES_AVAILABLE:
            try:
                if VIDEO_ANALYZER_AVAILABLE:
                    self.analyzer = video_analyzer.get_video_analyzer()
                    print("✅ 비디오 분석기 초기화 완료")
                
                if DB_BUILDER_AVAILABLE:
                    self.rag_system = db_builder.get_enhanced_video_rag_system()
                    print("✅ RAG 시스템 초기화 완료")
                
                print("✅ 비디오 분석 서비스 초기화 완료")
            except Exception as e:
                print(f"⚠️ 분석 시스템 초기화 실패: {e}")
                self.analyzer = None
                self.rag_system = None
        else:
            print("⚠️ 분석 모듈이 불완전하여 기본 모드로 실행")
    
    def analyze_video(self, video_id, analysis_type='enhanced'):
        """비디오 분석 실행"""
        try:
            video = Video.objects.get(id=video_id)
            print(f"🎬 비디오 분석 시작: {video.title} (ID: {video_id})")
            
            # 분석 상태 업데이트
            video.analysis_status = 'processing'
            video.save()
            
            if not self.analyzer:
                error_msg = "분석기가 초기화되지 않았습니다. 필요한 모듈들이 설치되어 있는지 확인해주세요."
                print(f"❌ {error_msg}")
                video.analysis_status = 'failed'
                video.error_message = error_msg
                video.save()
                return {'success': False, 'error': error_msg}
            
            # 파일 존재 확인
            video_path = self._find_video_path(video)
            if not video_path:
                error_msg = f"비디오 파일을 찾을 수 없습니다: {video.file_path}"
                print(f"❌ {error_msg}")
                video.analysis_status = 'failed'
                video.error_message = error_msg
                video.save()
                return {'success': False, 'error': error_msg}
            
            print(f"📁 비디오 파일 경로: {video_path}")
            
            # 진행률 콜백 함수
            def progress_callback(progress, message):
                print(f"📊 분석 진행률: {progress:.1f}% - {message}")
                # 진행률을 데이터베이스에 저장할 수도 있음
                # video.analysis_progress = progress
                # video.save()
            
            # 고도화된 비디오 분석 실행
            print("🔍 PAR 기반 비디오 분석 실행 중...")
            result = self.analyzer.analyze_video_comprehensive_advanced(
                video, analysis_type, progress_callback
            )
            
            if result and result.get('success', False):
                print("✅ 비디오 분석 완료")
                
                # 분석 결과를 Video 모델에 저장
                video.analysis_data = result
                video.is_analyzed = True
                video.analysis_status = 'completed'
                
                # 기본 메타데이터 업데이트
                if 'video_summary' in result:
                    summary = result['video_summary']
                    if 'temporal_analysis' in summary:
                        temporal = summary['temporal_analysis']
                        if 'total_time_span' in temporal:
                            video.duration = temporal['total_time_span']
                
                # 프레임별 결과를 AnalysisResult에 저장
                print("💾 프레임별 분석 결과 저장 중...")
                self._save_frame_results(video, result.get('frame_results', []))
                
                # RAG 데이터베이스 생성
                print("🧠 RAG 데이터베이스 생성 중...")
                rag_success = self.create_rag_database(video, result)
                
                if rag_success:
                    video.rag_db_created = True
                    print("✅ RAG 데이터베이스 생성 완료")
                else:
                    print("⚠️ RAG 데이터베이스 생성 실패")
                
                video.save()
                
                print(f"🎉 비디오 {video.title} 분석 완료!")
                if 'video_summary' in result:
                    summary = result['video_summary']
                    print(f"   - 검출된 총 인원: {summary.get('total_detections', 0)}명")
                    print(f"   - 고유 인원: {summary.get('unique_persons', 0)}명")
                print(f"   - 분석된 프레임: {result.get('total_frames_analyzed', 0)}개")
                
                return result
            else:
                error_msg = result.get('error', '분석 실패 - 알 수 없는 오류') if result else '분석 결과를 받지 못함'
                print(f"❌ 비디오 분석 실패: {error_msg}")
                video.analysis_status = 'failed'
                video.error_message = error_msg
                video.save()
                return {'success': False, 'error': error_msg}
                
        except Video.DoesNotExist:
            error_msg = f"비디오 ID {video_id}를 찾을 수 없습니다"
            print(f"❌ {error_msg}")
            return {'success': False, 'error': error_msg}
        except Exception as e:
            print(f"❌ 분석 중 오류 발생: {e}")
            import traceback
            print(f"상세 오류: {traceback.format_exc()}")
            
            try:
                video = Video.objects.get(id=video_id)
                video.analysis_status = 'failed'
                video.error_message = str(e)
                video.save()
            except:
                pass
            
            return {'success': False, 'error': str(e)}
    
    def _find_video_path(self, video):
        """비디오 파일 경로 찾기 (개선됨)"""
        # 가능한 경로들 확인
        possible_paths = []
        
        # 저장된 파일 경로
        if hasattr(video, 'file_path') and video.file_path:
            if os.path.isabs(video.file_path):
                possible_paths.append(video.file_path)
            else:
                possible_paths.append(os.path.join(settings.MEDIA_ROOT, video.file_path))
        
        # 다른 가능한 경로들
        if hasattr(video, 'filename') and video.filename:
            possible_paths.extend([
                os.path.join(settings.MEDIA_ROOT, 'videos', video.filename),
                os.path.join(settings.MEDIA_ROOT, 'uploads', video.filename),
                os.path.join(settings.MEDIA_ROOT, video.filename)
            ])
        
        # 경로 확인
        for path in possible_paths:
            if path and os.path.exists(path):
                print(f"📁 비디오 파일 발견: {path}")
                return path
        
        print(f"❌ 비디오 파일을 찾을 수 없음. 확인한 경로들:")
        for i, path in enumerate(possible_paths, 1):
            print(f"   {i}. {path} - {'존재함' if os.path.exists(path) else '존재하지 않음'}")
        
        return None
    
    def _save_frame_results(self, video, frame_results):
        """프레임별 분석 결과를 데이터베이스에 저장"""
        try:
            # 기존 결과 삭제
            AnalysisResult.objects.filter(video=video).delete()
            
            if not frame_results:
                print("⚠️ 저장할 프레임 결과가 없음")
                return
            
            # 이미지 저장 디렉토리 생성
            images_dir = os.path.join(settings.MEDIA_ROOT, 'images')
            os.makedirs(images_dir, exist_ok=True)
            
            # 비디오 파일 경로
            video_path = self._find_video_path(video)
            if not video_path:
                print("❌ 비디오 파일을 찾을 수 없어 이미지 저장을 건너뜁니다")
                video_path = None
            
            # 배치로 저장 (성능 향상)
            analysis_objects = []
            
            for frame_result in frame_results:
                try:
                    # ✅ 이미지 저장
                    image_path = None
                    if video_path:
                        image_path = self._save_frame_image(video, frame_result, video_path, images_dir)
                        if image_path:
                            print(f"✅ 이미지 저장 성공: {image_path}")
                        else:
                            print(f"⚠️ 이미지 저장 실패, 프레임 정보만 저장")
                    
                    analysis_obj = AnalysisResult(
                        video=video,
                        frame_id=frame_result.get('image_id', 0),
                        timestamp=frame_result.get('timestamp', 0),
                        persons_detected=frame_result.get('persons', []),
                        scene_analysis=frame_result.get('scene_analysis', {}),
                        quality_metrics=frame_result.get('quality_assessment', {}),
                        image_path=image_path  # ✅ 이미지 경로 저장
                    )
                    analysis_objects.append(analysis_obj)
                    
                    # 배치 크기 제한 (메모리 관리)
                    if len(analysis_objects) >= 100:
                        AnalysisResult.objects.bulk_create(analysis_objects)
                        analysis_objects = []
                
                except Exception as frame_error:
                    print(f"⚠️ 프레임 {frame_result.get('image_id', '?')} 결과 저장 실패: {frame_error}")
                    continue
            
            # 남은 객체들 저장
            if analysis_objects:
                AnalysisResult.objects.bulk_create(analysis_objects)
                
            print(f"💾 {len(frame_results)}개 프레임 결과 저장 완료")
            
        except Exception as e:
            print(f"⚠️ 프레임 결과 저장 실패: {e}")
            import traceback
            print(f"상세 오류: {traceback.format_exc()}")
    
    def _save_frame_image(self, video, frame_data, video_path, images_dir):
        """프레임 이미지를 media/images에 저장"""
        try:
            import cv2
            
            frame_id = frame_data.get('image_id', 0)
            timestamp = frame_data.get('timestamp', 0)
            
            print(f"🖼️ 프레임 이미지 저장 시작: video_id={video.id}, image_id={frame_id}")
            
            # OpenCV로 프레임 추출
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ret, frame = cap.read()
            cap.release()
            
            if ret and frame is not None:
                filename = f"video{video.id}_frame{frame_id}.jpg"
                filepath = os.path.join(images_dir, filename)
                success = cv2.imwrite(filepath, frame)
                
                if success:
                    relative_path = os.path.relpath(filepath, settings.MEDIA_ROOT)
                    return relative_path
                else:
                    print(f"❌ 이미지 저장 실패: {filename}")
                    return None
            else:
                print(f"❌ 프레임 읽기 실패: {frame_id}")
                return None
                
        except Exception as e:
            print(f"❌ 이미지 저장 오류: {e}")
            return None
    
    def create_rag_database(self, video, analysis_result):
        """RAG 데이터베이스 생성"""
        if not self.rag_system:
            print("⚠️ RAG 시스템이 초기화되지 않음")
            return False
            
        try:
            # analysis 디렉토리 생성
            analysis_dir = os.path.join(settings.MEDIA_ROOT, 'analysis')
            os.makedirs(analysis_dir, exist_ok=True)
            
            # JSON 파일로 분석 결과 저장
            json_path = os.path.join(analysis_dir, f'video_{video.id}_analysis.json')
            
            # JSON 저장을 위한 데이터 준비
            json_data = {
                'metadata': {
                    'video_id': video.id,
                    'video_title': video.title,
                    'analysis_date': timezone.now().isoformat(),
                    'analysis_type': getattr(video, 'analysis_type', 'enhanced'),
                    'duration': getattr(video, 'duration', 0) or 0,
                },
                'frame_results': analysis_result.get('frame_results', []),
                'video_summary': analysis_result.get('video_summary', {}),
                'quality_metrics': analysis_result.get('quality_metrics', {}),
                'analysis_config': analysis_result.get('analysis_config', {})
            }
            
            print(f"💾 분석 결과를 JSON으로 저장: {json_path}")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2, default=str)
            
            # RAG 시스템에서 JSON 처리
            print("🧠 RAG 벡터 데이터베이스 생성 중...")
            success = self.rag_system.process_video_analysis_json_advanced(
                json_path, str(video.id)
            )
            
            if success:
                video.rag_db_path = json_path
                return True
            else:
                return False
            
        except Exception as e:
            print(f"❌ RAG DB 생성 실패: {e}")
            import traceback
            print(f"상세 오류: {traceback.format_exc()}")
            return False
    
    def chat_with_video(self, video_id, message):
        """비디오와 채팅"""
        if not self.rag_system:
            return {'error': 'RAG 시스템이 초기화되지 않았습니다.'}
            
        try:
            video = Video.objects.get(id=video_id)
            
            if not video.rag_db_created:
                return {'error': 'RAG 데이터베이스가 생성되지 않았습니다. 분석이 완료될 때까지 기다려주세요.'}
            
            print(f"🤖 비디오 채팅: {message}")
            
            # RAG 시스템을 통한 답변 생성
            answer = self.rag_system.generate_contextual_korean_answer(
                str(video.id), message
            )
            
            # 검색 결과도 함께 반환
            search_results = self.rag_system.intelligent_search_video_content(
                str(video.id), message
            )
            
            print(f"✅ 채팅 응답 생성 완료")
            print(f"   - 검색 결과: {len(search_results)}개")
            
            return {
                'answer': answer,
                'search_results': search_results[:8],  # 상위 8개
                'video_info': {
                    'title': video.title,
                    'analysis_status': video.analysis_status,
                    'is_analyzed': video.is_analyzed
                }
            }
            
        except Video.DoesNotExist:
            return {'error': f'비디오 ID {video_id}를 찾을 수 없습니다.'}
        except Exception as e:
            print(f"❌ 채팅 처리 실패: {e}")
            import traceback
            print(f"상세 오류: {traceback.format_exc()}")
            return {'error': str(e)}
    
    def start_background_analysis(self, video_id, analysis_type='enhanced'):
        """백그라운드에서 비디오 분석 시작"""
        def analyze_in_background():
            print(f"🚀 백그라운드 분석 시작: 비디오 ID {video_id}")
            result = self.analyze_video(video_id, analysis_type)
            if result.get('success'):
                print(f"✅ 백그라운드 분석 완료: 비디오 ID {video_id}")
            else:
                print(f"❌ 백그라운드 분석 실패: 비디오 ID {video_id} - {result.get('error')}")
        
        thread = threading.Thread(target=analyze_in_background, daemon=True)
        thread.start()
        
        print(f"🔄 비디오 {video_id} 백그라운드 분석 스케줄 완료")
    
    def get_analysis_status(self, video_id):
        """분석 상태 조회"""
        try:
            video = Video.objects.get(id=video_id)
            
            # 기본 상태 정보
            status_info = {
                'video_id': video.id,
                'title': video.title,
                'status': video.analysis_status,
                'is_analyzed': video.is_analyzed,
                'rag_db_created': video.rag_db_created,
                'error_message': video.error_message,
                'upload_date': video.upload_date.isoformat() if video.upload_date else None,
                'file_size': video.file_size,
                'analysis_type': getattr(video, 'analysis_type', 'unknown')
            }
            
            # 분석 요약 정보 추가
            if video.analysis_data and isinstance(video.analysis_data, dict):
                video_summary = video.analysis_data.get('video_summary', {})
                status_info['analysis_summary'] = {
                    'total_detections': video_summary.get('total_detections', 0),
                    'unique_persons': video_summary.get('unique_persons', 0),
                    'frames_analyzed': video.analysis_data.get('total_frames_analyzed', 0),
                    'processing_time': video.analysis_data.get('analysis_config', {}).get('processing_time', 0)
                }
            
            # 시스템 상태 정보 추가
            status_info['system_status'] = {
                'analyzer_available': self.analyzer is not None,
                'rag_system_available': self.rag_system is not None,
                'modules_status': {
                    'video_analyzer': VIDEO_ANALYZER_AVAILABLE,
                    'db_builder': DB_BUILDER_AVAILABLE,
                    'analysis_modules': ANALYSIS_MODULES_AVAILABLE
                }
            }
            
            return status_info
            
        except Video.DoesNotExist:
            return {'error': f'비디오 ID {video_id}를 찾을 수 없습니다'}
        except Exception as e:
            print(f"❌ 상태 조회 실패: {e}")
            return {'error': str(e)}
    
    def get_system_status(self):
        """시스템 전체 상태 조회"""
        return {
            'service_initialized': True,
            'analyzer_available': self.analyzer is not None,
            'rag_system_available': self.rag_system is not None,
            'modules_status': {
                'video_analyzer': VIDEO_ANALYZER_AVAILABLE,
                'db_builder': DB_BUILDER_AVAILABLE,
                'analysis_modules': ANALYSIS_MODULES_AVAILABLE
            },
            'total_videos': Video.objects.count(),
            'analyzed_videos': Video.objects.filter(is_analyzed=True).count(),
            'pending_videos': Video.objects.filter(analysis_status='pending').count(),
            'failed_videos': Video.objects.filter(analysis_status='failed').count()
        }
    
    def cleanup_old_analysis(self, days=30):
        """오래된 분석 결과 정리"""
        try:
            from datetime import datetime, timedelta
            
            cutoff_date = datetime.now() - timedelta(days=days)
            old_videos = Video.objects.filter(upload_date__lt=cutoff_date, is_analyzed=False)
            
            deleted_count = 0
            for video in old_videos:
                try:
                    # 파일 삭제
                    if hasattr(video, 'file_path') and video.file_path:
                        file_path = video.file_path
                        if not os.path.isabs(file_path):
                            file_path = os.path.join(settings.MEDIA_ROOT, file_path)
                        
                        if os.path.exists(file_path):
                            os.remove(file_path)
                    
                    # DB에서 삭제
                    video.delete()
                    deleted_count += 1
                    
                except Exception as e:
                    print(f"⚠️ 비디오 {video.id} 삭제 실패: {e}")
            
            print(f"🧹 오래된 비디오 {deleted_count}개 정리 완료")
            return {'deleted_count': deleted_count}
            
        except Exception as e:
            print(f"❌ 정리 작업 실패: {e}")
            return {'error': str(e)}

# 전역 서비스 인스턴스
_video_analysis_service = None

def get_video_analysis_service():
    global _video_analysis_service
    if _video_analysis_service is None:
        _video_analysis_service = VideoAnalysisService()
    return _video_analysis_service

def get_video_analyzer():
    """VideoAnalyzer 인스턴스를 반환하는 함수"""
    try:
        if VIDEO_ANALYZER_AVAILABLE:
            service = get_video_analysis_service()
            return service.analyzer
        return None
    except Exception as e:
        print(f"⚠️ get_video_analyzer 오류: {e}")
        return None

def get_analyzer_status():
    """분석기 상태 확인 함수"""
    try:
        analyzer = get_video_analyzer()
        if analyzer:
            return {
                'status': 'available',
                'clip_available': getattr(analyzer, 'clip_available', False),
                'ocr_available': getattr(analyzer, 'ocr_available', False),
                'vqa_available': getattr(analyzer, 'vqa_available', False),
                'scene_graph_available': getattr(analyzer, 'scene_graph_available', False)
            }
        else:
            return {'status': 'unavailable'}
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }

def get_service_status():
    """서비스 상태 확인 함수"""
    try:
        service = get_video_analysis_service()
        return service.get_system_status()
    except Exception as e:
        return {
            'error': str(e),
            'service_initialized': False,
            'analyzer_available': False,
            'rag_system_available': False
        }