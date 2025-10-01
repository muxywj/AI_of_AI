# api/services/basic_video_analysis.py - LangChain 없이 작동하는 기본 분석 서비스
import os
import json
import cv2
import time
from datetime import datetime
from typing import Dict, List, Optional
from django.conf import settings
from api.models import Video, AnalysisResult

class BasicVideoAnalysisService:
    """LangChain 없이 작동하는 기본 비디오 분석 서비스"""
    
    def __init__(self):
        self.yolo_available = False
        self.yolo_model = None
        self._init_yolo()
    
    def _init_yolo(self):
        """YOLO 모델 초기화"""
        try:
            from ultralytics import YOLO
            model_path = os.path.join(settings.BASE_DIR, 'yolov8n.pt')
            if os.path.exists(model_path):
                self.yolo_model = YOLO(model_path)
                self.yolo_available = True
                print("✅ YOLO 모델 로드 성공")
            else:
                print("⚠️ YOLO 모델 파일을 찾을 수 없습니다")
        except ImportError:
            print("⚠️ YOLO 모듈이 설치되지 않았습니다")
        except Exception as e:
            print(f"⚠️ YOLO 초기화 실패: {e}")
    
    def analyze_video(self, video_id: int, analysis_type: str = 'basic') -> Dict:
        """비디오 분석 실행"""
        try:
            video = Video.objects.get(id=video_id)
            print(f"🎬 기본 비디오 분석 시작: {video.title} (ID: {video_id})")
            
            # 분석 상태 업데이트
            video.analysis_status = 'processing'
            video.save()
            
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
            
            # 기본 분석 실행
            analysis_result = self._perform_basic_analysis(video, video_path)
            
            if analysis_result['success']:
                # 분석 결과 저장
                self._save_analysis_result(video, analysis_result)
                video.analysis_status = 'completed'
                video.is_analyzed = True
                video.save()
                
                print(f"✅ 비디오 분석 완료: {video.title}")
                return {
                    'success': True,
                    'message': '비디오 분석이 완료되었습니다.',
                    'analysis_id': analysis_result.get('analysis_id'),
                    'features_detected': analysis_result.get('features_detected', 0)
                }
            else:
                video.analysis_status = 'failed'
                video.error_message = analysis_result.get('error', '알 수 없는 오류')
                video.save()
                return analysis_result
                
        except Video.DoesNotExist:
            return {'success': False, 'error': '비디오를 찾을 수 없습니다.'}
        except Exception as e:
            print(f"❌ 분석 중 오류 발생: {e}")
            return {'success': False, 'error': f'분석 중 오류가 발생했습니다: {str(e)}'}
    
    def _find_video_path(self, video: Video) -> Optional[str]:
        """비디오 파일 경로 찾기"""
        possible_paths = [
            video.file_path,
            os.path.join(settings.MEDIA_ROOT, video.filename),
            os.path.join(settings.MEDIA_ROOT, 'uploads', video.filename)
        ]
        
        for path in possible_paths:
            if path and os.path.exists(path):
                return path
        
        return None
    
    def _perform_basic_analysis(self, video: Video, video_path: str) -> Dict:
        """기본 비디오 분석 수행"""
        try:
            print(f"🔍 기본 분석 시작: {video_path}")
            
            # OpenCV로 비디오 열기
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {'success': False, 'error': '비디오 파일을 열 수 없습니다.'}
            
            # 비디오 정보 수집
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"📊 비디오 정보: {width}x{height}, {fps:.2f}FPS, {frame_count}프레임, {duration:.2f}초")
            
            # 비디오 정보 업데이트
            video.duration = duration
            video.save()
            
            # 프레임 샘플링 및 분석
            sample_frames = self._sample_frames(cap, frame_count, sample_count=10)
            
            # 객체 감지 (YOLO 사용 가능한 경우)
            detections = []
            if self.yolo_available and self.yolo_model:
                detections = self._detect_objects_in_frames(sample_frames)
            
            # 분석 결과 구성
            analysis_data = {
                'video_info': {
                    'duration': duration,
                    'fps': fps,
                    'resolution': f"{width}x{height}",
                    'frame_count': frame_count
                },
                'object_detections': detections,
                'sample_frames': len(sample_frames),
                'analysis_timestamp': datetime.now().isoformat(),
                'analysis_type': 'basic'
            }
            
            cap.release()
            
            return {
                'success': True,
                'analysis_data': analysis_data,
                'features_detected': len(detections),
                'message': f'기본 분석 완료: {len(detections)}개 객체 감지'
            }
            
        except Exception as e:
            print(f"❌ 기본 분석 실패: {e}")
            return {'success': False, 'error': f'분석 실패: {str(e)}'}
    
    def _sample_frames(self, cap, total_frames: int, sample_count: int = 10) -> List:
        """비디오에서 프레임 샘플링"""
        frames = []
        if total_frames == 0:
            return frames
        
        # 균등하게 샘플링
        step = max(1, total_frames // sample_count)
        frame_indices = list(range(0, total_frames, step))[:sample_count]
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        
        return frames
    
    def _detect_objects_in_frames(self, frames: List) -> List:
        """프레임들에서 객체 감지"""
        detections = []
        
        try:
            for i, frame in enumerate(frames):
                results = self.yolo_model(frame, verbose=False)
                
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            # 객체 정보 추출
                            conf = float(box.conf[0])
                            cls = int(box.cls[0])
                            class_name = self.yolo_model.names[cls]
                            
                            if conf > 0.5:  # 신뢰도 임계값
                                detections.append({
                                    'frame_index': i,
                                    'class_name': class_name,
                                    'confidence': conf,
                                    'class_id': cls
                                })
        except Exception as e:
            print(f"⚠️ 객체 감지 실패: {e}")
        
        return detections
    
    def _save_analysis_result(self, video: Video, analysis_result: Dict):
        """분석 결과 저장"""
        try:
            # JSON 파일로 저장
            timestamp = int(time.time())
            filename = f"analysis_{video.id}_{timestamp}_{video.filename}.json"
            filepath = os.path.join(settings.MEDIA_ROOT, 'analysis_results', filename)
            
            # 디렉토리 생성
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # JSON 저장
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(analysis_result['analysis_data'], f, ensure_ascii=False, indent=2)
            
            # 데이터베이스에 결과 저장
            analysis_obj = AnalysisResult.objects.create(
                video=video,
                analysis_type='basic',
                result_data=analysis_result['analysis_data'],
                json_file_path=filepath,
                features_detected=analysis_result.get('features_detected', 0),
                status='completed'
            )
            
            print(f"✅ 분석 결과 저장 완료: {filename}")
            
        except Exception as e:
            print(f"⚠️ 분석 결과 저장 실패: {e}")

# 전역 인스턴스
_basic_analysis_service = None

def get_basic_video_analysis_service():
    """기본 비디오 분석 서비스 인스턴스 반환"""
    global _basic_analysis_service
    if _basic_analysis_service is None:
        _basic_analysis_service = BasicVideoAnalysisService()
    return _basic_analysis_service
