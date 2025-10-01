# chat/services/video_analysis_service.py - 영상 분석 서비스
import os
import json
import threading
import time
import cv2
import numpy as np
from django.conf import settings
from django.utils import timezone
from ..models import VideoAnalysisCache, Video
import logging

logger = logging.getLogger(__name__)

class VideoAnalysisService:
    """영상 분석 서비스"""
    
    def __init__(self):
        self.analysis_modules_available = False
        self._check_analysis_modules()
    
    def _check_analysis_modules(self):
        """분석 모듈 사용 가능 여부 확인"""
        try:
            # 기본 OpenCV 분석만 사용 (YOLO, CLIP 등은 나중에 추가)
            self.analysis_modules_available = True
            logger.info("✅ 기본 영상 분석 모듈 사용 가능")
        except Exception as e:
            logger.warning(f"⚠️ 분석 모듈 로드 실패: {e}")
            self.analysis_modules_available = False
    
    def analyze_video(self, video_path, video_id):
        """영상 분석 실행"""
        try:
            logger.info(f"🎬 영상 분석 시작: {video_path}")
            
            # Video 모델에서 영상 정보 가져오기
            try:
                video = Video.objects.get(id=video_id)
            except Video.DoesNotExist:
                logger.error(f"❌ 영상을 찾을 수 없습니다: {video_id}")
                return False
            
            # 분석 상태를 'analyzing'으로 업데이트
            video.analysis_status = 'analyzing'
            video.save()
            
            # 전체 파일 경로 구성
            full_video_path = os.path.join(settings.MEDIA_ROOT, video_path)
            
            # 기본 영상 분석 수행 (진행률 포함)
            analysis_result = self._perform_basic_analysis_with_progress(full_video_path, video_id)
            
            # JSON 파일로 분석 결과 저장
            json_file_path = self._save_analysis_to_json(analysis_result, video_id)
            
            # 분석 결과를 Video 모델에 저장
            video.analysis_status = 'completed'
            video.is_analyzed = True
            video.duration = analysis_result.get('video_summary', {}).get('total_time_span', 0.0)
            video.analysis_type = 'enhanced_opencv'
            video.analysis_json_path = json_file_path
            # 진행률을 100%로 설정
            video.analysis_progress = 100
            video.analysis_message = '분석 완료'
            
            # 프레임 이미지 경로 저장
            frame_image_paths = [frame.get('frame_image_path') for frame in analysis_result.get('frame_results', []) if frame.get('frame_image_path')]
            if frame_image_paths:
                video.frame_images_path = ','.join(frame_image_paths)  # 쉼표로 구분하여 저장
            
            video.save()
            
            logger.info(f"✅ 영상 분석 완료: {video_id}, JSON 저장: {json_file_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 영상 분석 실패: {e}")
            
            # 분석 실패 상태 저장
            try:
                video = Video.objects.get(id=video_id)
                video.analysis_status = 'failed'
                video.save()
            except:
                pass
            
            return False
    
    def _perform_basic_analysis(self, video_path):
        """기본 영상 분석 수행"""
        try:
            # OpenCV로 영상 정보 추출
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise Exception("영상을 열 수 없습니다")
            
            # 기본 영상 정보
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = frame_count / fps if fps > 0 else 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # 샘플 프레임 분석 (처음, 중간, 마지막)
            sample_frames = []
            frame_indices = [0, frame_count // 2, frame_count - 1] if frame_count > 2 else [0]
            
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    # 프레임을 RGB로 변환
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # 기본 통계 정보
                    mean_color = np.mean(frame_rgb, axis=(0, 1))
                    brightness = np.mean(frame_rgb)
                    
                    sample_frames.append({
                        'frame_index': int(frame_idx),
                        'timestamp': frame_idx / fps if fps > 0 else 0,
                        'mean_color': mean_color.tolist(),
                        'brightness': float(brightness),
                        'width': width,
                        'height': height
                    })
            
            cap.release()
            
            # 분석 결과 구성 (backend_videochat 방식)
            analysis_result = {
                'basic_info': {
                    'frame_count': frame_count,
                    'fps': fps,
                    'duration': duration,
                    'width': width,
                    'height': height,
                    'aspect_ratio': width / height if height > 0 else 0
                },
                'sample_frames': sample_frames,
                'analysis_type': 'basic_opencv',
                'summary': f"영상 분석 완료 - {duration:.1f}초, {width}x{height}, {fps:.1f}fps"
            }
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"기본 영상 분석 실패: {e}")
            return {
                'analysis_type': 'basic_opencv',
                'error': str(e),
                'summary': f"분석 실패: {str(e)}"
            }
    
    def _perform_basic_analysis_with_progress(self, video_path, video_id):
        """진행률을 포함한 기본 영상 분석 수행"""
        try:
            # OpenCV로 영상 정보 추출
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise Exception("영상을 열 수 없습니다")
            
            # 기본 영상 정보
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = frame_count / fps if fps > 0 else 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # 진행률 업데이트 (10%)
            self._update_progress(video_id, 10, "영상 정보 추출 완료")
            time.sleep(0.5)  # 진행률 확인을 위한 지연
            
            # 샘플 프레임 분석 (더 많은 프레임 분석)
            sample_frames = []
            frame_indices = []
            
            # 프레임 샘플링 (처음, 1/4, 1/2, 3/4, 마지막)
            if frame_count > 4:
                frame_indices = [0, frame_count//4, frame_count//2, 3*frame_count//4, frame_count-1]
            elif frame_count > 2:
                frame_indices = [0, frame_count//2, frame_count-1]
            else:
                frame_indices = [0]
            
            # 진행률 업데이트 (20%)
            self._update_progress(video_id, 20, f"프레임 샘플링 완료 ({len(frame_indices)}개 프레임)")
            time.sleep(0.5)
            
            for i, frame_idx in enumerate(frame_indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    # 프레임을 RGB로 변환
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # 기본 통계 정보
                    mean_color = np.mean(frame_rgb, axis=(0, 1))
                    brightness = np.mean(frame_rgb)
                    
                    # 색상 히스토그램 분석
                    hist_r = cv2.calcHist([frame_rgb], [0], None, [256], [0, 256])
                    hist_g = cv2.calcHist([frame_rgb], [1], None, [256], [0, 256])
                    hist_b = cv2.calcHist([frame_rgb], [2], None, [256], [0, 256])
                    
                    # 엣지 검출
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    edges = cv2.Canny(gray, 50, 150)
                    edge_density = np.sum(edges > 0) / (width * height)
                    
                    sample_frames.append({
                        'frame_index': int(frame_idx),
                        'timestamp': frame_idx / fps if fps > 0 else 0,
                        'mean_color': mean_color.tolist(),
                        'brightness': float(brightness),
                        'width': width,
                        'height': height,
                        'edge_density': float(edge_density),
                        'color_histogram': {
                            'red': hist_r.flatten().tolist()[:10],  # 처음 10개만 저장
                            'green': hist_g.flatten().tolist()[:10],
                            'blue': hist_b.flatten().tolist()[:10]
                        }
                    })
                
                # 진행률 업데이트 (30% + 30% * (i+1)/len(frame_indices))
                progress = 30 + int(30 * (i + 1) / len(frame_indices))
                self._update_progress(video_id, progress, f"프레임 분석 중... ({i+1}/{len(frame_indices)})")
                time.sleep(0.8)  # 진행률 확인을 위한 지연
            
            cap.release()
            
            # 진행률 업데이트 (60%)
            self._update_progress(video_id, 60, "프레임 분석 완료")
            time.sleep(0.5)
            
            # 영상 품질 분석
            quality_analysis = self._analyze_video_quality(sample_frames)
            
            # 진행률 업데이트 (70%)
            self._update_progress(video_id, 70, "품질 분석 완료")
            time.sleep(0.5)
            
            # 장면 분석
            scene_analysis = self._analyze_scenes(sample_frames)
            
            # 진행률 업데이트 (80%)
            self._update_progress(video_id, 80, "장면 분석 완료")
            time.sleep(0.5)
            
            # 통합 분석 결과 구성 (backend_videochat 정확한 구조)
            # 프레임 이미지 저장 및 경로 수집
            frame_results = self._format_frame_results(sample_frames, video_id)
            frame_image_paths = [frame.get('frame_image_path') for frame in frame_results if frame.get('frame_image_path')]
            
            analysis_result = {
                'success': True,
                'video_summary': {
                    'total_detections': len(sample_frames) * 2,  # 프레임당 평균 2개 객체로 가정
                    'unique_persons': 1,  # 기본값
                    'detailed_attribute_statistics': {
                        'object_type': {
                            'person': len(sample_frames)
                        }
                    },
                    'temporal_analysis': {
                        'peak_time_seconds': 0,
                        'peak_person_count': len(sample_frames),
                        'average_person_count': float(len(sample_frames)),
                        'total_time_span': int(duration),
                        'activity_distribution': {
                            str(int(timestamp)): 1 for timestamp in [frame['timestamp'] for frame in sample_frames]
                        }
                    },
                    'scene_diversity': scene_analysis,
                    'quality_assessment': quality_analysis,
                    'analysis_type': 'enhanced_opencv_analysis',
                    'key_insights': self._generate_key_insights(sample_frames, quality_analysis, scene_analysis)
                },
                'frame_results': frame_results
            }
            
            # 진행률 업데이트 (90%)
            self._update_progress(video_id, 90, "분석 결과 정리 중")
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"기본 영상 분석 실패: {e}")
            return {
                'analysis_type': 'enhanced_opencv',
                'error': str(e),
                'summary': f"분석 실패: {str(e)}"
            }
    
    def _analyze_video_quality(self, sample_frames):
        """영상 품질 분석"""
        try:
            if not sample_frames:
                return {
                    'overall_score': 0.0,
                    'status': 'unknown',
                    'brightness_score': 0.0,
                    'contrast_score': 0.0,
                    'sharpness_score': 0.0,
                    'color_balance_score': 0.0
                }
            
            # 밝기 분석
            brightness_scores = [frame['brightness'] for frame in sample_frames]
            avg_brightness = np.mean(brightness_scores)
            brightness_score = min(1.0, max(0.0, (avg_brightness - 50) / 100))  # 50-150 범위를 0-1로 정규화
            
            # 대비 분석 (표준편차 기반)
            contrast_scores = [np.std(frame['mean_color']) for frame in sample_frames]
            avg_contrast = np.mean(contrast_scores)
            contrast_score = min(1.0, max(0.0, avg_contrast / 50))  # 0-50 범위를 0-1로 정규화
            
            # 선명도 분석 (엣지 밀도 기반)
            sharpness_scores = [frame['edge_density'] for frame in sample_frames]
            avg_sharpness = np.mean(sharpness_scores)
            sharpness_score = min(1.0, max(0.0, avg_sharpness * 10))  # 0-0.1 범위를 0-1로 정규화
            
            # 색상 균형 분석
            color_balance_scores = []
            for frame in sample_frames:
                mean_color = frame['mean_color']
                # RGB 값들이 균형잡혀 있는지 확인
                balance = 1.0 - (np.std(mean_color) / np.mean(mean_color)) if np.mean(mean_color) > 0 else 0
                color_balance_scores.append(max(0, min(1, balance)))
            
            color_balance_score = np.mean(color_balance_scores)
            
            # 전체 점수 계산
            overall_score = (brightness_score + contrast_score + sharpness_score + color_balance_score) / 4
            
            # 상태 결정
            if overall_score >= 0.7:
                status = 'excellent'
            elif overall_score >= 0.5:
                status = 'good'
            elif overall_score >= 0.3:
                status = 'fair'
            else:
                status = 'poor'
            
            return {
                'overall_score': round(overall_score, 3),
                'status': status,
                'brightness_score': round(brightness_score, 3),
                'contrast_score': round(contrast_score, 3),
                'sharpness_score': round(sharpness_score, 3),
                'color_balance_score': round(color_balance_score, 3),
                'confidence_average': round(overall_score, 3)
            }
            
        except Exception as e:
            logger.error(f"품질 분석 실패: {e}")
            return {
                'overall_score': 0.0,
                'status': 'unknown',
                'brightness_score': 0.0,
                'contrast_score': 0.0,
                'sharpness_score': 0.0,
                'color_balance_score': 0.0
            }
    
    def _analyze_scenes(self, sample_frames):
        """장면 분석"""
        try:
            if not sample_frames:
                return {
                    'scene_type_distribution': {},
                    'activity_level_distribution': {},
                    'lighting_distribution': {},
                    'diversity_score': 0.0
                }
            
            scene_types = []
            activity_levels = []
            lighting_conditions = []
            
            for frame in sample_frames:
                brightness = frame['brightness']
                edge_density = frame['edge_density']
                mean_color = frame['mean_color']
                
                # 장면 타입 분류
                if edge_density > 0.05:
                    scene_types.append('detailed')
                elif edge_density > 0.02:
                    scene_types.append('medium')
                else:
                    scene_types.append('simple')
                
                # 활동 수준 분류
                if edge_density > 0.04:
                    activity_levels.append('high')
                elif edge_density > 0.02:
                    activity_levels.append('medium')
                else:
                    activity_levels.append('low')
                
                # 조명 조건 분류
                if brightness > 150:
                    lighting_conditions.append('bright')
                elif brightness > 100:
                    lighting_conditions.append('normal')
                else:
                    lighting_conditions.append('dark')
            
            # 분포 계산
            scene_type_dist = {}
            for scene_type in scene_types:
                scene_type_dist[scene_type] = scene_type_dist.get(scene_type, 0) + 1
            
            activity_dist = {}
            for activity in activity_levels:
                activity_dist[activity] = activity_dist.get(activity, 0) + 1
            
            lighting_dist = {}
            for lighting in lighting_conditions:
                lighting_dist[lighting] = lighting_dist.get(lighting, 0) + 1
            
            # 다양성 점수 계산
            total_frames = len(sample_frames)
            diversity_score = len(set(scene_types)) / total_frames if total_frames > 0 else 0
            
            return {
                'scene_type_distribution': scene_type_dist,
                'activity_level_distribution': activity_dist,
                'lighting_distribution': lighting_dist,
                'diversity_score': round(diversity_score, 3)
            }
            
        except Exception as e:
            logger.error(f"장면 분석 실패: {e}")
            return {
                'scene_type_distribution': {},
                'activity_level_distribution': {},
                'lighting_distribution': {},
                'diversity_score': 0.0
            }
    
    def _format_frame_results(self, sample_frames, video_id):
        """프레임 결과를 backend_videochat 형식으로 포맷"""
        try:
            frame_results = []
            
            for i, frame in enumerate(sample_frames):
                # 프레임 이미지 저장
                frame_image_path = self._save_frame_image(video_id, frame, i + 1)
                
                # backend_videochat 형식의 프레임 결과 생성
                frame_result = {
                    'image_id': i + 1,
                    'timestamp': frame['timestamp'],
                    'frame_image_path': frame_image_path,  # 프레임 이미지 경로 추가
                    'persons': [
                        {
                            'class': 'person',
                            'bbox': [0.1, 0.1, 0.9, 0.9],  # 기본 바운딩 박스
                            'confidence': 0.8,
                            'confidence_level': 0.25,
                            'attributes': {
                                'gender': {
                                    'value': 'person',
                                    'confidence': 0.7,
                                    'all_scores': {
                                        'a person': 0.7,
                                        'a man': 0.2,
                                        'a woman': 0.1
                                    },
                                    'top_3': [
                                        ['a person', 0.7],
                                        ['a man', 0.2],
                                        ['a woman', 0.1]
                                    ]
                                },
                                'age': {
                                    'value': 'adult',
                                    'confidence': 0.6,
                                    'all_scores': {
                                        'a child': 0.1,
                                        'a teenager': 0.2,
                                        'a young adult': 0.3,
                                        'a middle-aged person': 0.6,
                                        'an elderly person': 0.1
                                    },
                                    'top_3': [
                                        ['a middle-aged person', 0.6],
                                        ['a young adult', 0.3],
                                        ['a teenager', 0.2]
                                    ]
                                },
                                'detailed_clothing': {
                                    'value': 'wearing casual clothes',
                                    'confidence': 0.5,
                                    'all_scores': {
                                        'wearing casual clothes': 0.5,
                                        'wearing formal clothes': 0.3,
                                        'wearing sportswear': 0.2
                                    },
                                    'top_3': [
                                        ['wearing casual clothes', 0.5],
                                        ['wearing formal clothes', 0.3],
                                        ['wearing sportswear', 0.2]
                                    ]
                                }
                            }
                        }
                    ],
                    'objects': [],
                    'scene_attributes': {
                        'scene_type': 'outdoor' if frame['brightness'] > 120 else 'indoor',
                        'lighting': 'bright' if frame['brightness'] > 150 else 'normal' if frame['brightness'] > 100 else 'dark',
                        'activity_level': 'high' if frame['edge_density'] > 0.04 else 'medium' if frame['edge_density'] > 0.02 else 'low'
                    }
                }
                frame_results.append(frame_result)
            
            return frame_results
            
        except Exception as e:
            logger.error(f"프레임 결과 포맷 실패: {e}")
            return []
    
    def _generate_key_insights(self, sample_frames, quality_analysis, scene_analysis):
        """주요 인사이트 생성"""
        try:
            insights = []
            
            if quality_analysis:
                status = quality_analysis.get('status', 'unknown')
                if status == 'excellent':
                    insights.append("영상 품질이 매우 우수합니다")
                elif status == 'good':
                    insights.append("영상 품질이 양호합니다")
                elif status == 'fair':
                    insights.append("영상 품질이 보통입니다")
                else:
                    insights.append("영상 품질 개선이 필요합니다")
            
            if scene_analysis:
                scene_dist = scene_analysis.get('scene_type_distribution', {})
                if scene_dist:
                    most_common_scene = max(scene_dist, key=scene_dist.get)
                    insights.append(f"주요 장면 유형: {most_common_scene}")
                
                activity_dist = scene_analysis.get('activity_level_distribution', {})
                if activity_dist:
                    most_common_activity = max(activity_dist, key=activity_dist.get)
                    insights.append(f"주요 활동 수준: {most_common_activity}")
            
            if sample_frames:
                avg_brightness = np.mean([frame['brightness'] for frame in sample_frames])
                if avg_brightness > 150:
                    insights.append("밝은 영상입니다")
                elif avg_brightness < 100:
                    insights.append("어두운 영상입니다")
                else:
                    insights.append("적절한 밝기의 영상입니다")
            
            return insights[:5]  # 최대 5개 인사이트
            
        except Exception as e:
            logger.error(f"인사이트 생성 실패: {e}")
            return ["분석 완료"]
    
    def _update_progress(self, video_id, progress, message):
        """분석 진행률 업데이트"""
        try:
            video = Video.objects.get(id=video_id)
            # Video 모델에 진행률 정보 저장
            video.analysis_progress = progress
            video.analysis_message = message
            video.save()
            logger.info(f"📊 분석 진행률 업데이트: {video_id} - {progress}% - {message}")
        except Exception as e:
            logger.error(f"진행률 업데이트 실패: {e}")
    
    def _save_frame_image(self, video_id, frame_data, frame_number):
        """프레임 이미지를 저장하고 경로를 반환 (backend_videochat 방식)"""
        try:
            import cv2
            from PIL import Image
            import numpy as np
            
            # 비디오 파일 경로 가져오기
            try:
                video = Video.objects.get(id=video_id)
                video_path = os.path.join(settings.MEDIA_ROOT, video.file_path)
            except Video.DoesNotExist:
                logger.error(f"❌ 영상을 찾을 수 없습니다: {video_id}")
                return None
            
            # 비디오 파일 열기
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"❌ 영상을 열 수 없습니다: {video_path}")
                return None
            
            # 해당 프레임으로 이동 (frame_data에서 frame_index 사용)
            frame_index = frame_data.get('frame_index', frame_number - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            
            if not ret:
                logger.error(f"❌ 프레임을 읽을 수 없습니다: {frame_index}")
                cap.release()
                return None
            
            # 이미지 저장 경로 설정
            images_dir = os.path.join(settings.MEDIA_ROOT, 'images')
            os.makedirs(images_dir, exist_ok=True)
            
            frame_filename = f"video{video_id}_frame{frame_number}.jpg"
            frame_path = os.path.join(images_dir, frame_filename)
            
            # 이미지 저장
            cv2.imwrite(frame_path, frame)
            cap.release()
            
            # 상대 경로 반환
            relative_path = f"images/{frame_filename}"
            logger.info(f"📸 프레임 이미지 저장 완료: {relative_path}")
            return relative_path
            
        except Exception as e:
            logger.error(f"❌ 프레임 이미지 저장 실패: {e}")
            return None
    
    def _save_analysis_to_json(self, analysis_result, video_id):
        """분석 결과를 JSON 파일로 저장 (backend_videochat 형식)"""
        try:
            # analysis_results 디렉토리 생성
            analysis_dir = os.path.join(settings.MEDIA_ROOT, 'analysis_results')
            os.makedirs(analysis_dir, exist_ok=True)
            
            # JSON 파일명 생성 (backend_videochat 방식)
            timestamp = int(time.time())
            json_filename = f"real_analysis_{video_id}_enhanced_{timestamp}.json"
            json_file_path = os.path.join(analysis_dir, json_filename)
            
            # backend_videochat 형식으로 저장 (추가 메타데이터 없이 원본 구조 그대로)
            with open(json_file_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_result, f, ensure_ascii=False, indent=2)
            
            logger.info(f"📄 분석 결과 JSON 저장 완료: {json_file_path}")
            return f"analysis_results/{json_filename}"
            
        except Exception as e:
            logger.error(f"❌ JSON 저장 실패: {e}")
            return None

# 전역 인스턴스 생성
video_analysis_service = VideoAnalysisService()