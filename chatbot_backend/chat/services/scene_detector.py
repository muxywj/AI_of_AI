# chat/services/scene_detector.py - 장면 감지 서비스
import cv2
import numpy as np
import os
import logging
from typing import List, Dict, Tuple
from django.conf import settings
from ..models import Video, VideoScene, SceneAnalysis

logger = logging.getLogger(__name__)

class SceneDetector:
    """비디오 장면 감지 및 분석 클래스"""
    
    def __init__(self):
        self.scene_threshold = 0.3  # 장면 변화 임계값
        self.min_scene_duration = 2.0  # 최소 장면 지속 시간 (초)
        self.max_scene_duration = 60.0  # 최대 장면 지속 시간 (초)
        
    def detect_scenes(self, video_path: str) -> List[Dict]:
        """비디오에서 장면 변화를 감지하고 장면 정보를 반환"""
        try:
            logger.info(f"🎬 장면 감지 시작: {video_path}")
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception("비디오를 열 수 없습니다")
            
            # 비디오 정보 추출
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            logger.info(f"📊 비디오 정보: {total_frames}프레임, {fps:.1f}fps, {duration:.1f}초")
            
            # 장면 변화 감지
            scene_changes = self._detect_scene_changes(cap, fps)
            
            # 장면 정보 생성
            scenes = self._create_scene_info(scene_changes, fps, duration)
            
            cap.release()
            
            logger.info(f"✅ 장면 감지 완료: {len(scenes)}개 장면")
            return scenes
            
        except Exception as e:
            logger.error(f"❌ 장면 감지 실패: {e}")
            raise
    
    def _detect_scene_changes(self, cap: cv2.VideoCapture, fps: float) -> List[int]:
        """장면 변화 지점을 감지"""
        scene_changes = [0]  # 첫 번째 장면 시작
        prev_frame = None
        frame_count = 0
        
        # 샘플링 간격 설정 (성능 최적화)
        sample_interval = max(1, int(fps / 2))  # 초당 2프레임 샘플링
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 샘플링
            if frame_count % sample_interval == 0:
                if prev_frame is not None:
                    # 히스토그램 기반 장면 변화 감지
                    hist_diff = self._calculate_histogram_difference(prev_frame, frame)
                    
                    if hist_diff < self.scene_threshold:
                        scene_changes.append(frame_count)
                        logger.debug(f"장면 변화 감지: 프레임 {frame_count} (차이: {hist_diff:.3f})")
                
                prev_frame = frame.copy()
            
            frame_count += 1
        
        # 마지막 프레임 추가
        scene_changes.append(frame_count - 1)
        
        return scene_changes
    
    def _calculate_histogram_difference(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """두 프레임 간의 히스토그램 차이 계산"""
        try:
            # HSV 색공간으로 변환
            hsv1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
            hsv2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
            
            # 히스토그램 계산
            hist1 = cv2.calcHist([hsv1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist2 = cv2.calcHist([hsv2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            
            # 히스토그램 정규화
            cv2.normalize(hist1, hist1)
            cv2.normalize(hist2, hist2)
            
            # 상관관계 계산
            correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            
            return correlation
            
        except Exception as e:
            logger.warning(f"히스토그램 차이 계산 실패: {e}")
            return 1.0  # 오류 시 변화 없음으로 처리
    
    def _create_scene_info(self, scene_changes: List[int], fps: float, duration: float) -> List[Dict]:
        """장면 변화 지점을 기반으로 장면 정보 생성"""
        scenes = []
        
        for i in range(len(scene_changes) - 1):
            start_frame = scene_changes[i]
            end_frame = scene_changes[i + 1]
            
            start_time = start_frame / fps
            end_time = end_frame / fps
            scene_duration = end_time - start_time
            
            # 최소/최대 지속 시간 필터링
            if scene_duration < self.min_scene_duration:
                continue
            
            if scene_duration > self.max_scene_duration:
                # 긴 장면을 여러 개로 분할
                sub_scenes = self._split_long_scene(start_time, end_time, self.max_scene_duration)
                scenes.extend(sub_scenes)
            else:
                scenes.append({
                    'scene_id': len(scenes) + 1,
                    'start_timestamp': start_time,
                    'end_timestamp': end_time,
                    'duration': scene_duration,
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'frame_count': end_frame - start_frame
                })
        
        return scenes
    
    def _split_long_scene(self, start_time: float, end_time: float, max_duration: float) -> List[Dict]:
        """긴 장면을 여러 개의 짧은 장면으로 분할"""
        scenes = []
        current_start = start_time
        scene_id = 1
        
        while current_start < end_time:
            current_end = min(current_start + max_duration, end_time)
            duration = current_end - current_start
            
            if duration >= self.min_scene_duration:
                scenes.append({
                    'scene_id': scene_id,
                    'start_timestamp': current_start,
                    'end_timestamp': current_end,
                    'duration': duration,
                    'start_frame': int(current_start * 30),  # 가정: 30fps
                    'end_frame': int(current_end * 30),
                    'frame_count': int(duration * 30)
                })
                scene_id += 1
            
            current_start = current_end
        
        return scenes

class SceneAnalyzer:
    """장면 분석 클래스 - LLM 통합"""
    
    def __init__(self):
        self.scene_detector = SceneDetector()
        
    def analyze_video_scenes(self, video_id: int) -> List[Dict]:
        """비디오의 모든 장면을 분석하고 DB에 저장"""
        try:
            logger.info(f"🎬 비디오 장면 분석 시작: {video_id}")
            
            # 비디오 정보 가져오기
            video = Video.objects.get(id=video_id)
            video_path = os.path.join(settings.MEDIA_ROOT, video.file_path)
            
            if not os.path.exists(video_path):
                raise Exception(f"비디오 파일을 찾을 수 없습니다: {video_path}")
            
            # 장면 감지
            scenes = self.scene_detector.detect_scenes(video_path)
            
            # 각 장면 분석 및 저장
            analyzed_scenes = []
            for scene_data in scenes:
                analyzed_scene = self._analyze_and_save_scene(video, scene_data, video_path)
                analyzed_scenes.append(analyzed_scene)
            
            logger.info(f"✅ 비디오 장면 분석 완료: {len(analyzed_scenes)}개 장면")
            return analyzed_scenes
            
        except Exception as e:
            logger.error(f"❌ 비디오 장면 분석 실패: {e}")
            raise
    
    def _analyze_and_save_scene(self, video: Video, scene_data: Dict, video_path: str) -> Dict:
        """개별 장면을 분석하고 DB에 저장"""
        try:
            # VideoScene 모델 생성
            scene = VideoScene.objects.create(
                video=video,
                scene_id=scene_data['scene_id'],
                start_timestamp=scene_data['start_timestamp'],
                end_timestamp=scene_data['end_timestamp'],
                duration=scene_data['duration']
            )
            
            # 장면 프레임 분석
            scene_analysis = self._analyze_scene_frames(scene_data, video_path)
            
            # SceneAnalysis 모델 생성
            analysis = SceneAnalysis.objects.create(
                scene=scene,
                detected_persons=scene_analysis.get('detected_persons', []),
                detected_objects=scene_analysis.get('detected_objects', []),
                person_count=scene_analysis.get('person_count', 0),
                object_count=scene_analysis.get('object_count', 0),
                activity_type=scene_analysis.get('activity_type', ''),
                activity_intensity=scene_analysis.get('activity_intensity', ''),
                brightness_level=scene_analysis.get('brightness_level', 0.0),
                contrast_level=scene_analysis.get('contrast_level', 0.0),
                sharpness_level=scene_analysis.get('sharpness_level', 0.0)
            )
            
            # 장면 정보 업데이트
            scene.scene_type = scene_analysis.get('scene_type', '')
            scene.dominant_objects = scene_analysis.get('dominant_objects', [])
            scene.dominant_colors = scene_analysis.get('dominant_colors', [])
            scene.weather_condition = scene_analysis.get('weather_condition', '')
            scene.time_of_day = scene_analysis.get('time_of_day', '')
            scene.lighting_condition = scene_analysis.get('lighting_condition', '')
            scene.quality_score = scene_analysis.get('quality_score', 0.0)
            scene.confidence_score = scene_analysis.get('confidence_score', 0.0)
            scene.save()
            
            return {
                'scene_id': scene.scene_id,
                'start_timestamp': scene.start_timestamp,
                'end_timestamp': scene.end_timestamp,
                'duration': scene.duration,
                'analysis': scene_analysis
            }
            
        except Exception as e:
            logger.error(f"❌ 장면 분석 및 저장 실패: {e}")
            raise
    
    def _analyze_scene_frames(self, scene_data: Dict, video_path: str) -> Dict:
        """장면의 프레임들을 분석하여 정보 추출"""
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # 장면의 중간 프레임들 샘플링
            start_frame = scene_data['start_frame']
            end_frame = scene_data['end_frame']
            sample_frames = self._sample_scene_frames(cap, start_frame, end_frame, fps)
            
            cap.release()
            
            # 프레임 분석
            analysis_result = self._analyze_frame_samples(sample_frames)
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"❌ 장면 프레임 분석 실패: {e}")
            return {}
    
    def _sample_scene_frames(self, cap: cv2.VideoCapture, start_frame: int, end_frame: int, fps: float) -> List[np.ndarray]:
        """장면에서 대표 프레임들을 샘플링"""
        frames = []
        frame_count = end_frame - start_frame
        
        # 샘플링 전략: 시작, 중간, 끝 프레임
        if frame_count <= 3:
            sample_indices = list(range(frame_count))
        else:
            sample_indices = [0, frame_count // 2, frame_count - 1]
        
        for idx in sample_indices:
            frame_number = start_frame + idx
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        
        return frames
    
    def _analyze_frame_samples(self, frames: List[np.ndarray]) -> Dict:
        """샘플 프레임들을 분석하여 장면 정보 추출"""
        if not frames:
            return {}
        
        # 기본 분석
        brightness_levels = []
        contrast_levels = []
        sharpness_levels = []
        dominant_colors = []
        detected_objects = []
        detected_persons = []
        
        for frame in frames:
            # 밝기 분석
            brightness = np.mean(frame)
            brightness_levels.append(brightness)
            
            # 대비 분석
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            contrast = np.std(gray)
            contrast_levels.append(contrast)
            
            # 선명도 분석 (라플라시안 변수)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_levels.append(laplacian_var)
            
            # 색상 분석
            colors = self._extract_dominant_colors(frame)
            dominant_colors.extend(colors)
            
            # 객체 감지 (간단한 버전)
            objects = self._detect_objects_simple(frame)
            detected_objects.extend(objects)
        
        # 평균값 계산
        avg_brightness = np.mean(brightness_levels)
        avg_contrast = np.mean(contrast_levels)
        avg_sharpness = np.mean(sharpness_levels)
        
        # 장면 유형 결정
        scene_type = self._determine_scene_type(avg_brightness, avg_contrast, dominant_colors)
        
        # 시간대 및 날씨 추정
        time_of_day = self._estimate_time_of_day(avg_brightness, dominant_colors)
        weather_condition = self._estimate_weather_condition(avg_brightness, dominant_colors)
        
        return {
            'scene_type': scene_type,
            'dominant_colors': list(set(dominant_colors)),
            'detected_objects': detected_objects,
            'detected_persons': detected_persons,
            'person_count': len(detected_persons),
            'object_count': len(detected_objects),
            'brightness_level': avg_brightness,
            'contrast_level': avg_contrast,
            'sharpness_level': avg_sharpness,
            'time_of_day': time_of_day,
            'weather_condition': weather_condition,
            'lighting_condition': 'bright' if avg_brightness > 150 else 'normal' if avg_brightness > 100 else 'dark',
            'quality_score': min(1.0, (avg_brightness / 255 + avg_contrast / 100 + avg_sharpness / 1000) / 3),
            'confidence_score': 0.8  # 기본 신뢰도
        }
    
    def _extract_dominant_colors(self, frame: np.ndarray) -> List[str]:
        """프레임에서 주요 색상 추출"""
        try:
            # HSV로 변환
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # 색상 범위 정의
            color_ranges = {
                'red': [(0, 30, 30), (10, 255, 255)],
                'orange': [(10, 30, 30), (25, 255, 255)],
                'yellow': [(25, 30, 30), (40, 255, 255)],
                'green': [(40, 30, 30), (80, 255, 255)],
                'blue': [(80, 30, 30), (130, 255, 255)],
                'purple': [(130, 30, 30), (160, 255, 255)],
                'pink': [(160, 30, 30), (180, 255, 255)]
            }
            
            dominant_colors = []
            for color_name, (lower, upper) in color_ranges.items():
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                ratio = np.sum(mask > 0) / (frame.shape[0] * frame.shape[1])
                if ratio > 0.05:  # 5% 이상이면 주요 색상으로 인정
                    dominant_colors.append(color_name)
            
            return dominant_colors
            
        except Exception as e:
            logger.warning(f"색상 추출 실패: {e}")
            return []
    
    def _detect_objects_simple(self, frame: np.ndarray) -> List[str]:
        """간단한 객체 감지 (실제로는 YOLO 등을 사용해야 함)"""
        # 여기서는 간단한 휴리스틱 사용
        # 실제 구현에서는 기존의 YOLO 모델을 활용
        return ['person']  # 기본값
    
    def _determine_scene_type(self, brightness: float, contrast: float, colors: List[str]) -> str:
        """장면 유형 결정"""
        if brightness < 80:
            return 'indoor_dark'
        elif brightness > 180:
            return 'outdoor_bright'
        elif 'blue' in colors and brightness > 120:
            return 'outdoor_day'
        elif 'orange' in colors or 'yellow' in colors:
            return 'outdoor_sunset'
        else:
            return 'indoor_normal'
    
    def _estimate_time_of_day(self, brightness: float, colors: List[str]) -> str:
        """시간대 추정"""
        if brightness < 60:
            return 'night'
        elif brightness < 100:
            return 'dawn' if 'blue' in colors else 'evening'
        elif brightness > 150:
            return 'afternoon'
        else:
            return 'morning' if 'orange' in colors else 'day'
    
    def _estimate_weather_condition(self, brightness: float, colors: List[str]) -> str:
        """날씨 조건 추정"""
        if brightness < 80 and 'gray' in colors:
            return 'rain'
        elif brightness > 180 and 'white' in colors:
            return 'snow'
        elif brightness > 120 and 'blue' in colors:
            return 'sunny'
        else:
            return 'cloudy'

# 전역 인스턴스
scene_analyzer = SceneAnalyzer()
