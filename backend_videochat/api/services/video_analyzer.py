# chat/video_analyzer.py - 고도화된 PAR 기반 영상 분석 시스템

import os
import json
import numpy as np
import cv2
import time
import torch
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv

# 환경 설정
load_dotenv()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# API 설정
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# 필수 모델들만 import
YOLO_AVAILABLE = False
CLIP_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print("✅ YOLO 로드 성공")
except ImportError:
    print("⚠️ YOLO 미설치 - 객체 감지 기능 제한")

try:
    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
    print("✅ CLIP 로드 성공")
except ImportError:
    print("⚠️ CLIP 미설치 - 속성 분석 기능 제한")

# API 클라이언트
try:
    from groq import Groq
    groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
except ImportError:
    groq_client = None

class AdvancedPedestrianAttributeRecognizer:
    """고도화된 보행자 속성 인식 모듈"""
    
    def __init__(self, device="cpu"):
        self.device = device
        self.clip_model = None
        self.clip_processor = None
        
        # 확장된 속성 분류 템플릿
        self.attribute_templates = {
            'gender': ['a man', 'a woman', 'a person'],
            'age': ['a child', 'a teenager', 'a young adult', 'a middle-aged person', 'an elderly person'],
            'detailed_clothing': [
                'wearing a t-shirt', 'wearing a long sleeve shirt', 'wearing a polo shirt', 
                'wearing a tank top', 'wearing a sweater', 'wearing a hoodie',
                'wearing jeans', 'wearing dress pants', 'wearing shorts', 'wearing leggings',
                'wearing a mini skirt', 'wearing a long skirt', 'wearing a dress'
            ],
            'clothing_color': [
                'wearing red clothes', 'wearing blue clothes', 'wearing black clothes', 
                'wearing white clothes', 'wearing green clothes', 'wearing yellow clothes',
                'wearing pink clothes', 'wearing purple clothes', 'wearing orange clothes',
                'wearing gray clothes', 'wearing brown clothes'
            ],
            'accessories': [
                'wearing glasses', 'wearing sunglasses', 'wearing a hat', 'wearing a cap',
                'carrying a bag', 'carrying a backpack', 'carrying a handbag',
                'wearing a watch', 'carrying a phone', 'wearing earphones'
            ],
            'posture': [
                'standing upright', 'walking normally', 'walking fast', 'running',
                'sitting down', 'looking at phone', 'talking on phone', 'looking around'
            ],
            'facial_attributes': [
                'smiling', 'serious expression', 'wearing a mask', 'no mask',
                'looking forward', 'looking down', 'looking sideways'
            ],
            'hair_style': [
                'with short hair', 'with long hair', 'with curly hair', 'with straight hair',
                'bald', 'wearing a hat covering hair'
            ]
        }
        
        if CLIP_AVAILABLE:
            self._load_clip_model()
    
    def _load_clip_model(self):
        """CLIP 모델 로딩"""
        try:
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
            print("✅ 고도화된 PAR용 CLIP 모델 로드 완료")
        except Exception as e:
            print(f"⚠️ CLIP 모델 로드 실패: {e}")
    
    def extract_detailed_attributes(self, person_crop, bbox_info=None, context_info=None):
        """고도화된 보행자 속성 추출"""
        if not self.clip_model:
            return self._extract_basic_attributes(person_crop, bbox_info)
        
        try:
            from PIL import Image
            
            # OpenCV 이미지를 PIL로 변환
            if len(person_crop.shape) == 3:
                person_crop_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            else:
                person_crop_rgb = person_crop
            
            image = Image.fromarray(person_crop_rgb)
            
            attributes = {}
            
            # 각 속성별로 CLIP 분석 수행 (임계값 낮춤)
            for attr_name, templates in self.attribute_templates.items():
                inputs = self.clip_processor(
                    text=templates, 
                    images=image, 
                    return_tensors="pt", 
                    padding=True
                )
                
                outputs = self.clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
                
                best_idx = probs.argmax().item()
                confidence = probs[0][best_idx].item()
                
                # 신뢰도 임계값을 낮춰서 더 많은 속성 추출
                if confidence > 0.2:  # 0.3에서 0.2로 낮춤
                    attributes[attr_name] = {
                        'value': templates[best_idx].replace('a ', '').replace('an ', ''),
                        'confidence': float(confidence),
                        'all_scores': {template: float(score) for template, score in zip(templates, probs[0])},
                        'top_3': self._get_top_n_results(templates, probs[0], 3)
                    }
                else:
                    # 신뢰도가 낮아도 최상위 결과는 저장
                    attributes[attr_name] = {
                        'value': templates[best_idx].replace('a ', '').replace('an ', ''),
                        'confidence': float(confidence),
                        'status': 'low_confidence'
                    }
            
            # 얼굴 영역 별도 분석
            face_region = self._extract_face_region(person_crop)
            if face_region is not None:
                attributes['facial_details'] = self._analyze_facial_attributes(face_region)
            
            # 포즈 분석
            attributes['pose_analysis'] = self._analyze_pose(person_crop)
            
            # 컨텍스트 정보 추가
            if context_info:
                attributes['context'] = context_info
            
            return attributes
            
        except Exception as e:
            print(f"⚠️ 고도화된 속성 추출 오류: {e}")
            return self._extract_basic_attributes(person_crop, bbox_info)
    
    def _get_top_n_results(self, templates, probs, n=3):
        """상위 N개 결과 반환"""
        top_indices = probs.argsort(descending=True)[:n]
        return [(templates[i], float(probs[i])) for i in top_indices]
    
    def _extract_face_region(self, person_crop):
        """얼굴 영역 추출"""
        try:
            # 상단 1/4 영역을 얼굴로 가정
            h, w = person_crop.shape[:2]
            face_region = person_crop[:h//4, :]
            
            if face_region.size > 0:
                return face_region
        except Exception as e:
            print(f"⚠️ 얼굴 영역 추출 실패: {e}")
        
        return None
    
    def _analyze_facial_attributes(self, face_region):
        """얼굴 속성 분석"""
        try:
            # 간단한 얼굴 분석 (실제로는 더 정교한 얼굴 인식 모델 사용 가능)
            avg_brightness = np.mean(face_region)
            
            return {
                'brightness': float(avg_brightness),
                'estimated_lighting': 'bright' if avg_brightness > 120 else 'dark',
                'face_size_ratio': face_region.size / (face_region.shape[0] * face_region.shape[1])
            }
        except Exception as e:
            print(f"⚠️ 얼굴 속성 분석 실패: {e}")
            return {}
    
    def _analyze_pose(self, person_crop):
        """포즈 분석"""
        try:
            h, w = person_crop.shape[:2]
            aspect_ratio = h / w
            
            # 간단한 포즈 추정
            pose_info = {
                'height_width_ratio': float(aspect_ratio),
                'estimated_pose': 'standing' if aspect_ratio > 2.0 else 'sitting_or_crouching',
                'body_orientation': self._estimate_body_orientation(person_crop)
            }
            
            return pose_info
        except Exception as e:
            print(f"⚠️ 포즈 분석 실패: {e}")
            return {}
    
    def _estimate_body_orientation(self, person_crop):
        """신체 방향 추정"""
        # 간단한 방향 추정 로직
        h, w = person_crop.shape[:2]
        left_half = person_crop[:, :w//2]
        right_half = person_crop[:, w//2:]
        
        left_intensity = np.mean(left_half)
        right_intensity = np.mean(right_half)
        
        if abs(left_intensity - right_intensity) < 10:
            return 'front_facing'
        elif left_intensity > right_intensity:
            return 'slightly_left'
        else:
            return 'slightly_right'
    
    def _extract_basic_attributes(self, person_crop, bbox_info=None):
        """기본 속성 추출 (CLIP 없을 때)"""
        h, w = person_crop.shape[:2]
        
        # 색상 기반 간단한 속성 추출
        hsv = cv2.cvtColor(person_crop, cv2.COLOR_BGR2HSV)
        
        # 상의 영역 (상단 1/3)
        upper_region = person_crop[:h//3, :]
        upper_color = self._get_dominant_color(upper_region)
        
        # 하의 영역 (중간 1/3)
        lower_region = person_crop[h//3:2*h//3, :]
        lower_color = self._get_dominant_color(lower_region)
        
        attributes = {
            'gender': {'value': 'unknown', 'confidence': 0.0},
            'age': {'value': 'unknown', 'confidence': 0.0},
            'detailed_clothing': {'value': f'{upper_color} shirt', 'confidence': 0.5},
            'clothing_color': {'value': upper_color, 'confidence': 0.6},
            'accessories': {'value': 'unknown', 'confidence': 0.0},
            'posture': {'value': 'standing', 'confidence': 0.4},
            'pose_analysis': {
                'height_width_ratio': float(h/w),
                'estimated_pose': 'standing' if h/w > 2.0 else 'sitting'
            }
        }
        
        return attributes
    
    def _get_dominant_color(self, image_region):
        """영역의 주요 색상 추출"""
        try:
            # HSV로 변환하여 색상 분석
            hsv = cv2.cvtColor(image_region, cv2.COLOR_BGR2HSV)
            h_mean = np.mean(hsv[:, :, 0])
            
            # 색상 범위별 분류 (더 세분화)
            if h_mean < 10 or h_mean > 170:
                return 'red'
            elif h_mean < 25:
                return 'orange'
            elif h_mean < 40:
                return 'yellow'
            elif h_mean < 80:
                return 'green'
            elif h_mean < 130:
                return 'blue'
            elif h_mean < 160:
                return 'purple'
            else:
                return 'pink'
        except:
            return 'unknown'

class EnhancedPersonTracker:
    """향상된 보행자 추적 모듈"""
    
    def __init__(self):
        self.tracked_persons = {}
        self.next_id = 1
        self.max_distance = 80  # 거리 임계값 조정
        self.track_history = {}  # 추적 이력 저장
        
    def update_tracks_advanced(self, detections, frame_id, timestamp):
        """고도화된 추적 ID 업데이트"""
        current_detections = []
        
        for detection in detections:
            if detection.get('class') == 'person':
                bbox = detection['bbox']
                center = self._get_bbox_center(bbox)
                
                # 기존 트랙과 매칭
                best_match_id = self._find_best_match(center, detection, frame_id)
                
                if best_match_id:
                    # 기존 트랙 업데이트
                    self._update_existing_track(best_match_id, center, detection, frame_id, timestamp)
                    track_id = best_match_id
                else:
                    # 새로운 트랙 생성
                    track_id = self._create_new_track(center, detection, frame_id, timestamp)
                
                detection['track_id'] = track_id
                detection['track_confidence'] = self._calculate_track_confidence(track_id)
                current_detections.append(detection)
        
        # 오래된 트랙 정리
        self._cleanup_old_tracks(frame_id)
        
        return current_detections
    
    def _find_best_match(self, center, detection, frame_id):
        """가장 적합한 기존 트랙 찾기"""
        best_match_id = None
        min_distance = float('inf')
        
        for track_id, track_info in self.tracked_persons.items():
            # 거리 기반 매칭
            last_center = track_info['last_center']
            distance = np.sqrt((center[0] - last_center[0])**2 + (center[1] - last_center[1])**2)
            
            # 속성 유사도 고려
            attribute_similarity = self._calculate_attribute_similarity(
                detection.get('attributes', {}), 
                track_info.get('attributes', {})
            )
            
            # 종합 점수 계산
            composite_score = distance * (1 - attribute_similarity * 0.3)
            
            if composite_score < min_distance and distance < self.max_distance:
                min_distance = composite_score
                best_match_id = track_id
        
        return best_match_id
    
    def _calculate_attribute_similarity(self, attrs1, attrs2):
        """속성 유사도 계산"""
        if not attrs1 or not attrs2:
            return 0.0
        
        similarity_score = 0.0
        comparison_count = 0
        
        key_attributes = ['clothing_color', 'detailed_clothing', 'accessories']
        
        for attr in key_attributes:
            if attr in attrs1 and attr in attrs2:
                val1 = attrs1[attr].get('value', '')
                val2 = attrs2[attr].get('value', '')
                
                if val1 and val2 and val1 == val2:
                    similarity_score += 1.0
                comparison_count += 1
        
        return similarity_score / max(comparison_count, 1)
    
    def _update_existing_track(self, track_id, center, detection, frame_id, timestamp):
        """기존 트랙 업데이트"""
        track_info = self.tracked_persons[track_id]
        track_info['last_center'] = center
        track_info['last_frame'] = frame_id
        track_info['last_timestamp'] = timestamp
        
        # 속성 정보 업데이트 (더 높은 신뢰도로)
        new_attributes = detection.get('attributes', {})
        if new_attributes:
            if 'attributes' not in track_info:
                track_info['attributes'] = new_attributes
            else:
                track_info['attributes'] = self._merge_attributes(
                    track_info['attributes'], new_attributes
                )
        
        # 이동 이력 저장
        if track_id not in self.track_history:
            self.track_history[track_id] = []
        
        self.track_history[track_id].append({
            'frame_id': frame_id,
            'timestamp': timestamp,
            'center': center,
            'bbox': detection['bbox']
        })
    
    def _create_new_track(self, center, detection, frame_id, timestamp):
        """새로운 트랙 생성"""
        track_id = self.next_id
        self.tracked_persons[track_id] = {
            'first_frame': frame_id,
            'last_frame': frame_id,
            'first_timestamp': timestamp,
            'last_timestamp': timestamp,
            'last_center': center,
            'attributes': detection.get('attributes', {}),
            'track_quality': 1.0
        }
        
        # 이력 초기화
        self.track_history[track_id] = [{
            'frame_id': frame_id,
            'timestamp': timestamp,
            'center': center,
            'bbox': detection['bbox']
        }]
        
        self.next_id += 1
        return track_id
    
    def _merge_attributes(self, old_attrs, new_attrs):
        """속성 정보 병합 (더 높은 신뢰도 우선)"""
        merged = old_attrs.copy()
        
        for attr_name, new_attr_data in new_attrs.items():
            if attr_name not in merged:
                merged[attr_name] = new_attr_data
            else:
                old_confidence = merged[attr_name].get('confidence', 0)
                new_confidence = new_attr_data.get('confidence', 0)
                
                if new_confidence > old_confidence:
                    merged[attr_name] = new_attr_data
        
        return merged
    
    def _calculate_track_confidence(self, track_id):
        """추적 신뢰도 계산"""
        if track_id not in self.track_history:
            return 0.5
        
        history = self.track_history[track_id]
        
        # 지속 시간 기반 신뢰도
        duration_frames = len(history)
        duration_confidence = min(1.0, duration_frames / 30)  # 30프레임까지 증가
        
        # 이동 일관성 기반 신뢰도
        movement_consistency = self._calculate_movement_consistency(history)
        
        return (duration_confidence + movement_consistency) / 2
    
    def _calculate_movement_consistency(self, history):
        """이동 일관성 계산"""
        if len(history) < 3:
            return 0.5
        
        velocities = []
        for i in range(1, len(history)):
            prev_center = history[i-1]['center']
            curr_center = history[i]['center']
            
            velocity = np.sqrt(
                (curr_center[0] - prev_center[0])**2 + 
                (curr_center[1] - prev_center[1])**2
            )
            velocities.append(velocity)
        
        if not velocities:
            return 0.5
        
        # 속도 변화의 표준편차가 낮을수록 일관성 높음
        velocity_std = np.std(velocities)
        consistency = max(0.0, 1.0 - velocity_std / 50.0)  # 정규화
        
        return consistency
    
    def _get_bbox_center(self, bbox):
        """바운딩 박스의 중심점 계산"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def _cleanup_old_tracks(self, current_frame, max_age=45):  # 30에서 45로 증가
        """오래된 트랙 제거"""
        to_remove = []
        for track_id, track_info in self.tracked_persons.items():
            if current_frame - track_info['last_frame'] > max_age:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.tracked_persons[track_id]
            if track_id in self.track_history:
                del self.track_history[track_id]

class AdaptiveFrameSampler:
    """적응적 프레임 샘플링"""
    
    def __init__(self):
        self.sampling_strategies = {
            'basic': {'interval_multiplier': 1.0, 'min_interval': 30},
            'enhanced': {'interval_multiplier': 0.5, 'min_interval': 15},
            'comprehensive': {'interval_multiplier': 0.33, 'min_interval': 10},
            'custom': {'interval_multiplier': 0.25, 'min_interval': 8}
        }
    
    def calculate_sampling_interval(self, fps, analysis_type, content_complexity=None):
        """샘플링 간격 계산"""
        strategy = self.sampling_strategies.get(analysis_type, self.sampling_strategies['enhanced'])
        
        base_interval = max(strategy['min_interval'], int(fps * strategy['interval_multiplier']))
        
        # 콘텐츠 복잡도에 따른 조정
        if content_complexity == 'high':
            base_interval = max(strategy['min_interval'] // 2, base_interval // 2)
        elif content_complexity == 'low':
            base_interval = min(int(fps), base_interval * 2)
        
        return base_interval
    
    def estimate_content_complexity(self, frame_sample):
        """콘텐츠 복잡도 추정"""
        try:
            # 에지 밀도로 복잡도 추정
            gray = cv2.cvtColor(frame_sample, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            if edge_density > 0.1:
                return 'high'
            elif edge_density < 0.05:
                return 'low'
            else:
                return 'medium'
        except Exception:
            return 'medium'

class EnhancedVideoAnalyzer:
    """고도화된 비디오 분석기"""
    
    def __init__(self, model_path="yolov8n.pt"):
        self.model = None
        self.par_recognizer = AdvancedPedestrianAttributeRecognizer()
        self.person_tracker = EnhancedPersonTracker()
        self.frame_sampler = AdaptiveFrameSampler()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 다중 모델 지원
        self.specialized_models = {}
        
        print(f"🚀 고도화된 PAR 비디오 분석기 초기화 (디바이스: {self.device})")
        
        # YOLO 모델 로딩
        if YOLO_AVAILABLE:
            try:
                self.model = YOLO(model_path)
                print(f"✅ 기본 YOLO 모델 로드: {model_path}")
                
                # 특화 모델들 로드 시도
                self._load_specialized_models()
            except Exception as e:
                print(f"⚠️ YOLO 로드 실패: {e}")
    
    def _load_specialized_models(self):
        """특화된 모델들 로드"""
        specialized_model_paths = {
            'person': 'yolov8n-person.pt',  # 사람 특화 (있다면)
            'vehicle': 'yolov8n-vehicle.pt'  # 차량 특화 (있다면)
        }
        
        for model_type, model_path in specialized_model_paths.items():
            try:
                if os.path.exists(model_path):
                    self.specialized_models[model_type] = YOLO(model_path)
                    print(f"✅ {model_type} 특화 모델 로드: {model_path}")
            except Exception as e:
                print(f"⚠️ {model_type} 특화 모델 로드 실패: {e}")
    
    def detect_and_analyze_persons_advanced(self, frame, frame_id, timestamp, context_info=None):
        """고도화된 보행자 검출 및 속성 분석"""
        if not self.model:
            return []
        
        try:
            # 다중 신뢰도 레벨로 검출
            confidence_levels = [0.25, 0.4, 0.6]
            all_detections = []
            
            for conf_level in confidence_levels:
                results = self.model(frame, verbose=False, conf=conf_level, classes=[0])  # person class만
                
                detections = []
                h, w = frame.shape[:2]
                
                for result in results:
                    if result.boxes is not None:
                        boxes = result.boxes.xyxy.cpu().numpy()
                        confidences = result.boxes.conf.cpu().numpy()
                        
                        for box, conf in zip(boxes, confidences):
                            # 바운딩 박스 정규화
                            normalized_bbox = [
                                float(box[0]/w), float(box[1]/h),
                                float(box[2]/w), float(box[3]/h)
                            ]
                            
                            # 보행자 영역 추출
                            person_crop = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                            
                            if person_crop.size > 0:
                                # 고도화된 속성 분석
                                attributes = self.par_recognizer.extract_detailed_attributes(
                                    person_crop, normalized_bbox, context_info
                                )
                                
                                detection = {
                                    'class': 'person',
                                    'bbox': normalized_bbox,
                                    'confidence': float(conf),
                                    'confidence_level': conf_level,
                                    'attributes': attributes,
                                    'frame_id': frame_id,
                                    'timestamp': timestamp,
                                    'crop_quality': self._assess_crop_quality(person_crop)
                                }
                                
                                detections.append(detection)
                
                # 중복 제거 후 최고 품질만 유지
                detections = self._deduplicate_detections(detections)
                all_detections.extend(detections)
            
            # 최종 중복 제거 및 품질 기반 필터링
            final_detections = self._filter_best_detections(all_detections)
            
            # 고도화된 추적 ID 할당
            tracked_detections = self.person_tracker.update_tracks_advanced(
                final_detections, frame_id, timestamp
            )
            
            return tracked_detections
            
        except Exception as e:
            print(f"⚠️ 고도화된 보행자 분석 오류: {e}")
            return []
    
    def _assess_crop_quality(self, person_crop):
        """보행자 crop 품질 평가"""
        try:
            h, w = person_crop.shape[:2]
            
            # 크기 점수
            size_score = min(1.0, (h * w) / (100 * 200))  # 100x200 기준
            
            # 선명도 점수 (Laplacian variance)
            gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(1.0, sharpness / 500)  # 정규화
            
            # 밝기 점수
            brightness = np.mean(person_crop)
            brightness_score = 1.0 - abs(brightness - 128) / 128  # 128 기준
            
            overall_quality = (size_score + sharpness_score + brightness_score) / 3
            
            return {
                'overall': float(overall_quality),
                'size': float(size_score),
                'sharpness': float(sharpness_score),
                'brightness': float(brightness_score)
            }
        except Exception:
            return {'overall': 0.5}
    
    def _deduplicate_detections(self, detections):
        """동일 신뢰도 레벨 내 중복 제거"""
        if not detections:
            return []
        
        # IoU 기반 중복 제거
        filtered_detections = []
        
        for detection in detections:
            is_duplicate = False
            
            for existing in filtered_detections:
                iou = self._calculate_iou(detection['bbox'], existing['bbox'])
                
                # IoU가 높고 신뢰도가 낮으면 제거
                if iou > 0.5:
                    if detection['confidence'] <= existing['confidence']:
                        is_duplicate = True
                        break
                    else:
                        # 더 높은 신뢰도면 기존 것 제거
                        filtered_detections.remove(existing)
            
            if not is_duplicate:
                filtered_detections.append(detection)
        
        return filtered_detections
    
    def _calculate_iou(self, box1, box2):
        """IoU 계산"""
        try:
            x1_max = max(box1[0], box2[0])
            y1_max = max(box1[1], box2[1])
            x2_min = min(box1[2], box2[2])
            y2_min = min(box1[3], box2[3])
            
            if x2_min <= x1_max or y2_min <= y1_max:
                return 0.0
            
            intersection = (x2_min - x1_max) * (y2_min - y1_max)
            
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union = area1 + area2 - intersection
            
            return intersection / union if union > 0 else 0.0
        except:
            return 0.0
    
    def _filter_best_detections(self, all_detections):
        """품질 기반 최고 검출 결과 필터링"""
        if not all_detections:
            return []
        
        # 품질 점수 기반 정렬
        scored_detections = []
        
        for detection in all_detections:
            quality = detection.get('crop_quality', {}).get('overall', 0.5)
            confidence = detection['confidence']
            
            # 종합 점수 계산
            composite_score = (confidence * 0.7) + (quality * 0.3)
            
            scored_detections.append({
                'detection': detection,
                'score': composite_score
            })
        
        # 점수순 정렬
        scored_detections.sort(key=lambda x: x['score'], reverse=True)
        
        # 상위 결과만 반환 (최대 20개)
        return [item['detection'] for item in scored_detections[:20]]
    
    def analyze_video_comprehensive_advanced(self, video, analysis_type='enhanced', progress_callback=None):
        """고도화된 종합 비디오 분석"""
        start_time = time.time()
        
        try:
            # 비디오 파일 경로 찾기
            video_path = self._find_video_path(video)
            if not video_path:
                raise Exception("비디오 파일을 찾을 수 없습니다")
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception("비디오 파일을 열 수 없습니다")
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            # 콘텐츠 복잡도 추정을 위한 샘플 프레임
            sample_frames = self._extract_sample_frames(cap, 5)
            content_complexity = self._estimate_overall_complexity(sample_frames)
            
            # 적응적 샘플링 간격 계산
            sample_interval = self.frame_sampler.calculate_sampling_interval(
                fps, analysis_type, content_complexity
            )
            
            print(f"📊 고도화된 분석 시작")
            print(f"   - 분석 타입: {analysis_type}")
            print(f"   - 콘텐츠 복잡도: {content_complexity}")
            print(f"   - 샘플링 간격: {sample_interval} 프레임")
            
            frame_results = []
            person_database = []
            scene_analysis = []
            processed_frames = 0
            frame_id = 0
            
            # 품질 메트릭 추적
            quality_metrics = {
                'total_detections': 0,
                'high_quality_detections': 0,
                'tracking_continuity': 0,
                'attribute_confidence_avg': 0
            }
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_id += 1
                
                # 더 많은 프레임을 처리하기 위해 샘플링 간격을 절반으로 줄임
                effective_interval = max(1, sample_interval // 2)
                if frame_id % effective_interval != 0:
                    continue
                
                timestamp = frame_id / fps
                
                try:
                    # 프레임별 컨텍스트 정보
                    context_info = {
                        'timestamp': timestamp,
                        'frame_id': frame_id,
                        'lighting_condition': self._analyze_lighting(frame),
                        'scene_complexity': self.frame_sampler.estimate_content_complexity(frame)
                    }
                    
                    # 고도화된 객체 검출 및 속성 분석 (모든 클래스 포함)
                    detected_objects = self.detect_and_analyze_persons_advanced(
                        frame, frame_id, timestamp, context_info
                    )
                    
                    # person과 non-person 객체 분리
                    detected_persons = [obj for obj in detected_objects if obj.get('class') == 'person']
                    other_objects = [obj for obj in detected_objects if obj.get('class') != 'person']
                    
                    # 씬 레벨 분석
                    scene_info = self._analyze_scene_advanced(frame, detected_persons, context_info)
                    
                    # 프레임 결과 저장 (모든 객체 포함)
                    frame_data = {
                        'image_id': frame_id,
                        'timestamp': timestamp,
                        'objects': detected_objects,  # 모든 객체 저장
                        'persons': detected_persons,  # 하위 호환성을 위해 유지
                        'other_objects': other_objects,  # person이 아닌 객체들
                        'person_count': len(detected_persons),
                        'object_count': len(detected_objects),
                        'scene_analysis': scene_info,
                        'context': context_info,
                        'quality_assessment': self._assess_frame_quality(frame, detected_persons)
                    }
                    
                    frame_results.append(frame_data)
                    person_database.extend(detected_objects)  # 모든 객체를 포함
                    scene_analysis.append(scene_info)
                    processed_frames += 1
                    
                    # 품질 메트릭 업데이트
                    self._update_quality_metrics(quality_metrics, detected_persons)
                    
                    # 진행률 콜백
                    if progress_callback and processed_frames % 5 == 0:
                        progress = (frame_id / total_frames) * 100
                        progress_callback(progress, f"고도화된 분석: {len(person_database)}명 검출")
                
                except Exception as e:
                    print(f"⚠️ 프레임 {frame_id} 분석 실패: {e}")
                    continue
            
            cap.release()
            
            # 고도화된 분석 결과 요약 생성
            analysis_summary = self._create_advanced_analysis_summary(
                person_database, scene_analysis, video, quality_metrics
            )
            
            processing_time = time.time() - start_time
            
            print(f"✅ 고도화된 분석 완료")
            print(f"   - 처리 시간: {processing_time:.1f}초")
            print(f"   - 검출된 인원: {len(person_database)}명")
            print(f"   - 품질 점수: {quality_metrics.get('overall_quality', 0):.2f}")
            
            return {
                'success': True,
                'video_summary': analysis_summary,
                'frame_results': frame_results,
                'person_database': person_database,
                'scene_analysis_summary': self._summarize_scene_analysis(scene_analysis),
                'quality_metrics': quality_metrics,
                'analysis_config': {
                    'method': 'Advanced_PAR_Analysis',
                    'analysis_type': analysis_type,
                    'content_complexity': content_complexity,
                    'sampling_interval': sample_interval,
                    'processing_time': processing_time,
                    'total_persons_detected': len(person_database),
                    'frames_analyzed': processed_frames
                },
                'total_frames_analyzed': processed_frames
            }
            
        except Exception as e:
            print(f"❌ 고도화된 분석 실패: {e}")
            return {'success': False, 'error': str(e)}
    
    def _extract_sample_frames(self, cap, num_samples=5):
        """샘플 프레임 추출"""
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_indices = np.linspace(0, total_frames-1, num_samples, dtype=int)
        
        sample_frames = []
        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                sample_frames.append(frame)
        
        # 원래 위치로 되돌리기
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return sample_frames
    
    def _estimate_overall_complexity(self, sample_frames):
        """전체 콘텐츠 복잡도 추정"""
        if not sample_frames:
            return 'medium'
        
        complexities = []
        for frame in sample_frames:
            complexity = self.frame_sampler.estimate_content_complexity(frame)
            complexities.append(complexity)
        
        # 가장 빈번한 복잡도 반환
        complexity_counts = Counter(complexities)
        return complexity_counts.most_common(1)[0][0]
    
    def _analyze_lighting(self, frame):
        """조명 조건 분석"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            avg_brightness = np.mean(gray)
            
            if avg_brightness > 150:
                return 'bright'
            elif avg_brightness < 80:
                return 'dark'
            else:
                return 'normal'
        except:
            return 'normal'
    
    def _analyze_scene_advanced(self, frame, detected_persons, context_info):
        """고도화된 씬 분석"""
        try:
            scene_info = {
                'lighting': context_info.get('lighting_condition', 'normal'),
                'complexity': context_info.get('scene_complexity', 'medium'),
                'person_count': len(detected_persons),
                'person_density': self._calculate_person_density(frame, detected_persons),
                'dominant_colors': self._analyze_dominant_colors(frame),
                'activity_level': self._estimate_activity_level(detected_persons),
                'scene_type': self._classify_scene_type(frame, detected_persons)
            }
            
            return scene_info
        except Exception as e:
            print(f"⚠️ 씬 분석 오류: {e}")
            return {}
    
    def _calculate_person_density(self, frame, detected_persons):
        """인원 밀도 계산"""
        if not detected_persons:
            return 0.0
        
        h, w = frame.shape[:2]
        frame_area = h * w
        
        total_person_area = 0
        for person in detected_persons:
            bbox = person['bbox']
            person_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) * w * h
            total_person_area += person_area
        
        return total_person_area / frame_area
    
    def _analyze_dominant_colors(self, frame):
        """주요 색상 분석"""
        try:
            # 이미지를 작게 리사이즈
            small_frame = cv2.resize(frame, (150, 150))
            
            # K-means로 주요 색상 추출
            data = small_frame.reshape((-1, 3))
            data = np.float32(data)
            
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
            _, labels, centers = cv2.kmeans(data, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            # BGR을 색상명으로 변환
            color_names = []
            for center in centers:
                color_name = self._bgr_to_color_name(center)
                color_names.append(color_name)
            
            return color_names
        except:
            return ['unknown']
    
    def _bgr_to_color_name(self, bgr):
        """BGR 값을 색상명으로 변환"""
        b, g, r = bgr
        
        if r > 150 and g < 100 and b < 100:
            return 'red'
        elif g > 150 and r < 100 and b < 100:
            return 'green'
        elif b > 150 and r < 100 and g < 100:
            return 'blue'
        elif r > 200 and g > 200 and b > 200:
            return 'white'
        elif r < 50 and g < 50 and b < 50:
            return 'black'
        elif r > 150 and g > 150 and b < 100:
            return 'yellow'
        else:
            return 'mixed'
    
    def _estimate_activity_level(self, detected_persons):
        """활동 수준 추정"""
        if not detected_persons:
            return 'none'
        
        activity_scores = []
        for person in detected_persons:
            posture = person.get('attributes', {}).get('posture', {}).get('value', '')
            
            if 'running' in posture:
                activity_scores.append(3)
            elif 'walking' in posture:
                activity_scores.append(2)
            elif 'standing' in posture:
                activity_scores.append(1)
            else:
                activity_scores.append(0)
        
        if not activity_scores:
            return 'low'
        
        avg_activity = np.mean(activity_scores)
        
        if avg_activity > 2:
            return 'high'
        elif avg_activity > 1:
            return 'medium'
        else:
            return 'low'
    
    def _classify_scene_type(self, frame, detected_persons):
        """씬 타입 분류"""
        person_count = len(detected_persons)
        
        if person_count == 0:
            return 'empty'
        elif person_count == 1:
            return 'individual'
        elif person_count <= 3:
            return 'small_group'
        elif person_count <= 10:
            return 'medium_group'
        else:
            return 'crowd'
    
    def _assess_frame_quality(self, frame, detected_persons):
        """프레임 품질 평가"""
        try:
            # 전체적인 품질 지표
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 선명도
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # 밝기 분포
            brightness_std = np.std(gray)
            
            # 검출 품질
            detection_quality = 0
            if detected_persons:
                quality_scores = [p.get('crop_quality', {}).get('overall', 0) for p in detected_persons]
                detection_quality = np.mean(quality_scores)
            
            overall_quality = (
                min(1.0, sharpness / 500) * 0.4 +
                min(1.0, brightness_std / 50) * 0.3 +
                detection_quality * 0.3
            )
            
            return {
                'overall': float(overall_quality),
                'sharpness': float(sharpness),
                'brightness_distribution': float(brightness_std),
                'detection_quality': float(detection_quality)
            }
        except:
            return {'overall': 0.5}
    
    def _update_quality_metrics(self, quality_metrics, detected_persons):
        """품질 메트릭 업데이트"""
        quality_metrics['total_detections'] += len(detected_persons)
        
        high_quality_count = sum(1 for p in detected_persons 
                               if p.get('crop_quality', {}).get('overall', 0) > 0.7)
        quality_metrics['high_quality_detections'] += high_quality_count
        
        # 속성 신뢰도 평균 계산
        confidence_scores = []
        for person in detected_persons:
            attrs = person.get('attributes', {})
            for attr_name, attr_data in attrs.items():
                if isinstance(attr_data, dict) and 'confidence' in attr_data:
                    confidence_scores.append(attr_data['confidence'])
        
        if confidence_scores:
            current_avg = quality_metrics.get('attribute_confidence_avg', 0)
            new_avg = (current_avg + np.mean(confidence_scores)) / 2
            quality_metrics['attribute_confidence_avg'] = new_avg
    
    def _create_advanced_analysis_summary(self, person_database, scene_analysis, video, quality_metrics):
        """고도화된 분석 결과 요약"""
        if not person_database:
            return {'message': '검출된 보행자가 없습니다'}
        
        # 고유 인물 추출
        unique_persons = {}
        for person in person_database:
            track_id = person.get('track_id')
            if track_id and track_id not in unique_persons:
                unique_persons[track_id] = person
        
        # 상세 속성 통계
        detailed_stats = self._calculate_detailed_statistics(unique_persons.values())
        
        # 시간대별 분석
        temporal_analysis = self._analyze_temporal_patterns(person_database)
        
        # 씬 다양성 분석
        scene_diversity = self._analyze_scene_diversity(scene_analysis)
        
        # 품질 평가
        overall_quality = self._calculate_overall_quality(quality_metrics)
        
        return {
            'total_detections': len(person_database),
            'unique_persons': len(unique_persons),
            'detailed_attribute_statistics': detailed_stats,
            'temporal_analysis': temporal_analysis,
            'scene_diversity': scene_diversity,
            'quality_assessment': overall_quality,
            'analysis_type': 'advanced_par_analysis',
            'key_insights': self._generate_key_insights(detailed_stats, temporal_analysis, scene_diversity)
        }
    
    def _calculate_detailed_statistics(self, unique_persons):
        """상세 속성 통계 계산"""
        stats = defaultdict(lambda: defaultdict(int))
        
        for person in unique_persons:
            attributes = person.get('attributes', {})
            
            for attr_name, attr_data in attributes.items():
                if isinstance(attr_data, dict) and attr_data.get('confidence', 0) > 0.3:
                    value = attr_data['value']
                    stats[attr_name][value] += 1
        
        # 상위 결과만 유지
        filtered_stats = {}
        for attr_name, values in stats.items():
            if values:
                # 가장 빈번한 상위 5개
                top_values = dict(sorted(values.items(), key=lambda x: x[1], reverse=True)[:5])
                filtered_stats[attr_name] = top_values
        
        return filtered_stats
    
    def _analyze_temporal_patterns(self, person_database):
        """시간적 패턴 분석"""
        if not person_database:
            return {}
        
        # 시간대별 인원 수
        time_buckets = defaultdict(int)
        
        for person in person_database:
            timestamp = person.get('timestamp', 0)
            time_bucket = int(timestamp // 10) * 10  # 10초 단위
            time_buckets[time_bucket] += 1
        
        # 피크 시간 찾기
        if time_buckets:
            peak_time = max(time_buckets.items(), key=lambda x: x[1])
            avg_count = np.mean(list(time_buckets.values()))
            
            return {
                'peak_time_seconds': peak_time[0],
                'peak_person_count': peak_time[1],
                'average_person_count': round(avg_count, 2),
                'total_time_span': max(time_buckets.keys()) - min(time_buckets.keys()),
                'activity_distribution': dict(time_buckets)
            }
        
        return {}
    
    def _analyze_scene_diversity(self, scene_analysis):
        """씬 다양성 분석"""
        if not scene_analysis:
            return {}
        
        scene_types = [scene.get('scene_type', 'unknown') for scene in scene_analysis]
        activity_levels = [scene.get('activity_level', 'unknown') for scene in scene_analysis]
        lighting_conditions = [scene.get('lighting', 'unknown') for scene in scene_analysis]
        
        return {
            'scene_type_distribution': dict(Counter(scene_types)),
            'activity_level_distribution': dict(Counter(activity_levels)),
            'lighting_distribution': dict(Counter(lighting_conditions)),
            'diversity_score': len(set(scene_types)) / max(len(scene_types), 1)
        }
    
    def _calculate_overall_quality(self, quality_metrics):
        """전체 품질 점수 계산"""
        total_detections = quality_metrics.get('total_detections', 0)
        high_quality_detections = quality_metrics.get('high_quality_detections', 0)
        
        if total_detections == 0:
            return {'overall_score': 0.0, 'status': 'no_data'}
        
        quality_ratio = high_quality_detections / total_detections
        confidence_avg = quality_metrics.get('attribute_confidence_avg', 0)
        
        overall_score = (quality_ratio * 0.6) + (confidence_avg * 0.4)
        
        if overall_score > 0.8:
            status = 'excellent'
        elif overall_score > 0.6:
            status = 'good'
        elif overall_score > 0.4:
            status = 'fair'
        else:
            status = 'poor'
        
        return {
            'overall_score': round(overall_score, 3),
            'status': status,
            'quality_ratio': round(quality_ratio, 3),
            'confidence_average': round(confidence_avg, 3)
        }
    
    def _generate_key_insights(self, detailed_stats, temporal_analysis, scene_diversity):
        """핵심 인사이트 생성"""
        insights = []
        
        # 인구통계학적 인사이트
        if 'gender' in detailed_stats:
            gender_stats = detailed_stats['gender']
            if gender_stats:
                dominant_gender = max(gender_stats.items(), key=lambda x: x[1])[0]
                insights.append(f"주요 성별: {dominant_gender}")
        
        # 시간적 인사이트
        if temporal_analysis and 'peak_time_seconds' in temporal_analysis:
            peak_time = temporal_analysis['peak_time_seconds']
            insights.append(f"최대 활동 시간: {peak_time}초 지점")
        
        # 활동 인사이트
        if scene_diversity and 'activity_level_distribution' in scene_diversity:
            activity_dist = scene_diversity['activity_level_distribution']
            if activity_dist:
                dominant_activity = max(activity_dist.items(), key=lambda x: x[1])[0]
                insights.append(f"주요 활동 수준: {dominant_activity}")
        
        return insights
    
    def _summarize_scene_analysis(self, scene_analysis):
        """씬 분석 요약"""
        if not scene_analysis:
            return {}
        
        return {
            'total_scenes_analyzed': len(scene_analysis),
            'average_person_density': np.mean([s.get('person_density', 0) for s in scene_analysis]),
            'scene_complexity_distribution': dict(Counter([s.get('complexity', 'unknown') for s in scene_analysis])),
            'lighting_conditions': dict(Counter([s.get('lighting', 'unknown') for s in scene_analysis]))
        }
    
    # 기존 메서드들 (하위 호환성)
    def analyze_video_comprehensive(self, video, analysis_type='enhanced', progress_callback=None):
        """기존 메서드 호환성 유지"""
        return self.analyze_video_comprehensive_advanced(video, analysis_type, progress_callback)
    
    def detect_and_analyze_persons(self, frame, frame_id):
        """기존 메서드 호환성 유지"""
        timestamp = frame_id / 30.0  # 30fps 가정
        return self.detect_and_analyze_persons_advanced(frame, frame_id, timestamp)
    
    # 기존 유틸리티 메서드들 유지
    def _find_video_path(self, video):
        """비디오 파일 경로 찾기"""
        from django.conf import settings
        
        possible_paths = [
            os.path.join(settings.MEDIA_ROOT, 'videos', video.filename),
            os.path.join(settings.MEDIA_ROOT, 'uploads', video.filename),
            getattr(video, 'file_path', None)
        ]
        
        for path in [p for p in possible_paths if p]:
            if os.path.exists(path):
                return path
        
        return None

# 전역 인스턴스 관리
_global_enhanced_video_analyzer = None

def get_video_analyzer():
    """고도화된 비디오 분석기 인스턴스 반환"""
    global _global_enhanced_video_analyzer
    if _global_enhanced_video_analyzer is None:
        _global_enhanced_video_analyzer = EnhancedVideoAnalyzer()
    return _global_enhanced_video_analyzer

def get_analyzer_status():
    """분석기 상태 반환"""
    analyzer = get_video_analyzer()
    return {
        'status': 'enhanced' if analyzer.model else 'limited',
        'features': {
            'yolo': analyzer.model is not None,
            'specialized_models': len(analyzer.specialized_models),
            'clip': CLIP_AVAILABLE,
            'advanced_par': True,
            'adaptive_sampling': True,
            'quality_assessment': True
        },
        'device': analyzer.device,
        'analysis_modes': ['basic', 'enhanced', 'comprehensive', 'custom']
    }

# 호환성을 위한 클래스명 별칭
VideoAnalyzer = EnhancedVideoAnalyzer