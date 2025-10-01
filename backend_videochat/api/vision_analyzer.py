# vision_analyzer.py
import os
import cv2
import numpy as np
from PIL import Image
import torch
from typing import List, Dict, Any, Optional
import base64
import io

class VisionAnalyzer:
    """CLIP을 사용한 이미지 분석 클래스"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.preprocess = None
        self._initialize_model()
    
    def _initialize_model(self):
        """CLIP 모델 초기화"""
        try:
            import clip
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
            print("✅ CLIP 모델 로드 완료")
        except ImportError:
            print("⚠️ CLIP이 설치되지 않음 - 기본 이미지 분석 사용")
            self.model = None
        except Exception as e:
            print(f"⚠️ CLIP 모델 로드 실패: {e}")
            self.model = None
    
    def analyze_frame(self, image_path: str) -> Dict[str, Any]:
        """단일 프레임 이미지 분석"""
        if not os.path.exists(image_path):
            return {"error": "이미지 파일을 찾을 수 없습니다"}
        
        try:
            # 이미지 로드
            image = Image.open(image_path).convert("RGB")
            
            if self.model is not None:
                return self._analyze_with_clip(image, image_path)
            else:
                return self._analyze_with_basic_cv(image, image_path)
                
        except Exception as e:
            return {"error": f"이미지 분석 실패: {str(e)}"}
    
    def _analyze_with_clip(self, image: Image.Image, image_path: str) -> Dict[str, Any]:
        """CLIP을 사용한 이미지 분석"""
        try:
            import clip
            # 이미지 전처리
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # 시각적 특징 추출
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # 텍스트 프롬프트들 (야외 시장/거리 우선)
            text_prompts = [
                # 야외 시장/거리 관련 (우선순위 높음)
                "outdoor market scene with colorful umbrellas and people",
                "street market with vendors and colorful stalls", 
                "outdoor commercial street with shops",
                "people walking on outdoor street",
                "outdoor shopping area with stalls",
                "street vendors with colorful umbrellas and tables",
                "outdoor market with colorful stalls and people",
                "people sitting at outdoor tables with umbrellas",
                "outdoor dining area with colorful umbrellas",
                "street scene with outdoor shops and people",
                "outdoor activity with colorful elements",
                "outdoor scene with people and stalls",
                "outdoor gathering with colorful umbrellas",
                "outdoor market with colorful vendors",
                "street scene with outdoor activities",
                
                # 실내 쇼핑몰 관련
                "people walking in a shopping mall",
                "people walking in a store",
                "people walking in an indoor space",
                "shopping mall interior",
                "store interior with people",
                "people walking in a corridor",
                "indoor commercial space",
                "people walking in a building",
                "shopping center with people",
                
                # 기타 장면들
                "people walking in a park",
                "people sitting and talking",
                "children playing",
                "elderly people resting",
                "crowded scene with many people",
                "quiet and peaceful scene",
                "people wearing colorful clothes",
                "people wearing red clothes",
                "people wearing blue clothes",
                "people wearing green clothes",
                "indoor activity",
                "group of people",
                "single person",
                "family gathering",
                "friends meeting"
            ]
            
            # 텍스트 인코딩
            text_tokens = clip.tokenize(text_prompts).to(self.device)
            with torch.no_grad():
                text_features = self.model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # 유사도 계산
            similarities = (image_features @ text_features.T).squeeze(0)
            similarities = similarities.cpu().numpy()
            
            # 상위 5개 매칭 결과
            top_indices = similarities.argsort()[-5:][::-1]
            top_matches = []
            
            for idx in top_indices:
                if similarities[idx] > 0.2:  # 임계값
                    top_matches.append({
                        "description": text_prompts[idx],
                        "confidence": float(similarities[idx])
                    })
            
            # 기본 시각적 정보
            width, height = image.size
            dominant_colors = self._extract_dominant_colors(image)
            
            return {
                "analysis_type": "clip",
                "image_path": image_path,
                "image_size": {"width": width, "height": height},
                "dominant_colors": dominant_colors,
                "scene_descriptions": top_matches,
                "overall_scene": self._generate_scene_description(top_matches),
                "confidence": float(similarities.max())
            }
            
        except Exception as e:
            return {"error": f"CLIP 분석 실패: {str(e)}"}
    
    def _analyze_with_basic_cv(self, image: Image.Image, image_path: str) -> Dict[str, Any]:
        """기본 OpenCV를 사용한 이미지 분석"""
        try:
            # PIL을 OpenCV 형식으로 변환
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # 기본 정보
            height, width = cv_image.shape[:2]
            
            # 색상 분석
            dominant_colors = self._extract_dominant_colors(image)
            
            # 밝기 분석
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            
            # 대비 분석
            contrast = np.std(gray)
            
            # 간단한 장면 분류
            scene_type = self._classify_scene_basic(cv_image, brightness, contrast)
            
            return {
                "analysis_type": "basic_cv",
                "image_path": image_path,
                "image_size": {"width": width, "height": height},
                "dominant_colors": dominant_colors,
                "brightness": float(brightness),
                "contrast": float(contrast),
                "scene_type": scene_type,
                "scene_descriptions": [{"description": scene_type, "confidence": 0.7}],
                "overall_scene": scene_type
            }
            
        except Exception as e:
            return {"error": f"기본 분석 실패: {str(e)}"}
    
    def _extract_dominant_colors(self, image: Image.Image) -> List[str]:
        """이미지에서 주요 색상 추출"""
        try:
            # 이미지를 작은 크기로 리사이즈
            image_small = image.resize((150, 150))
            
            # RGB 배열로 변환
            data = np.array(image_small)
            data = data.reshape((-1, 3))
            
            # K-means로 주요 색상 추출
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=5, random_state=42)
            kmeans.fit(data)
            
            colors = kmeans.cluster_centers_.astype(int)
            color_names = []
            
            for color in colors:
                r, g, b = color
                color_name = self._rgb_to_color_name(r, g, b)
                color_names.append(color_name)
            
            return color_names
            
        except Exception as e:
            return ["unknown"]
    
    def _rgb_to_color_name(self, r: int, g: int, b: int) -> str:
        """RGB 값을 색상 이름으로 변환"""
        # 간단한 색상 분류
        if r > 200 and g < 100 and b < 100:
            return "red"
        elif r < 100 and g > 200 and b < 100:
            return "green"
        elif r < 100 and g < 100 and b > 200:
            return "blue"
        elif r > 200 and g > 200 and b < 100:
            return "yellow"
        elif r > 200 and g < 100 and b > 200:
            return "purple"
        elif r < 100 and g > 200 and b > 200:
            return "cyan"
        elif r > 200 and g > 200 and b > 200:
            return "white"
        elif r < 100 and g < 100 and b < 100:
            return "black"
        elif r > 150 and g > 150 and b > 150:
            return "light"
        else:
            return "mixed"
    
    def _classify_scene_basic(self, cv_image, brightness: float, contrast: float) -> str:
        """기본적인 장면 분류"""
        if brightness > 150:
            return "bright outdoor scene"
        elif brightness < 80:
            return "dark indoor scene"
        elif contrast > 50:
            return "high contrast scene with clear details"
        else:
            return "low contrast scene"
    
    def _generate_scene_description(self, matches: List[Dict]) -> str:
        """매칭 결과를 바탕으로 장면 설명 생성"""
        if not matches:
            return "unknown scene"
        
        descriptions = [match["description"] for match in matches[:3]]
        
        # 야외 시장/거리 관련 우선 인식 (새로운 프롬프트 포함)
        if "outdoor market scene with colorful umbrellas and people" in descriptions:
            return "다양한 색깔의 파라솔이 있는 공간에서 사람들이 활동하는 장면"
        elif "street market with vendors and colorful stalls" in descriptions:
            return "다양한 색깔의 상점이 있는 환경에서 상인과 고객들이 있는 장면"
        elif "outdoor commercial street with shops" in descriptions:
            return "상점들이 있는 상업 공간에서의 장면"
        elif "people walking on outdoor street" in descriptions:
            return "야외 공간에서 사람들이 걸어다니는 장면"
        elif "outdoor shopping area with stalls" in descriptions:
            return "상점들이 있는 야외 공간에서의 장면"
        elif "street vendors with colorful umbrellas and tables" in descriptions:
            return "다양한 색깔의 파라솔과 테이블이 있는 환경에서 상인들의 장면"
        elif "outdoor market with colorful stalls and people" in descriptions:
            return "다양한 색깔의 상점과 사람들이 있는 야외 공간 장면"
        elif "people sitting at outdoor tables with umbrellas" in descriptions:
            return "파라솔이 있는 야외 테이블에 앉아 있는 사람들의 장면"
        elif "outdoor dining area with colorful umbrellas" in descriptions:
            return "다양한 색깔의 파라솔이 있는 야외 식사 공간에서의 장면"
        elif "street scene with outdoor shops and people" in descriptions:
            return "야외 상점들과 사람들이 있는 공간의 장면"
        elif "outdoor activity with colorful elements" in descriptions:
            return "다양한 색깔의 요소가 있는 야외 활동 장면"
        elif "outdoor scene with people and stalls" in descriptions:
            return "사람들과 상점들이 있는 야외 장면"
        elif "outdoor gathering with colorful umbrellas" in descriptions:
            return "다양한 색깔의 파라솔이 있는 야외 모임 장면"
        elif "outdoor market with colorful vendors" in descriptions:
            return "다양한 색깔의 상인들이 있는 야외 공간 장면"
        elif "street scene with outdoor activities" in descriptions:
            return "야외 활동들이 있는 공간의 장면"
        elif "outdoor market scene with people" in descriptions:
            return "야외 공간에서 사람들이 활동하는 장면"
        elif "street market with vendors and customers" in descriptions:
            return "공간에서 상인과 고객들이 있는 장면"
        elif "outdoor commercial street" in descriptions:
            return "야외 상업 공간에서의 장면"
        elif "people walking on the street" in descriptions:
            return "공간에서 사람들이 걸어다니는 장면"
        elif "outdoor shopping area" in descriptions:
            return "야외 쇼핑 공간에서의 장면"
        elif "street vendors with colorful umbrellas" in descriptions:
            return "다양한 색깔의 파라솔이 있는 환경에서 상인들의 장면"
        elif "outdoor market with stalls" in descriptions:
            return "야외 시장의 상점들이 있는 장면"
        elif "people sitting at outdoor tables" in descriptions:
            return "야외 테이블에 앉아 있는 사람들의 장면"
        elif "outdoor dining area" in descriptions:
            return "야외 식사 공간에서의 장면"
        elif "street scene with shops" in descriptions:
            return "상점들이 있는 거리 장면"
        elif "outdoor activity" in descriptions:
            return "야외 활동 장면"
        elif "outdoor scene" in descriptions:
            return "야외 장면"
        elif "outdoor gathering" in descriptions:
            return "야외 모임 장면"
        elif "outdoor market" in descriptions:
            return "야외 시장 장면"
        elif "street scene" in descriptions:
            return "거리 장면"
        
        # 실내 쇼핑몰 관련
        elif "people walking in a shopping mall" in descriptions:
            return "상업 공간에서 사람들이 걸어다니는 장면"
        elif "people walking in a store" in descriptions:
            return "상점에서 사람들이 걸어다니는 장면"
        elif "shopping mall interior" in descriptions:
            return "실내 상업 공간의 장면"
        elif "store interior with people" in descriptions:
            return "상점 내부에서 사람들이 활동하는 장면"
        elif "people walking in a corridor" in descriptions:
            return "통로에서 사람들이 걸어다니는 장면"
        elif "indoor commercial space" in descriptions:
            return "실내 상업 공간에서의 장면"
        elif "people walking in a building" in descriptions:
            return "실내 공간에서 사람들이 걸어다니는 장면"
        elif "shopping center with people" in descriptions:
            return "상업 공간에서 사람들이 활동하는 장면"
        
        # 기타 장면들
        elif "people walking in a park" in descriptions:
            return "공원에서 사람들이 산책하고 있는 장면"
        elif "people sitting and talking" in descriptions:
            return "사람들이 앉아서 대화를 나누는 장면"
        elif "children playing" in descriptions:
            return "어린이들이 놀고 있는 장면"
        elif "elderly people resting" in descriptions:
            return "노인들이 휴식을 취하는 장면"
        elif "crowded scene with many people" in descriptions:
            return "많은 사람들이 모여 있는 활발한 장면"
        elif "quiet and peaceful scene" in descriptions:
            return "조용하고 평화로운 장면"
        else:
            return "다양한 활동이 이루어지는 장면"
    
    def analyze_video_frames(self, video_id: int, frame_paths: List[str]) -> Dict[str, Any]:
        """비디오의 여러 프레임들을 분석"""
        if not frame_paths:
            return {"error": "분석할 프레임이 없습니다"}
        
        frame_analyses = []
        scene_descriptions = []
        
        for frame_path in frame_paths:
            analysis = self.analyze_frame(frame_path)
            if "error" not in analysis:
                frame_analyses.append(analysis)
                if "overall_scene" in analysis:
                    scene_descriptions.append(analysis["overall_scene"])
        
        if not frame_analyses:
            return {"error": "모든 프레임 분석 실패"}
        
        # 전체 비디오 분석 결과
        all_colors = []
        all_scenes = []
        
        for analysis in frame_analyses:
            if "dominant_colors" in analysis:
                all_colors.extend(analysis["dominant_colors"])
            if "overall_scene" in analysis:
                all_scenes.append(analysis["overall_scene"])
        
        # 가장 많이 나타나는 색상과 장면
        from collections import Counter
        common_colors = Counter(all_colors).most_common(3)
        common_scenes = Counter(all_scenes).most_common(3)
        
        return {
            "video_id": video_id,
            "total_frames_analyzed": len(frame_analyses),
            "frame_analyses": frame_analyses,
            "common_colors": [color[0] for color in common_colors],
            "common_scenes": [scene[0] for scene in common_scenes],
            "overall_description": self._generate_video_description(common_scenes, common_colors)
        }
    
    def _generate_video_description(self, common_scenes: List, common_colors: List) -> str:
        """전체 비디오 설명 생성"""
        if not common_scenes:
            return "분석할 수 없는 장면들"
        
        main_scene = common_scenes[0][0] if common_scenes else "일반적인 장면"
        main_colors = [color[0] for color in common_colors[:3]] if common_colors else []
        
        description = f"이 비디오는 주로 {main_scene}을 보여줍니다."
        
        if main_colors:
            color_text = ", ".join(main_colors)
            description += f" 주요 색상은 {color_text}입니다."
        
        return description

# 전역 인스턴스
try:
    vision_analyzer = VisionAnalyzer()
except Exception as e:
    print(f"⚠️ Vision Analyzer 초기화 실패: {e}")
    vision_analyzer = None
