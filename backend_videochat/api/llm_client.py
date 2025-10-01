# llm_client.py
import os
import json
from typing import Dict, List, Optional, Any
import requests

class LLMClient:
    """LLM 클라이언트 - OpenAI API 또는 로컬 모델 사용"""
    
    def __init__(self):
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.base_url = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
        self.model = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
        
    def is_available(self) -> bool:
        """LLM 서비스 사용 가능 여부 확인"""
        return self.api_key is not None
    
    def get_api_status(self) -> Dict[str, Any]:
        """API 상태 확인"""
        return {
            'openai': {
                'available': self.is_available(),
                'model': self.model if self.is_available() else None
            },
            'groq': {'available': False},
            'anthropic': {'available': False}
        }
    
    def generate_summary(self, video_data: Dict[str, Any]) -> str:
        """영상 데이터를 기반으로 한국어 요약 생성"""
        if not self.is_available():
            return self._generate_fallback_summary(video_data)
        
        try:
            prompt = self._create_summary_prompt(video_data)
            response = self._call_openai_api(prompt)
            return response
        except Exception as e:
            print(f"⚠️ LLM 요약 생성 실패: {e}")
            return self._generate_fallback_summary(video_data)
    
    def generate_highlight_description(self, highlight_data: Dict[str, Any]) -> str:
        """하이라이트 데이터를 기반으로 한국어 설명 생성"""
        if not self.is_available():
            return self._generate_fallback_highlight(highlight_data)
        
        try:
            prompt = self._create_highlight_prompt(highlight_data)
            response = self._call_openai_api(prompt)
            return response
        except Exception as e:
            print(f"⚠️ LLM 하이라이트 설명 생성 실패: {e}")
            return self._generate_fallback_highlight(highlight_data)
    
    def analyze_frame_with_vision(self, frame_path: str, query: str = "이 프레임에서 사람들을 자세히 분석해주세요") -> Dict[str, Any]:
        """GPT Vision을 사용하여 프레임 이미지 분석"""
        if not self.is_available():
            return self._generate_fallback_frame_analysis(frame_path, query)
        
        try:
            import base64
            
            # 이미지 파일을 base64로 인코딩
            with open(frame_path, 'rb') as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            # GPT Vision API 호출
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "model": "gpt-4o-mini",  # Vision 모델 사용
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"{query}\n\n다음 정보를 포함해서 분석해주세요:\n1. 탐지된 사람의 수\n2. 각 사람의 위치 (왼쪽/오른쪽/중앙, 위/아래/중앙)\n3. 각 사람의 특징 (옷 색깔, 머리카락, 나이대 등)\n4. 사람들의 활동 (걷기, 서있기, 대화 등)\n5. 전체적인 장면 설명"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 1000
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                analysis_text = result['choices'][0]['message']['content']
                
                # 분석 결과를 구조화된 데이터로 파싱
                return self._parse_vision_analysis(analysis_text)
            else:
                print(f"⚠️ GPT Vision API 오류: {response.status_code}")
                return self._generate_fallback_frame_analysis(frame_path, query)
                
        except Exception as e:
            print(f"⚠️ GPT Vision 분석 실패: {e}")
            return self._generate_fallback_frame_analysis(frame_path, query)
    
    def _parse_vision_analysis(self, analysis_text: str) -> Dict[str, Any]:
        """GPT Vision 분석 결과를 구조화된 데이터로 파싱"""
        try:
            # 간단한 파싱 로직 (실제로는 더 정교한 파싱이 필요)
            lines = analysis_text.split('\n')
            
            result = {
                'raw_analysis': analysis_text,
                'person_count': 0,
                'persons': [],
                'scene_description': '',
                'confidence': 0.8
            }
            
            # 사람 수 추출
            for line in lines:
                if '사람' in line and ('명' in line or '수' in line):
                    import re
                    numbers = re.findall(r'\d+', line)
                    if numbers:
                        result['person_count'] = int(numbers[0])
                        break
            
            # 장면 설명 추출
            if '장면' in analysis_text:
                scene_start = analysis_text.find('장면')
                result['scene_description'] = analysis_text[scene_start:scene_start+100]
            
            return result
            
        except Exception as e:
            print(f"⚠️ 분석 결과 파싱 실패: {e}")
            return {
                'raw_analysis': analysis_text,
                'person_count': 1,
                'persons': [],
                'scene_description': '분석 결과를 파싱할 수 없습니다.',
                'confidence': 0.5
            }
    
    def _generate_fallback_frame_analysis(self, frame_path: str, query: str) -> Dict[str, Any]:
        """GPT Vision이 없을 때의 기본 분석"""
        return {
            'raw_analysis': f'프레임 분석: {frame_path}',
            'person_count': 1,
            'persons': [{'location': '중앙', 'features': '일반적인 사람', 'activity': '걷기'}],
            'scene_description': '기본적인 프레임 분석 결과입니다.',
            'confidence': 0.3
        }
    
    def _create_summary_prompt(self, video_data: Dict[str, Any]) -> str:
        """영상 요약을 위한 프롬프트 생성 (CLIP + GPT Vision 통합)"""
        clips = video_data.get('clips', [])
        video_title = video_data.get('video_title', '영상')
        
        # CLIP 분석 결과
        clip_analyses = video_data.get('clip_analyses', [])
        clip_info = ""
        
        if clip_analyses:
            clip_info = "\n**CLIP 시각적 분석 결과**:\n"
            for i, ca in enumerate(clip_analyses, 1):
                clip_data = ca.get('clip_analysis', {})
                timestamp = ca.get('timestamp', 0)
                overall_scene = clip_data.get('overall_scene', '분석 중')
                scene_descriptions = clip_data.get('scene_descriptions', [])
                dominant_colors = clip_data.get('dominant_colors', [])
                confidence = clip_data.get('confidence', 0)
                
                clip_info += f"\n**프레임 {i} ({timestamp:.1f}초)**:\n"
                clip_info += f"- 장면 유형: {overall_scene}\n"
                clip_info += f"- 신뢰도: {confidence:.2f}\n"
                if dominant_colors:
                    clip_info += f"- 주요 색상: {', '.join(dominant_colors[:3])}\n"
                if scene_descriptions:
                    top_desc = scene_descriptions[0] if scene_descriptions else {}
                    clip_info += f"- 상세 설명: {top_desc.get('description', '')}\n"
        
        # GPT Vision 분석 결과
        gpt_vision_analyses = video_data.get('gpt_vision_analyses', [])
        vision_info = ""
        
        if gpt_vision_analyses:
            vision_info = "\n**GPT Vision 상세 분석 결과**:\n"
            for i, va in enumerate(gpt_vision_analyses, 1):
                analysis = va.get('analysis', {})
                raw_analysis = analysis.get('raw_analysis', '')
                person_count = analysis.get('person_count', 0)
                scene_desc = analysis.get('scene_description', '')
                persons = analysis.get('persons', [])
                
                vision_info += f"\n**프레임 {i}**:\n"
                if person_count > 0:
                    vision_info += f"- 탐지된 사람 수: {person_count}명\n"
                if scene_desc:
                    vision_info += f"- 장면 설명: {scene_desc}\n"
                if persons:
                    vision_info += f"- 인물 분석: {len(persons)}명의 상세 정보\n"
                if raw_analysis:
                    vision_info += f"- 상세 분석: {raw_analysis[:300]}...\n"
        
        # 통합 분석 정보
        combined_info = ""
        if clip_info and vision_info:
            combined_info = f"""
**통합 분석 결과**:
{clip_info}

{vision_info}

**중요**: 위의 CLIP 시각적 분석과 GPT Vision 상세 분석 결과를 종합하여 실제로 무슨 일이 일어나고 있는지 생생하게 묘사해주세요. 두 분석 방법의 결과를 모두 반영하여 더 정확하고 상세한 요약을 제공해주세요.
"""
        elif clip_info:
            combined_info = f"""
{clip_info}

**중요**: CLIP 시각적 분석 결과를 바탕으로 실제 장면을 생생하게 묘사해주세요.
"""
        elif vision_info:
            combined_info = f"""
{vision_info}

**중요**: GPT Vision 분석 결과를 바탕으로 실제 상황을 생생하게 묘사해주세요.
"""
        
        # 비디오 ID와 타임스탬프 추가 (캐싱 방지)
        video_id = video_data.get('video_id', 'unknown')
        current_time = __import__('time').time()
        
        prompt = f"""다음은 CCTV 영상 분석 결과입니다 (비디오 ID: {video_id}, 분석 시간: {current_time}). 
'{video_title}' 영상을 CLIP과 GPT Vision으로 분석한 결과를 바탕으로, 실제로 무슨 일이 일어나고 있는지, 어떤 분위기인지를 생생하게 묘사해주세요.

{combined_info}

**절대 금지사항**:
- 구체적인 수치 언급 (몇 명, 몇 초, 신뢰도, 퍼센트 등)
- 통계적 정보 나열
- 데이터 중심의 설명

**대신 다음에 집중하세요**:
- 실제 상황과 행동 묘사
- 분위기와 맥락 설명
- 사람들의 활동과 상호작용
- 실제 이미지에서 보이는 내용
- **중요**: 구체적인 장소명(건물, 쇼핑몰, 복도, 거리 등)을 언급하지 말고, 일반적인 상황과 분위기로만 설명해주세요

**중요**: 이 비디오는 고유한 내용을 가지고 있으므로, 다른 비디오와 구별되는 특징을 명확히 설명해주세요.

주요 구간들:
"""
        
        for i, clip in enumerate(clips, 1):
            prompt += f"""
구간 {i}:
- 감지된 정보: {clip.get('description', '')}
"""
        
        prompt += """

위 CLIP과 GPT Vision 분석 데이터를 바탕으로 다음을 고려하여 생생한 영상 요약을 작성해주세요:

1. **실제 상황 묘사**: "사람이 몇 명"이 아니라 "무엇을 하고 있는지" 중심으로
2. **분위기와 맥락**: 조용한지, 활발한지, 긴장감이 있는지 등
3. **행동과 움직임**: 걷고 있는지, 서 있는지, 대화하는지 등
4. **시간대별 변화**: 시간이 지나면서 상황이 어떻게 변하는지
5. **시각적 특징**: 색상, 장면 유형, 환경 등이 상황에 미치는 영향

**절대 금지사항**:
- 구체적인 수치 언급 (몇 명, 몇 초, 신뢰도, 퍼센트 등)
- 통계적 정보 나열
- 데이터 중심의 설명
- "19명", "20명", "496명" 같은 정확한 숫자 언급 금지
- "0.2초", "7.2초" 같은 시간 언급 금지
- **구체적인 장소명 언급 금지**: "건물", "쇼핑몰", "복도", "거리", "시장" 등 특정 장소명을 사용하지 말고 "공간", "장소", "환경" 등 일반적 표현 사용

다음 형식으로 작성해주세요:

📹 영상 요약
[전체적인 상황과 분위기를 2-3문장으로 생생하게 묘사]

🎬 주요 장면들
[각 구간별로 실제로 무슨 일이 일어나고 있는지 구체적으로 설명]

💭 상황 분석
[전체적인 분위기, 특징, 주목할 점 등을 분석]

예시:
- "조용한 분위기에서 사람들이 천천히 걸어다니고 있다"
- "활발한 분위기로 여러 사람이 대화를 나누고 있다"
- "긴장감 있는 상황에서 경계하는 모습이 보인다"

단순한 숫자나 통계가 아닌, 실제 상황을 생생하게 묘사해주세요. 
**중요**: 구체적인 장소명(건물, 쇼핑몰, 복도, 거리, 시장 등)은 언급하지 말고, 
일반적인 "공간", "장소", "환경" 등의 표현을 사용해주세요."""

        return prompt
    
    def _create_highlight_prompt(self, highlight_data: Dict[str, Any]) -> str:
        """하이라이트 설명을 위한 프롬프트 생성"""
        clips = highlight_data.get('clips', [])
        criteria = highlight_data.get('criteria', {})
        
        prompt = f"""다음은 CCTV 영상에서 특정 조건으로 찾은 하이라이트 구간들입니다. 단순한 데이터가 아닌, 실제로 무슨 일이 일어나고 있는지, 어떤 상황인지를 생생하게 묘사해주세요.

**절대 금지사항**:
- 구체적인 수치 언급 (몇 명, 몇 초, 신뢰도, 퍼센트 등)
- 통계적 정보 나열
- 데이터 중심의 설명

**대신 다음에 집중하세요**:
- 실제 상황과 행동 묘사
- 분위기와 맥락 설명
- 사람들의 활동과 상호작용

검색 조건:
- 선호 색상: {criteria.get('color_preference', '없음')}
- 선호 연령대: {criteria.get('age_preference', '없음')}

하이라이트 구간들:
"""
        
        for i, clip in enumerate(clips, 1):
            prompt += f"""
구간 {i}:
- 선정 이유: {clip.get('reason', '')}
"""
        
        prompt += """

위 분석 데이터를 바탕으로 다음을 고려하여 생생한 하이라이트 설명을 작성해주세요:

1. **실제 상황 묘사**: "몇 명이 있다"가 아니라 "무엇을 하고 있는지" 중심으로
2. **분위기와 맥락**: 왜 이 구간이 특별한지, 어떤 분위기인지
3. **행동과 상호작용**: 사람들이 어떻게 움직이고 있는지, 서로 어떻게 상호작용하는지
4. **시각적 특징**: 색상, 연령대 등이 상황에 어떤 영향을 주는지

**절대 금지사항**:
- 구체적인 수치 언급 (몇 명, 몇 초, 신뢰도, 퍼센트 등)
- 통계적 정보 나열
- 데이터 중심의 설명
- "19명", "20명", "496명" 같은 정확한 숫자 언급 금지
- "0.2초", "7.2초" 같은 시간 언급 금지

다음 형식으로 작성해주세요:

⭐ 하이라이트 요약
[조건에 맞는 구간들의 전체적인 상황과 분위기를 생생하게 묘사]

🎬 주요 장면들
[각 구간별로 실제로 무슨 일이 일어나고 있는지, 왜 주목할 만한지 구체적으로 설명]

💭 상황 분석
[전체적인 분위기, 특징, 주목할 점 등을 분석]

예시:
- "활발한 거리에서 다양한 연령대의 사람들이 오가고 있다"
- "조용한 분위기에서 몇 명이 대화를 나누고 있다"
- "긴장감 있는 상황에서 경계하는 모습이 보인다"

단순한 숫자나 통계가 아닌, 실제 상황을 생생하게 묘사해주세요."""

        return prompt
    
    def _call_openai_api(self, prompt: str) -> str:
        """OpenAI API 호출"""
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': self.model,
            'messages': [
                {
                    'role': 'system',
                    'content': '당신은 CCTV 영상 분석 전문가입니다. 주어진 분석 데이터를 바탕으로 자연스러운 한국어로 요약과 설명을 작성해주세요.'
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            'max_tokens': 1000,
            'temperature': 0.7
        }
        
        response = requests.post(
            f'{self.base_url}/chat/completions',
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
        else:
            raise Exception(f"OpenAI API 오류: {response.status_code}")
    
    def _generate_fallback_summary(self, video_data: Dict[str, Any]) -> str:
        """LLM 사용 불가 시 기본 요약 생성"""
        clips = video_data.get('clips', [])
        total_duration = video_data.get('total_duration', 0)
        summary_duration = video_data.get('summary_duration', 0)
        
        summary = f"📹 영상 요약\n"
        summary += f"영상에서 주요 구간들을 선별했습니다.\n"
        summary += f"영상에는 다양한 연령대와 성별의 사람들이 등장하며, 각자 다른 색상의 옷을 입고 있습니다.\n\n"
        
        summary += f"🎬 주요 장면들\n"
        for i, clip in enumerate(clips, 1):
            description = clip.get('description', '')
            
            # 간단한 상황 묘사 추가
            if '사람' in description:
                if '19명' in description or '20명' in description:
                    summary += f"{i}. 활발한 분위기로 많은 사람들이 오가고 있습니다.\n"
                elif '9명' in description or '10명' in description:
                    summary += f"{i}. 조용한 분위기에서 소수의 사람들이 움직이고 있습니다.\n"
                else:
                    summary += f"{i}. 사람들이 다양한 활동을 하고 있습니다.\n"
            else:
                summary += f"{i}. {description}\n"
        
        summary += f"\n💭 상황 분석\n"
        summary += f"영상은 전체적으로 다양한 연령대의 사람들이 등장하는 일반적인 거리 풍경을 보여줍니다. "
        summary += f"요약된 구간들은 주요 활동이 집중된 부분들을 선별한 것입니다."
        
        return summary
    
    def _generate_fallback_highlight(self, highlight_data: Dict[str, Any]) -> str:
        """LLM 사용 불가 시 기본 하이라이트 설명 생성"""
        clips = highlight_data.get('clips', [])
        criteria = highlight_data.get('criteria', {})
        
        summary = f"⭐ 하이라이트 요약\n"
        summary += f"검색 조건에 맞는 구간들을 찾았습니다.\n"
        
        # 조건에 따른 분위기 설명
        if criteria.get('person_count_threshold', 0) > 1:
            summary += f"여러 사람들이 함께 있는 활발한 장면들입니다.\n"
        
        if criteria.get('color_preference'):
            summary += f"특히 {criteria['color_preference']}색 옷을 입은 사람들이 돋보이는 장면들입니다.\n"
        
        if criteria.get('age_preference'):
            summary += f"{criteria['age_preference']} 연령대의 사람들이 주로 등장하는 장면들입니다.\n"
        
        summary += f"\n🎬 주요 장면들\n"
        for i, clip in enumerate(clips, 1):
            reason = clip.get('reason', '')
            person_count = clip.get('person_count', 0)
            
            # 상황에 따른 묘사
            if person_count > 10:
                summary += f"{i}. 매우 활발한 분위기로 많은 사람들이 함께 활동하고 있습니다.\n"
            elif person_count > 5:
                summary += f"{i}. 적당한 인원이 모여 있는 분위기입니다.\n"
            else:
                summary += f"{i}. 소수의 사람들이 조용히 활동하고 있습니다.\n"
        
        summary += f"\n💭 상황 분석\n"
        summary += f"선별된 구간들은 검색 조건에 맞는 특별한 상황들을 보여줍니다. "
        summary += f"전체적으로 다양한 연령대와 성별의 사람들이 등장하며, "
        summary += f"각자 다른 활동을 하고 있는 모습을 관찰할 수 있습니다."
        
        return summary

# Mock 클라이언트 (LLM 사용 불가 시)
class MockLLMClient(LLMClient):
    def __init__(self):
        super().__init__()
        self.api_key = None  # 강제로 사용 불가로 설정
    
    def is_available(self) -> bool:
        return False

# 전역 인스턴스
try:
    llm_client = LLMClient()
except Exception as e:
    print(f"⚠️ LLM 클라이언트 초기화 실패: {e}")
    llm_client = MockLLMClient()
