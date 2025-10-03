from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import api_view, authentication_classes, permission_classes
from rest_framework.authentication import TokenAuthentication
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.parsers import MultiPartParser, FormParser
from django.http import HttpResponse, Http404
from chat.serializers import UserSerializer, VideoChatSessionSerializer, VideoChatMessageSerializer, VideoAnalysisCacheSerializer
from chat.models import VideoChatSession, VideoChatMessage, VideoAnalysisCache, Video
from .services.video_analysis_service import video_analysis_service
from django.utils import timezone
import threading
import openai
import anthropic
from groq import Groq
import ollama
import anthropic
import os
import sys
import io
import PyPDF2
from PIL import Image
import pytesseract
# import cv2  # NumPy 호환성 문제로 조건부 import
# import numpy as np  # NumPy 호환성 문제로 조건부 import
from pdf2image import convert_from_bytes
import base64
import tempfile
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import requests
import uuid
from django.contrib.auth import get_user_model
from chat.models import User, SocialAccount

# 인코딩 문제 해결을 위한 설정
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 파일 처리 유틸리티 함수들
def extract_text_from_pdf(file_content):
    """PDF에서 텍스트 추출 (직접 추출 + OCR 백업)"""
    try:
        pdf_file = io.BytesIO(file_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        
        # 먼저 직접 텍스트 추출 시도
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            text += page_text + "\n"
        
        # 추출된 텍스트가 충분하지 않으면 OCR 시도
        if len(text.strip()) < 100:  # 텍스트가 너무 적으면 OCR 사용
            print("PDF 직접 추출 텍스트가 부족하여 OCR을 사용합니다.")
            return extract_text_from_pdf_ocr(file_content)
        
        return text.strip()
    except Exception as e:
        print(f"PDF 직접 추출 실패, OCR을 사용합니다: {str(e)}")
        return extract_text_from_pdf_ocr(file_content)

def extract_text_from_pdf_ocr(file_content):
    """PDF를 이미지로 변환 후 OCR로 텍스트 추출"""
    try:
        # PDF를 이미지로 변환
        images = convert_from_bytes(file_content, dpi=300)
        all_text = ""
        
        for i, image in enumerate(images):
            # 간단한 이미지 전처리 (NumPy 없이)
            # 이미지를 그레이스케일로 변환
            if image.mode != 'L':
                image = image.convert('L')
            
            # OCR 수행 (전처리 없이)
            page_text = pytesseract.image_to_string(image, lang='kor+eng')
            all_text += f"\n--- 페이지 {i+1} ---\n{page_text}\n"
        
        return all_text.strip()
    except Exception as e:
        return f"PDF OCR 처리 중 오류 발생: {str(e)}"

def extract_text_from_image(file_content):
    """이미지에서 OCR을 사용하여 텍스트 추출"""
    try:
        # 이미지 열기
        image = Image.open(io.BytesIO(file_content))
        
        # 이미지 전처리 (간단한 방식)
        if image.mode != 'L':
            image = image.convert('L')  # 그레이스케일로 변환
        
        # OCR 수행 (한국어 + 영어)
        text = pytesseract.image_to_string(image, lang='kor+eng')
        
        return text.strip()
    except Exception as e:
        return f"이미지 텍스트 추출 중 오류 발생: {str(e)}"

def process_uploaded_file(file):
    """업로드된 파일 처리"""
    file_content = file.read()
    file_extension = file.name.split('.')[-1].lower()
    
    if file_extension == 'pdf':
        return extract_text_from_pdf(file_content)
    elif file_extension in ['jpg', 'jpeg', 'png', 'bmp', 'tiff']:
        # 이미지 파일의 경우 파일 경로를 반환 (Ollama가 직접 읽도록)
        return f"IMAGE_FILE:{file.name}"
    else:
        return "지원하지 않는 파일 형식입니다. PDF 또는 이미지 파일을 업로드해주세요."

def summarize_content(content, api_key=None, file_path=None):
    """내용을 요약하는 함수 (Ollama 사용)"""
    try:
        # 이미지 파일인지 확인
        if content.startswith("IMAGE_FILE:"):
            if file_path and os.path.exists(file_path):
                return analyze_image_with_ollama(file_path)
            else:
                return "이미지 파일을 찾을 수 없습니다."
        
        # 텍스트 내용인 경우
        # 내용이 너무 길면 자르기 (토큰 제한 고려)
        if len(content) > 12000:
            content = content[:12000] + "..."
        
        # 요약 프롬프트
        prompt = f"""당신은 문서 내용을 요약하는 전문가입니다. 

주어진 내용이 PDF에서 추출된 텍스트인 경우:
- OCR 오류나 불완전한 텍스트가 있을 수 있음을 고려
- 가능한 한 원문의 의도를 파악하여 요약
- 중요한 정보는 보존하되 간결하게 정리

요약 시 다음을 포함해주세요:
1. 문서의 주요 주제/목적
2. 핵심 내용과 중요한 포인트
3. 결론이나 요약 (있는 경우)

원문의 주요 내용을 보존하면서도 간결하게 작성해주세요.

다음 내용을 요약해주세요:

{content}"""
        
        # Ollama 클라이언트로 요약 수행
        response = ollama.chat(
                   model='llama3.2:latest',  # 또는 사용 가능한 다른 모델
            messages=[
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            options={
                'temperature': 0.3,
                'num_predict': 1500
            }
        )
        
        return response['message']['content']
    except Exception as e:
        print(f"Ollama 요약 오류: {str(e)}")
        # Ollama 실패 시 기본 요약
        if len(content) > 1000:
            return f"문서 요약 (Ollama 오류로 간단 요약): {content[:500]}..."
        return content

def analyze_image_with_ollama(image_path):
    """하이브리드 이미지 분석 (OCR + Ollama)"""
    try:
        # 1단계: OCR로 텍스트 추출 시도
        try:
            from PIL import Image
            import pytesseract
            
            image = Image.open(image_path)
            ocr_text = pytesseract.image_to_string(image, lang='kor+eng')
            
            if len(ocr_text.strip()) > 10:  # 텍스트가 충분히 있으면 OCR 결과 사용
                return f"이미지에서 텍스트를 추출했습니다: {ocr_text.strip()[:200]}"
        except:
            pass
        
        # 2단계: OCR 실패 시 Ollama로 간단한 분석 (영어로 답변)
        prompt = """IMPORTANT: Count objects very carefully. Look at the image multiple times.

Count the exact number of objects in this image. Be very precise about the count.
Then describe each object's type and main colors.

Examples:
- "1 gray and white cat, blue background"
- "2 dogs, white background" 
- "3 cars, street scene"

Answer in English very concisely. Double-check your count."""
        
        # 성능 최적화: 더 가벼운 모델 사용
        try:
            # llava:7b 사용 (가장 가벼운 비전 모델)
            response = ollama.chat(
                model='llava:7b',
                messages=[
                    {
                        'role': 'user',
                        'content': prompt,
                        'images': [image_path]
                    }
                ],
                options={
                    'temperature': 0.1,  # 더 낮은 temperature로 일관성 향상
                    'num_predict': 300,  # 토큰 수 더 줄임
                    'num_ctx': 1024  # 컨텍스트 크기 더 제한
                }
            )
            
            # Ollama 응답 로깅
            ollama_response = response['message']['content']
            print(f"Ollama 분석 결과: {ollama_response}")
            return ollama_response
            
        except Exception as e:
            print(f"Ollama 이미지 분석 실패, GPT API로 fallback: {str(e)}")
            # GPT API로 fallback (비용이 들지만 정확도 높음)
            try:
                import openai
                import base64
                
                with open(image_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                
                response = openai.chat.completions.create(
                    model="gpt-4o-mini",  # 가장 저렴한 GPT-4 비전 모델
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Count the exact number of objects in this image. Then describe each object's type and main colors only. Answer in English."},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                                }
                            ]
                        }
                    ],
                    max_tokens=100  # 토큰 수 제한으로 비용 절약
                )
                
                gpt_response = response.choices[0].message.content
                print(f"GPT 분석 결과: {gpt_response}")
                return gpt_response
            except Exception as gpt_error:
                print(f"GPT API fallback도 실패: {str(gpt_error)}")
                return "이미지 분석 중 오류가 발생했습니다. 이미지를 다시 업로드해주세요."
    except Exception as e:
        print(f"Ollama 이미지 분석 오류: {str(e)}")
        return f"이미지 분석 중 오류가 발생했습니다: {str(e)}"

def generate_optimal_response_with_ollama(ai_responses, user_question):
    """Ollama를 사용하여 최적의 답변 생성 (비용 절약 + 품질 향상)"""
    try:
        # AI 응답들을 정리
        responses_text = ""
        model_names = []
        for model_name, response in ai_responses.items():
            responses_text += f"### {model_name.upper()}:\n{response}\n\n"
            model_names.append(model_name.upper())
        
        # AI 분석 섹션 생성
        analysis_sections = ""
        for name in model_names:
            analysis_sections += f"### {name}\n- 장점: [주요 장점]\n- 단점: [주요 단점]\n- 특징: [특별한 특징]\n"
        
        # 비용 절약을 위한 간소화된 프롬프트
        prompt = f"""AI 응답을 분석하여 최적의 통합 답변을 제공해주세요.

형식:
## 통합 답변
[모든 AI의 장점을 결합한 최적 답변]

## 각 AI 분석
{analysis_sections}
## 분석 근거
[통합 답변을 만든 구체적 이유]

## 최종 추천
[상황별 AI 선택 가이드]

질문: {user_question}

AI 답변들:
{responses_text}

위 답변들을 분석하여 최적의 통합 답변을 제공해주세요."""
        
        response = ollama.chat(
                   model='llama3.2:latest',
            messages=[
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            options={
                'temperature': 0.7,
                'num_predict': 2500
            }
        )
        
        return response['message']['content']
    except Exception as e:
        return f"Ollama 최적 답변 생성 중 오류가 발생했습니다: {str(e)}"

def generate_optimal_response(ai_responses, user_question, api_key=None):
    """AI들의 응답을 통합하여 최적의 답변 생성 (Ollama 사용)"""
    try:
        # Ollama로 최적 답변 생성 (비용 절약)
        if not api_key:
            return generate_optimal_response_with_ollama(ai_responses, user_question)
        
        import openai
        client = openai.OpenAI(api_key=api_key)
        
        # AI 응답들을 정리
        responses_text = ""
        model_names = []
        for model_name, response in ai_responses.items():
            responses_text += f"### {model_name.upper()}:\n{response}\n\n"
            model_names.append(model_name.upper())
        
        # 모델별 분석 섹션 동적 생성
        analysis_sections = ""
        for model_name in model_names:
            analysis_sections += f"""
### {model_name}
- 장점: [주요 장점]
- 단점: [주요 단점]
- 특징: [특별한 특징]
"""
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"""당신은 AI 응답 분석 및 최적화 전문가입니다. 여러 AI의 답변을 분석하여 가장 완전하고 정확한 통합 답변을 제공해야 합니다.

다음 형식으로 응답해주세요:

## 🎯 통합 답변
[가장 완전하고 정확한 통합 답변 - 모든 AI의 장점을 결합한 최적의 답변]

## 📊 각 AI 분석
{analysis_sections}

## 🔍 분석 근거
[각 AI의 정보를 어떻게 조합하여 통합 답변을 만들었는지 구체적으로 설명]

## 🏆 최종 추천
[가장 추천하는 답변과 그 이유 - 어떤 상황에서 어떤 AI를 선택해야 하는지 포함]

## 💡 추가 인사이트
[질문에 대한 더 깊은 이해나 추가 고려사항]"""},
                {"role": "user", "content": f"질문: {user_question}\n\n다음은 여러 AI의 답변입니다:\n\n{responses_text}\n위 답변들을 분석하여 최적의 통합 답변을 제공해주세요."}
            ],
            temperature=0.7,
            max_tokens=2500
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"최적화된 답변 생성 중 오류가 발생했습니다: {str(e)}"

class ChatBot:
    def __init__(self, api_key, model, api_type):
        self.conversation_history = []
        self.model = model
        self.api_type = api_type
        self.api_key = api_key  # api_key 속성 추가
        
        # API 키가 비어있는지 확인
        if not api_key:
            raise ValueError(f"{api_type.upper()} API 키가 설정되지 않았습니다.")
        
        if api_type == 'openai':
            self.client = openai.OpenAI(api_key=api_key)
        elif api_type == 'anthropic':
            self.client = anthropic.Client(api_key=api_key)
        elif api_type == 'groq':
            self.client = Groq(api_key=api_key)
    
    def chat(self, user_input):
        try:
            # 대화 시작 시 시스템 메시지 추가 (특수 문자 제거)
            if not self.conversation_history:
                if self.api_type == 'anthropic':
                    system_content = "You are Claude, an AI assistant that can analyze images and respond in Korean. When you receive image analysis results from other AI systems (like Ollama), you should treat them as if you analyzed the image yourself and provide detailed, natural descriptions in Korean. Make the descriptions rich, engaging, and easy to understand while maintaining the accuracy of the original analysis."
                elif self.api_type == 'openai':
                    system_content = "You are GPT, an AI assistant that can analyze images and respond in Korean. When you receive image analysis results from other AI systems (like Ollama), you should treat them as if you analyzed the image yourself and provide detailed, natural descriptions in Korean. Make the descriptions rich, engaging, and easy to understand while maintaining the accuracy of the original analysis."
                elif self.api_type == 'groq':
                    system_content = "You are Mixtral, an AI assistant that can analyze images and respond in Korean. When you receive image analysis results from other AI systems (like Ollama), you should treat them as if you analyzed the image yourself and provide detailed, natural descriptions in Korean. Make the descriptions rich, engaging, and easy to understand while maintaining the accuracy of the original analysis."
                else:
                    system_content = "You are an AI assistant that can analyze images and respond in Korean. When you receive image analysis results from other AI systems (like Ollama), you should treat them as if you analyzed the image yourself and provide detailed, natural descriptions in Korean."
                
                self.conversation_history.append({
                    "role": "system",
                    "content": system_content
                })

                # 사용자 입력 출력 (인코딩 안전하게 처리)
                try:
                    safe_input = user_input.encode('ascii', 'ignore').decode('ascii')
                    print(f"User input: {safe_input}")
                except:
                    print("User input received")
            
            self.conversation_history.append({"role": "user", "content": user_input})
            
            # 인코딩 안전한 응답 변수 초기화
            assistant_response = ""
            
            if self.api_type == 'openai':
                # OpenAI 방식 처리
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.conversation_history,
                    temperature=0.7,
                    max_tokens=1024
                )
                assistant_response = response.choices[0].message.content
            
            elif self.api_type == 'anthropic':
                # Anthropic Messages API 방식 처리
                try:
                    client = anthropic.Client(api_key=self.api_key)
                    
                    # 대화 히스토리를 포함한 메시지 생성
                    messages = []
                    for msg in self.conversation_history:
                        if msg['role'] == 'system':
                            continue  # Claude는 system 메시지를 지원하지 않음
                        messages.append({
                            "role": msg['role'],
                            "content": msg['content']
                        })
                    
                    message = client.messages.create(
                        model="claude-3-5-haiku-20241022",
                        max_tokens=4096,
                        temperature=0.7,
                        messages=messages
                    )
                    
                    # 응답 추출
                    raw_response = message.content[0].text
                    assistant_response = raw_response
                    
                    print("Claude response processed successfully")
                    
                except Exception as claude_error:
                    print(f"Claude API error: {str(claude_error)}")
                    print(f"API Key: {self.api_key[:20] if self.api_key else 'None'}...")
                    # API 키가 없거나 오류가 발생한 경우 기본 응답
                    assistant_response = f"안녕하세요! '{user_input}'에 대해 도움을 드릴 수 있습니다. 구체적으로 어떤 정보가 필요하신가요?"


            
            elif self.api_type == 'groq':
                # Groq 방식 처리
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.conversation_history,
                    temperature=0.7,
                    max_tokens=1024
                )
                assistant_response = response.choices[0].message.content
            
            # 대화 이력에 추가
            self.conversation_history.append({"role": "assistant", "content": assistant_response})
            return assistant_response
        except Exception as e:
            # 인코딩 안전한 오류 처리
            try:
                error_msg = str(e)
                # 특수 문자 제거
                import re
                safe_error_msg = re.sub(r'[^\x00-\x7F]+', '', error_msg)
                print(f"Error: {safe_error_msg}")
                return f"오류가 발생했습니다: {safe_error_msg}"
            except:
                print("Error occurred (encoding issue)")
                return "오류가 발생했습니다: 인코딩 문제"

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', '')
GROQ_API_KEY = os.getenv('GROQ_API_KEY', '')



# API 키가 있는 경우에만 ChatBot 인스턴스 생성
chatbots = {}
try:
    if OPENAI_API_KEY:
        chatbots['gpt'] = ChatBot(OPENAI_API_KEY, 'gpt-3.5-turbo', 'openai')
except ValueError as e:
    print(f"GPT 모델 초기화 실패: {e}")

try:
    if ANTHROPIC_API_KEY:
        chatbots['claude'] = ChatBot(ANTHROPIC_API_KEY, 'claude-3-5-haiku-20241022', 'anthropic')
except ValueError as e:
    print(f"Claude 모델 초기화 실패: {e}")

try:
    if GROQ_API_KEY:
        chatbots['mixtral'] = ChatBot(GROQ_API_KEY, 'llama-3.1-8b-instant', 'groq')
        chatbots['optimal'] = ChatBot(GROQ_API_KEY, 'llama-3.1-8b-instant', 'groq')
        # 현재 사용 가능한 Groq 모델들:
        # - llama-3.1-8b-instant (현재 사용 가능)
        # - llama-3.1-70b-versatile (deprecated)
        # - mixtral-8x7b-32768 (deprecated)
        # - mixtral-8x7b-instruct (not found)
except ValueError as e:
    print(f"Groq 모델 초기화 실패: {e}")

class ChatView(APIView):
    def post(self, request, bot_name):
        try:
            data = request.data
            user_message = data.get('message')
            uploaded_file = request.FILES.get('file')
            
            if not user_message and not uploaded_file:
                return Response({'error': 'No message or file provided'}, status=status.HTTP_400_BAD_REQUEST)
            
            chatbot = chatbots.get(bot_name)
            if not chatbot:
                return Response({'error': 'Invalid bot name'}, status=status.HTTP_400_BAD_REQUEST)

            # 파일이 업로드된 경우 처리
            if uploaded_file:
                try:
                    print(f"파일 업로드 감지: {uploaded_file.name}")
                    
                    # 파일에서 텍스트 추출 또는 이미지 파일 식별
                    extracted_content = process_uploaded_file(uploaded_file)
                    print(f"처리된 내용: {extracted_content[:100]}...")
                    
                    # Ollama로 분석 (이미지는 직접, 텍스트는 요약)
                    print("Ollama를 사용하여 파일 분석 중...")
                    
                    # 임시 파일 저장
                    temp_file_path = None
                    if extracted_content.startswith("IMAGE_FILE:"):
                        # 이미지 파일을 임시로 저장
                        import tempfile
                        import shutil
                        temp_dir = tempfile.mkdtemp()
                        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(temp_file_path, 'wb') as temp_file:
                            for chunk in uploaded_file.chunks():
                                temp_file.write(chunk)
                        print(f"이미지 파일 임시 저장: {temp_file_path}")
                    
                    analyzed_content = summarize_content(extracted_content, file_path=temp_file_path)
                    
                    # 파일 타입에 따라 다른 메시지 생성
                    if uploaded_file.name.lower().endswith('.pdf'):
                        final_message = f"다음 문서 내용을 한국어로 요약해주세요:\n\n{analyzed_content}"
                    else:
                        # 모든 AI가 이미지 분석 결과를 받아서 재구성하여 답변하도록 수정
                        if bot_name in ['claude', 'gpt', 'mixtral']:
                            final_message = f"""이미지 분석 결과를 받았습니다. 다음은 Ollama가 분석한 내용입니다:

{analyzed_content}

위 분석 결과를 바탕으로 이 이미지에 대해 한국어로 자세하고 자연스럽게 설명해주세요. 분석 결과의 내용을 그대로 전달하되, 더 풍부하고 이해하기 쉬운 표현으로 재구성해주세요."""
                        else:
                            final_message = f"이미지를 분석해보니 {analyzed_content}입니다. 이 이미지에 대해 한국어로 자세히 설명해주세요."
                    print("분석 완료")
                except Exception as e:
                    print(f"파일 처리 오류: {str(e)}")
                    final_message = f"파일 처리 중 오류가 발생했습니다: {str(e)}"
            else:
                final_message = user_message

            # optimal 모델인 경우 특별 처리
            if bot_name == 'optimal':
                # 다른 AI들의 응답을 수집
                other_responses_str = data.get('other_responses', '{}')
                try:
                    import json
                    other_responses = json.loads(other_responses_str)
                except:
                    other_responses = {}
                
                if other_responses and len(other_responses) > 0:
                    # 비용 절약: Ollama 사용으로 최적화된 통합 답변 생성
                    response = generate_optimal_response(other_responses, final_message, OPENAI_API_KEY)
                else:
                    # 다른 모델들의 응답이 없으면 일반적인 응답
                    response = chatbot.chat(final_message)
            else:
                # 비용 절약: 파일 분석 시 간소화된 프롬프트 사용
                if uploaded_file and '파일 내용을 분석해' in final_message:
                    # 이미 Ollama로 분석된 내용이므로 간단한 응답 요청
                    simplified_message = f"다음 분석 내용에 대해 간단한 의견을 제시해주세요:\n\n{final_message.split('다음 파일 내용을 분석해주세요:')[1] if '다음 파일 내용을 분석해주세요:' in final_message else final_message}"
                    response = chatbot.chat(simplified_message)
                else:
                    response = chatbot.chat(final_message)
                
            return Response({'response': response})
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

def generate_unique_username(email, name=None):
    """이메일 기반으로 고유한 사용자명 생성"""
    base_username = email.split('@')[0]
    username = base_username
    counter = 1
    
    while User.objects.filter(username=username).exists():
        username = f"{base_username}_{counter}"
        counter += 1
    
    return username

@api_view(['GET'])
@authentication_classes([TokenAuthentication])
@permission_classes([AllowAny])
def google_callback(request):
    try:
        # 액세스 토큰 추출
        auth_header = request.headers.get('Authorization', '')
        if not auth_header.startswith('Bearer '):
            return Response(
                {'error': '잘못된 인증 헤더'}, 
                status=status.HTTP_401_UNAUTHORIZED
            )
        
        access_token = auth_header.split(' ')[1]

        # Google API로 사용자 정보 요청
        user_info_response = requests.get(
            'https://www.googleapis.com/oauth2/v3/userinfo',
            headers={'Authorization': f'Bearer {access_token}'}
        )

        if user_info_response.status_code != 200:
            return Response(
                {'error': 'Google에서 사용자 정보를 가져오는데 실패했습니다'}, 
                status=status.HTTP_400_BAD_REQUEST
            )

        user_info = user_info_response.json()
        email = user_info.get('email')
        name = user_info.get('name')
        
        if not email:
            return Response(
                {'error': '이메일이 제공되지 않았습니다'}, 
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            # 기존 사용자 검색
            user = User.objects.get(email=email)
            # 기존 사용자의 이름이 없으면 업데이트
            if name and (not user.first_name and not user.last_name):
                if ' ' in name:
                    first_name, last_name = name.split(' ', 1)
                    user.first_name = first_name
                    user.last_name = last_name
                else:
                    user.first_name = name
                user.save()
        except User.DoesNotExist:
            # 새로운 사용자 생성
            username = generate_unique_username(email, name)
            user = User.objects.create(
                username=username,
                email=email,
                is_active=True
            )
            
            # 이름 설정
            if name:
                if ' ' in name:
                    first_name, last_name = name.split(' ', 1)
                    user.first_name = first_name
                    user.last_name = last_name
                else:
                    user.first_name = name
            
            # 기본 비밀번호 설정 (선택적)
            random_password = uuid.uuid4().hex
            user.set_password(random_password)
            user.save()

        # 소셜 계정 정보 생성 또는 업데이트
        social_account, created = SocialAccount.objects.get_or_create(
            email=email,
            provider='google',
            defaults={'user': user}
        )

        if not created and social_account.user != user:
            social_account.user = user
            social_account.save()

        # 사용자 정보 직렬화
        serializer = UserSerializer(user)
        
        return Response({
            'message': '구글 로그인 성공',
            'user': serializer.data
        })
        
    except Exception as e:
        return Response(
            {'error': str(e)}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['POST'])
@authentication_classes([TokenAuthentication])
@permission_classes([AllowAny])
def kakao_callback(request):
    """카카오 로그인 콜백"""
    try:
        data = request.data
        access_token = data.get('access_token')
        
        if not access_token:
            return Response(
                {'error': '액세스 토큰이 제공되지 않았습니다'}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # 카카오 API로 사용자 정보 가져오기
        user_info_response = requests.get(
            'https://kapi.kakao.com/v2/user/me',
            headers={'Authorization': f'Bearer {access_token}'}
        )
        
        if user_info_response.status_code != 200:
            return Response(
                {'error': '카카오에서 사용자 정보를 가져오는데 실패했습니다'}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        user_info = user_info_response.json()
        kakao_account = user_info.get('kakao_account', {})
        profile = kakao_account.get('profile', {})
        
        email = kakao_account.get('email')
        name = profile.get('nickname')
        
        if not email:
            return Response(
                {'error': '이메일이 제공되지 않았습니다'}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            # 기존 사용자 검색
            user = User.objects.get(email=email)
            # 기존 사용자의 이름이 없으면 업데이트
            if name and (not user.first_name and not user.last_name):
                user.first_name = name
                user.save()
        except User.DoesNotExist:
            # 새로운 사용자 생성
            username = generate_unique_username(email, name)
            user = User.objects.create(
                username=username,
                email=email,
                is_active=True
            )
            
            # 이름 설정
            if name:
                user.first_name = name
            
            # 기본 비밀번호 설정 (선택적)
            random_password = uuid.uuid4().hex
            user.set_password(random_password)
            user.save()
        
        # 소셜 계정 정보 생성 또는 업데이트
        social_account, created = SocialAccount.objects.get_or_create(
            email=email,
            provider='kakao',
            defaults={'user': user}
        )
        
        if not created and social_account.user != user:
            social_account.user = user
            social_account.save()
        
        serializer = UserSerializer(user)
        return Response({
            'message': '카카오 로그인 성공',
            'user': serializer.data
        })
        
    except Exception as e:
        return Response(
            {'error': str(e)}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['POST'])
@authentication_classes([TokenAuthentication])
@permission_classes([AllowAny])
def naver_callback(request):
    """네이버 로그인 콜백"""
    try:
        data = request.data
        access_token = data.get('access_token')
        
        if not access_token:
            return Response(
                {'error': '액세스 토큰이 제공되지 않았습니다'}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # 네이버 API로 사용자 정보 가져오기
        user_info_response = requests.get(
            'https://openapi.naver.com/v1/nid/me',
            headers={'Authorization': f'Bearer {access_token}'}
        )
        
        if user_info_response.status_code != 200:
            return Response(
                {'error': '네이버에서 사용자 정보를 가져오는데 실패했습니다'}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        user_info = user_info_response.json()
        response_data = user_info.get('response', {})
        
        email = response_data.get('email')
        name = response_data.get('name')
        nickname = response_data.get('nickname')
        
        if not email:
            return Response(
                {'error': '이메일이 제공되지 않았습니다'}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # 이름이 없으면 닉네임 사용
        display_name = name or nickname
        
        try:
            # 기존 사용자 검색
            user = User.objects.get(email=email)
            # 기존 사용자의 이름이 없으면 업데이트
            if display_name and (not user.first_name and not user.last_name):
                user.first_name = display_name
                user.save()
        except User.DoesNotExist:
            # 새로운 사용자 생성
            username = generate_unique_username(email, display_name)
            user = User.objects.create(
                username=username,
                email=email,
                is_active=True
            )
            
            # 이름 설정
            if display_name:
                user.first_name = display_name
            
            # 기본 비밀번호 설정 (선택적)
            random_password = uuid.uuid4().hex
            user.set_password(random_password)
            user.save()
        
        # 소셜 계정 정보 생성 또는 업데이트
        social_account, created = SocialAccount.objects.get_or_create(
            email=email,
            provider='naver',
            defaults={'user': user}
        )
        
        if not created and social_account.user != user:
            social_account.user = user
            social_account.save()
        
        serializer = UserSerializer(user)
        return Response({
            'message': '네이버 로그인 성공',
            'user': serializer.data
        })
        
    except Exception as e:
        return Response(
            {'error': str(e)}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

class VideoUploadView(APIView):
    """영상 업로드 뷰 - 독립적인 영상 처리"""
    permission_classes = [AllowAny]  # 임시로 AllowAny로 변경
    parser_classes = (MultiPartParser, FormParser)
    
    def post(self, request):
        try:
            import os
            import uuid
            import time
            from django.core.files.storage import default_storage
            from django.conf import settings
            
            # 업로드된 파일 확인 (backend_videochat 방식)
            if 'video' not in request.FILES:
                return Response({
                    'error': '비디오 파일이 없습니다'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            video_file = request.FILES['video']
            
            # 파일 확장자 검증 (backend_videochat 방식)
            if not video_file.name.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                return Response({
                    'error': '지원하지 않는 파일 형식입니다. MP4, AVI, MOV, MKV, WEBM 형식만 지원됩니다.'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # 파일 크기 검증 (50MB 제한)
            max_size = 50 * 1024 * 1024  # 50MB
            if video_file.size > max_size:
                return Response({
                    'error': f'파일 크기가 너무 큽니다. 최대 50MB까지 업로드 가능합니다. (현재: {video_file.size / (1024*1024):.1f}MB)'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # 파일명 길이 검증
            if len(video_file.name) > 200:
                return Response({
                    'error': '파일명이 너무 깁니다. 200자 이하로 제한됩니다.'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # 고유한 파일명 생성 (backend_videochat 방식)
            timestamp = int(time.time())
            filename = f"upload_{timestamp}_{video_file.name}"
            
            # 파일 저장 (backend_videochat 방식)
            from django.core.files.base import ContentFile
            file_path = default_storage.save(
                f'uploads/{filename}',
                ContentFile(video_file.read())
            )
            full_path = os.path.join(settings.MEDIA_ROOT, file_path)
            
            # 파일 저장 검증
            if not os.path.exists(full_path):
                return Response({
                    'error': '파일 저장에 실패했습니다. 다시 시도해주세요.'
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
            # 파일 크기 재검증 (실제 저장된 파일)
            actual_size = os.path.getsize(full_path)
            if actual_size == 0:
                return Response({
                    'error': '빈 파일이 업로드되었습니다. 유효한 영상 파일을 선택해주세요.'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Create Video model instance (backend_videochat 방식)
            video = Video.objects.create(
                filename=filename,
                original_name=video_file.name,
                file_path=file_path,
                file_size=video_file.size,
                file=file_path,  # file 필드도 저장
                analysis_status='pending'
            )
            
            # 백그라운드에서 영상 분석 시작
            def analyze_video_background():
                try:
                    print(f"🎬 백그라운드 영상 분석 시작: {video.id}")
                    
                    # 파일 존재 여부 재확인
                    if not os.path.exists(full_path):
                        print(f"❌ 영상 파일이 존재하지 않음: {full_path}")
                        video.analysis_status = 'failed'
                        video.analysis_message = '영상 파일을 찾을 수 없습니다.'
                        video.save()
                        return
                    
                    analysis_result = video_analysis_service.analyze_video(file_path, video.id)
                    if analysis_result and analysis_result is not True:
                        # 분석 결과가 딕셔너리인 경우 (오류 정보 포함)
                        if isinstance(analysis_result, dict) and not analysis_result.get('success', True):
                            print(f"❌ 영상 분석 실패: {video.id} - {analysis_result.get('error_message', 'Unknown error')}")
                            video.analysis_status = 'failed'
                            video.analysis_message = analysis_result.get('error_message', '분석 중 오류가 발생했습니다.')
                        else:
                            print(f"✅ 영상 분석 완료: {video.id}")
                            video.analysis_status = 'completed'
                            video.is_analyzed = True
                    else:
                        print(f"❌ 영상 분석 실패: {video.id}")
                        video.analysis_status = 'failed'
                        video.analysis_message = '분석 중 오류가 발생했습니다.'
                    
                    video.save()
                except Exception as e:
                    print(f"❌ 백그라운드 분석 오류: {e}")
                    video.analysis_status = 'failed'
                    video.analysis_message = f'분석 중 오류가 발생했습니다: {str(e)}'
                    video.save()
            
            # 별도 스레드에서 분석 실행
            analysis_thread = threading.Thread(target=analyze_video_background)
            analysis_thread.daemon = True
            analysis_thread.start()
            
            return Response({
                'success': True,
                'video_id': video.id,
                'filename': filename,
                'message': f'비디오 "{video_file.name}"이 성공적으로 업로드되었습니다.'
            })
                
        except Exception as e:
            return Response({
                'error': f'영상 업로드 중 오류 발생: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class VideoListView(APIView):
    """비디오 목록 조회 - backend_videochat 방식"""
    permission_classes = [AllowAny]
    
    def get(self, request):
        try:
            videos = Video.objects.all()
            video_list = []
            
            for video in videos:
                # 분석 상태 결정 (더 정확한 판단)
                actual_analysis_status = video.analysis_status
                if video.analysis_status == 'completed' and not video.analysis_json_path:
                    actual_analysis_status = 'failed'
                    print(f"⚠️ 영상 {video.id}: analysis_status는 completed이지만 analysis_json_path가 없음")
                
                video_data = {
                    'id': video.id,
                    'filename': video.filename,
                    'original_name': video.original_name,
                    'duration': video.duration,
                    'is_analyzed': video.is_analyzed,
                    'analysis_status': actual_analysis_status,  # 실제 상태 사용
                    'uploaded_at': video.uploaded_at,
                    'file_size': video.file_size
                }
                video_list.append(video_data)
            
            return Response({
                'videos': video_list,
                'count': len(video_list)
            })
            
        except Exception as e:
            return Response({
                'error': f'비디오 목록 조회 중 오류 발생: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class VideoAnalysisView(APIView):
    """영상 분석 상태 확인 및 시작 - backend_videochat 방식"""
    permission_classes = [AllowAny]
    
    def get(self, request, video_id):
        try:
            video = Video.objects.get(id=video_id)
            
            # 진행률 정보 추출
            progress_info = {
                'analysis_progress': video.analysis_progress,
                'analysis_message': video.analysis_message or ''
            }
            
            # 분석 상태 결정 (더 정확한 판단)
            actual_analysis_status = video.analysis_status
            if video.analysis_status == 'completed' and not video.analysis_json_path:
                actual_analysis_status = 'failed'
                print(f"⚠️ 영상 {video_id}: analysis_status는 completed이지만 analysis_json_path가 없음")
            
            return Response({
                'video_id': video.id,
                'filename': video.filename,
                'original_name': video.original_name,
                'analysis_status': actual_analysis_status,  # 실제 상태 사용
                'is_analyzed': video.is_analyzed,
                'duration': video.duration,
                'uploaded_at': video.uploaded_at,
                'file_size': video.file_size,
                'analysis_json_path': video.analysis_json_path,
                'frame_images_path': video.frame_images_path,
                'progress': progress_info
            })
        except Video.DoesNotExist:
            return Response({
                'error': '영상을 찾을 수 없습니다'
            }, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({
                'error': f'영상 분석 조회 중 오류 발생: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    def post(self, request, video_id):
        """영상 분석 시작"""
        try:
            video = Video.objects.get(id=video_id)
            
            # 이미 분석 중이거나 완료된 경우
            if video.analysis_status == 'pending':
                return Response({
                    'message': '이미 분석이 진행 중입니다.',
                    'status': 'pending'
                })
            elif video.analysis_status == 'completed':
                return Response({
                    'message': '이미 분석이 완료되었습니다.',
                    'status': 'completed'
                })
            
            # 분석 상태를 pending으로 변경
            video.analysis_status = 'pending'
            video.save()
            
            # 백그라운드에서 영상 분석 시작
            def analyze_video_background():
                try:
                    print(f"🎬 백그라운드 영상 분석 시작: {video.id}")
                    analysis_result = video_analysis_service.analyze_video(video.file_path, video.id)
                    if analysis_result:
                        print(f"✅ 영상 분석 완료: {video.id}")
                        # Video 모델 업데이트
                        video.analysis_status = 'completed'
                        video.is_analyzed = True
                        video.save()
                    else:
                        print(f"❌ 영상 분석 실패: {video.id}")
                        video.analysis_status = 'failed'
                        video.save()
                except Exception as e:
                    print(f"❌ 백그라운드 분석 오류: {e}")
                    video.analysis_status = 'failed'
                    video.save()
            
            # 별도 스레드에서 분석 실행
            analysis_thread = threading.Thread(target=analyze_video_background)
            analysis_thread.daemon = True
            analysis_thread.start()
            
            return Response({
                'message': '영상 분석을 시작했습니다.',
                'status': 'pending'
            })
            
        except Video.DoesNotExist:
            return Response({
                'error': '영상을 찾을 수 없습니다'
            }, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({
                'error': f'영상 분석 시작 중 오류 발생: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class VideoChatView(APIView):
    """영상 채팅 뷰 - 다중 AI 응답 및 통합"""
    permission_classes = [AllowAny]  # 임시로 AllowAny로 변경
    
    def get(self, request, video_id=None):
        """채팅 세션 목록 조회"""
        try:
            print(f"🔍 VideoChatView GET 요청 - video_id: {video_id}")
            
            # 사용자 정보 처리 (인증되지 않은 경우 기본 사용자 사용)
            user = None
            if hasattr(request, 'user') and request.user.is_authenticated:
                user = request.user
            else:
                # 기본 사용자 생성 또는 가져오기
                from chat.models import User
                user, created = User.objects.get_or_create(
                    username='anonymous',
                    defaults={'email': 'anonymous@example.com'}
                )
                print(f"✅ 기본 사용자 생성/가져오기: {user.username}")
            
            if video_id:
                # 특정 영상의 채팅 세션 조회
                sessions = VideoChatSession.objects.filter(
                    user=user, 
                    video_id=video_id,
                    is_active=True
                ).order_by('-created_at')
            else:
                # 사용자의 모든 채팅 세션 조회
                sessions = VideoChatSession.objects.filter(
                    user=user,
                    is_active=True
                ).order_by('-created_at')
            
            serializer = VideoChatSessionSerializer(sessions, many=True)
            return Response({
                'sessions': serializer.data,
                'total_count': sessions.count()
            })
            
        except Exception as e:
            return Response({
                'error': f'채팅 세션 조회 중 오류 발생: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    def post(self, request, video_id):
        """영상 채팅 메시지 전송"""
        try:
            print(f"🔍 VideoChatView POST 요청 - video_id: {video_id}")
            # Django WSGIRequest에서 JSON 데이터 파싱
            import json
            if hasattr(request, 'data'):
                message = request.data.get('message')
            else:
                body = request.body.decode('utf-8')
                data = json.loads(body)
                message = data.get('message')
            print(f"📝 메시지: {message}")
            
            if not message:
                return Response({
                    'error': '메시지가 필요합니다'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # 영상 분석 상태 확인 (Video 모델에서 직접 확인)
            try:
                video = Video.objects.get(id=video_id)
                if video.analysis_status == 'pending':
                    return Response({
                        'error': '영상 분석이 진행 중입니다. 잠시 후 다시 시도해주세요.',
                        'status': 'analyzing'
                    }, status=status.HTTP_202_ACCEPTED)
                elif video.analysis_status == 'failed':
                    return Response({
                        'error': '영상 분석에 실패했습니다. 다른 영상을 업로드해주세요.',
                        'status': 'failed'
                    }, status=status.HTTP_400_BAD_REQUEST)
            except Video.DoesNotExist:
                return Response({
                    'error': '영상을 찾을 수 없습니다'
                }, status=status.HTTP_404_NOT_FOUND)
            
            # 사용자 정보 처리 (인증되지 않은 경우 기본 사용자 사용)
            user = request.user if request.user.is_authenticated else None
            if not user:
                # 기본 사용자 생성 또는 가져오기
                from chat.models import User
                user, created = User.objects.get_or_create(
                    username='anonymous',
                    defaults={'email': 'anonymous@example.com'}
                )
            
            # 채팅 세션 가져오기 또는 생성
            session, created = VideoChatSession.objects.get_or_create(
                user=user,
                video_id=video_id,
                is_active=True,
                defaults={
                    'video_title': f"Video {video_id}",
                    'video_analysis_data': {}
                }
            )
            
            # 사용자 메시지 저장
            user_message = VideoChatMessage.objects.create(
                session=session,
                message_type='user',
                content=message
            )
            
            # 영상 분석 데이터 가져오기 (Video 모델에서 직접)
            analysis_data = {
                'original_name': video.original_name,
                'file_size': video.file_size,
                'uploaded_at': video.uploaded_at.isoformat(),
                'analysis_status': video.analysis_status,
                'duration': video.duration,
                'is_analyzed': video.is_analyzed
            }
            
            # JSON 분석 결과 로드
            analysis_json_data = None
            if video.analysis_json_path:
                try:
                    from django.conf import settings
                    json_path = os.path.join(settings.MEDIA_ROOT, video.analysis_json_path)
                    print(f"🔍 JSON 파일 경로: {json_path}")
                    print(f"🔍 파일 존재 여부: {os.path.exists(json_path)}")
                    
                    with open(json_path, 'r', encoding='utf-8') as f:
                        analysis_json_data = json.load(f)
                    print(f"✅ JSON 분석 결과 로드 성공: {json_path}")
                    print(f"📊 JSON 데이터 키: {list(analysis_json_data.keys())}")
                    if 'frame_results' in analysis_json_data:
                        print(f"📊 frame_results 개수: {len(analysis_json_data['frame_results'])}")
                        if analysis_json_data['frame_results']:
                            print(f"📊 첫 번째 프레임: {analysis_json_data['frame_results'][0]}")
                except Exception as e:
                    print(f"❌ JSON 분석 결과 로드 실패: {e}")
                    import traceback
                    print(f"❌ 상세 오류: {traceback.format_exc()}")
            else:
                print("❌ analysis_json_path가 없습니다.")
                print(f"❌ video.analysis_json_path: {video.analysis_json_path}")
            
            # 프레임 검색 및 이미지 URL 생성
            print(f"🔍 프레임 검색 시작 - analysis_json_data: {analysis_json_data is not None}")
            if analysis_json_data:
                print(f"📊 frame_results 존재: {'frame_results' in analysis_json_data}")
                if 'frame_results' in analysis_json_data:
                    print(f"📊 frame_results 개수: {len(analysis_json_data['frame_results'])}")
            else:
                print("❌ analysis_json_data가 None입니다!")
                print(f"❌ video.analysis_json_path: {video.analysis_json_path}")
                print(f"❌ video.analysis_status: {video.analysis_status}")
                print(f"❌ video.is_analyzed: {video.is_analyzed}")
            
            relevant_frames = self._find_relevant_frames(message, analysis_json_data, video_id)
            print(f"🔍 검색된 프레임 수: {len(relevant_frames)}")
            if relevant_frames:
                print(f"📸 첫 번째 프레임: {relevant_frames[0]}")
                print(f"📸 모든 프레임 정보:")
                for i, frame in enumerate(relevant_frames):
                    print(f"  프레임 {i+1}: {frame}")
            else:
                print("❌ 검색된 프레임이 없습니다!")
                print(f"❌ analysis_json_data keys: {list(analysis_json_data.keys()) if analysis_json_data else 'None'}")
                if analysis_json_data and 'frame_results' in analysis_json_data:
                    print(f"❌ frame_results 개수: {len(analysis_json_data['frame_results'])}")
                    if analysis_json_data['frame_results']:
                        print(f"❌ 첫 번째 frame_result: {analysis_json_data['frame_results'][0]}")
            
            # 다중 AI 응답 생성
            ai_responses = {}
            individual_messages = []
            
            # 기본 채팅 시스템과 동일한 AI 모델 초기화
            try:
                # 전역 chatbots 변수 사용 (이미 초기화되어 있음)
                print(f"✅ 사용 가능한 AI 모델: {list(chatbots.keys())}")
            except Exception as e:
                print(f"⚠️ AI 모델 초기화 실패: {e}")
                # 전역 chatbots 변수는 이미 초기화되어 있으므로 덮어쓰지 않음
            
            # AI 모델 확인
            print(f"🤖 사용 가능한 AI 모델: {list(chatbots.keys()) if chatbots else 'None'}")
            
            # AI 모델이 없는 경우 기본 응답 (프레임 정보 포함)
            if not chatbots:
                print("⚠️ 사용 가능한 AI 모델이 없습니다. 기본 응답을 생성합니다.")
                
                # 프레임 정보를 포함한 더 나은 응답 생성
                if relevant_frames:
                    frame_count = len(relevant_frames)
                    default_response = f"영상에서 '{message}'와 관련된 {frame_count}개의 프레임을 찾았습니다!\n\n"
                    
                    for i, frame in enumerate(relevant_frames, 1):
                        default_response += f"📸 프레임 {i}:\n"
                        default_response += f"   ⏰ 시간: {frame['timestamp']:.1f}초\n"
                        default_response += f"   🎯 관련도: {frame['relevance_score']}점\n"
                        
                        if frame['persons'] and len(frame['persons']) > 0:
                            default_response += f"   👤 사람 {len(frame['persons'])}명 감지\n"
                        
                        if frame['objects'] and len(frame['objects']) > 0:
                            default_response += f"   📦 객체 {len(frame['objects'])}개 감지\n"
                        
                        scene_attrs = frame.get('scene_attributes', {})
                        if scene_attrs:
                            scene_type = scene_attrs.get('scene_type', 'unknown')
                            lighting = scene_attrs.get('lighting', 'unknown')
                            activity = scene_attrs.get('activity_level', 'unknown')
                            default_response += f"   🏞️ 장면: {scene_type}, 조명: {lighting}, 활동: {activity}\n"
                        
                        default_response += "\n"
                    
                    default_response += "💡 AI 모델이 활성화되면 더 자세한 분석을 제공할 수 있습니다."
                else:
                    default_response = f"죄송합니다. '{message}'와 관련된 프레임을 찾을 수 없습니다.\n\n"
                    default_response += "다른 키워드로 시도해보세요:\n"
                    default_response += "• 사람, 자동차, 동물, 음식, 옷, 건물, 자연, 물체"
                
                ai_responses = {
                    'default': default_response
                }
            else:
                # 각 AI 모델에 질문 전송
                for bot_name, chatbot in chatbots.items():
                    if bot_name == 'optimal':
                        continue  # optimal은 나중에 처리
                    
                    try:
                        # 색상 검색 모드 확인
                        is_color_search = any(keyword in message.lower() for keyword in ['빨간색', '파란색', '노란색', '초록색', '보라색', '분홍색', '검은색', '흰색', '회색', '주황색', '갈색', '옷'])
                        
                        # 영상 정보와 프레임 정보를 포함한 프롬프트 생성
                        video_context = f"""
영상 정보:
- 파일명: {analysis_data.get('original_name', 'Unknown')}
- 파일 크기: {analysis_data.get('file_size', 0) / (1024*1024):.1f}MB
- 업로드 시간: {analysis_data.get('uploaded_at', 'Unknown')}
- 상태: {analysis_data.get('analysis_status', 'Unknown')}
"""
                        
                        # 관련 프레임 정보 추가 (색상 검색 모드에 따라 다르게 처리)
                        frame_context = ""
                        if relevant_frames:
                            if is_color_search:
                                frame_context = "\n\n관련 프레임 정보 (색상 분석 필요):\n"
                                frame_context += "⚠️ 중요: 현재 분석 결과에는 색상 정보가 포함되어 있지 않습니다.\n"
                                frame_context += "하지만 실제 프레임 이미지들을 통해 색상을 직접 확인할 수 있습니다.\n\n"
                                
                                for i, frame in enumerate(relevant_frames, 1):
                                    frame_context += f"프레임 {i}: 시간 {frame['timestamp']:.1f}초\n"
                                    frame_context += f"  - 이미지 URL: {frame['image_url']}\n"
                                    frame_context += f"  - 실제 파일 경로: {frame.get('actual_image_path', 'N/A')}\n"
                                    
                                    # 색상 분석 결과 추가
                                    dominant_colors = frame.get('dominant_colors', [])
                                    if dominant_colors:
                                        frame_context += f"  - 색상 분석 결과: {dominant_colors}\n"
                                        color_match = frame.get('color_search_info', {}).get('color_match_found', False)
                                        frame_context += f"  - 색상 매칭: {'✅ 발견' if color_match else '❌ 없음'}\n"
                                    else:
                                        frame_context += f"  - 색상 분석 결과: 없음\n"
                                    
                                    # 실제 이미지 파일을 base64로 인코딩하여 포함
                                    actual_image_path = frame.get('actual_image_path')
                                    if actual_image_path and os.path.exists(actual_image_path):
                                        try:
                                            import base64
                                            with open(actual_image_path, 'rb') as img_file:
                                                img_data = img_file.read()
                                                img_base64 = base64.b64encode(img_data).decode('utf-8')
                                                # 이미지 크기가 너무 크면 URL만 제공
                                                if len(img_base64) > 100000:  # 100KB 제한
                                                    frame_context += f"  - 이미지 URL (직접 확인 필요): {frame['image_url']}\n"
                                                    print(f"⚠️ 프레임 {i} 이미지가 너무 커서 URL만 제공 (크기: {len(img_base64)} 문자)")
                                                else:
                                                    frame_context += f"  - 실제 이미지 (base64): data:image/jpeg;base64,{img_base64}\n"
                                                    print(f"✅ 프레임 {i} 이미지 base64 인코딩 완료 (크기: {len(img_base64)} 문자)")
                                        except Exception as e:
                                            frame_context += f"  - 이미지 로드 실패: {str(e)}\n"
                                            print(f"❌ 프레임 {i} 이미지 로드 실패: {str(e)}")
                                    
                                    if frame['persons']:
                                        frame_context += f"  - 사람 {len(frame['persons'])}명 감지됨!\n"
                                        for j, person in enumerate(frame['persons'], 1):
                                            confidence = person.get('confidence', 0)
                                            bbox = person.get('bbox', [])
                                            frame_context += f"    사람 {j}: 신뢰도 {confidence:.2f}, 위치 {bbox}\n"
                                    frame_context += "\n"
                                
                                frame_context += "💡 각 프레임 이미지를 직접 확인하여 요청하신 색상의 옷을 입은 사람이 있는지 분석해주세요.\n"
                                frame_context += f"🔗 이미지 접근 방법: 각 프레임의 이미지 URL을 브라우저에서 열어서 직접 확인할 수 있습니다.\n"
                                frame_context += f"📋 분석 요청: 위 이미지들을 보고 '{message}'에서 요청한 색상의 옷을 입은 사람이 있는지 정확히 분석해주세요.\n"
                                frame_context += f"🎨 색상 분석 결과: 위에서 제공된 색상 분석 결과를 참고하여 요청된 색상과 일치하는지 확인해주세요.\n"
                            else:
                                frame_context = "\n\n관련 프레임 정보 (사람 감지됨):\n"
                                for i, frame in enumerate(relevant_frames, 1):
                                    frame_context += f"프레임 {i}: 시간 {frame['timestamp']:.1f}초, 관련도 {frame['relevance_score']}점\n"
                                    if frame['persons']:
                                        frame_context += f"  - 사람 {len(frame['persons'])}명 감지됨!\n"
                                        # 각 사람의 상세 정보 추가
                                        for j, person in enumerate(frame['persons'], 1):
                                            confidence = person.get('confidence', 0)
                                            bbox = person.get('bbox', [])
                                            frame_context += f"    사람 {j}: 신뢰도 {confidence:.2f}, 위치 {bbox}\n"
                                            # 속성 정보 추가
                                            attrs = person.get('attributes', {})
                                            if 'gender' in attrs:
                                                gender_info = attrs['gender']
                                                frame_context += f"      성별: {gender_info.get('value', 'unknown')} (신뢰도: {gender_info.get('confidence', 0):.2f})\n"
                                            if 'age' in attrs:
                                                age_info = attrs['age']
                                                frame_context += f"      나이: {age_info.get('value', 'unknown')} (신뢰도: {age_info.get('confidence', 0):.2f})\n"
                                    if frame['objects']:
                                        frame_context += f"  - 객체 {len(frame['objects'])}개 감지\n"
                                    scene_attrs = frame.get('scene_attributes', {})
                                    if scene_attrs:
                                        frame_context += f"  - 장면: {scene_attrs.get('scene_type', 'unknown')}, 조명: {scene_attrs.get('lighting', 'unknown')}\n"
                                    frame_context += "\n"
                        else:
                            frame_context = "\n\n관련 프레임 정보: 사람이 감지된 프레임이 없습니다.\n"
                        
                        enhanced_message = f"""{video_context}{frame_context}

사용자 질문: "{message}"

위 영상 분석 정보를 바탕으로 사용자의 질문에 정확하고 도움이 되는 답변을 제공해주세요.

답변 시 다음을 포함해주세요:
1. 질문에 대한 직접적인 답변
2. 관련 프레임의 구체적인 정보 (시간, 내용 등)
3. 영상에서 관찰할 수 있는 세부사항
4. 추가로 확인할 수 있는 다른 요소들

답변은 한국어로 작성하고, 구체적이고 실용적인 정보를 제공해주세요.

중요: 위 프레임 정보에서 사람이 감지되었다면, 반드시 그 사실을 명확히 언급하고 구체적인 정보를 제공해주세요. 사람이 감지되지 않았다면 그 사실도 명확히 말해주세요.

"🎨 색상 검색 모드: 위에서 제공된 프레임 이미지들을 직접 확인하여 요청하신 색상의 옷을 입은 사람이 있는지 분석해주세요. 각 프레임의 실제 이미지(base64)를 직접 보고 색상을 분석해주세요.

⚠️ 중요: 현재 분석 시스템은 색상 정보를 제공하지 않으므로, 반드시 실제 이미지를 직접 확인하여 색상을 분석해야 합니다. 분석 결과에 색상 정보가 없다고 해서 해당 색상의 옷을 입은 사람이 없다고 결론내리지 마세요. 실제 이미지를 보고 정확한 색상을 분석해주세요.

🎯 특별 지시: 각 프레임 이미지에서 실제로 보이는 색상을 정확히 분석하고, 요청된 색상과 일치하는지 판단해주세요. 배경에 있는 사람들도 놓치지 말고 확인해주세요. 

📸 이미지 분석: 위에 제공된 base64 이미지들을 직접 보고, 분홍색 옷을 입은 사람이 있는지 정확히 분석해주세요." if is_color_search else """""
                        
                        # 기본 채팅 시스템과 동일한 방식으로 응답 생성
                        ai_response = chatbot.chat(enhanced_message)
                        ai_responses[bot_name] = ai_response
                        
                        # 개별 AI 응답 저장
                        ai_message = VideoChatMessage.objects.create(
                            session=session,
                            message_type='ai',
                            content=ai_response,
                            ai_model=bot_name,
                            parent_message=user_message
                        )
                        individual_messages.append(ai_message)
                        
                    except Exception as e:
                        print(f"AI {bot_name} 응답 생성 실패: {str(e)}")
                        continue
            
            # 통합 응답 생성 (기본 채팅 시스템과 동일한 방식)
            optimal_response = ""
            if ai_responses and len(ai_responses) > 1:
                try:
                    # 기본 채팅 시스템의 generate_optimal_response 사용
                    optimal_response = generate_optimal_response(ai_responses, message, os.getenv('OPENAI_API_KEY'))
                    
                    # 프레임 정보 추가 (더 자세한 정보 포함)
                    if relevant_frames:
                        frame_summary = f"\n\n📸 관련 프레임 {len(relevant_frames)}개 발견:\n"
                        for i, frame in enumerate(relevant_frames, 1):
                            frame_summary += f"• 프레임 {i}: {frame['timestamp']:.1f}초 (관련도 {frame['relevance_score']:.2f}점)\n"
                            
                            # 프레임별 세부 정보 추가
                            if frame.get('persons'):
                                frame_summary += f"  👤 사람 {len(frame['persons'])}명 감지됨!\n"
                                # 각 사람의 상세 정보 추가
                                for j, person in enumerate(frame['persons'], 1):
                                    confidence = person.get('confidence', 0)
                                    frame_summary += f"    사람 {j}: 신뢰도 {confidence:.2f}\n"
                                    # 속성 정보 추가
                                    attrs = person.get('attributes', {})
                                    if 'gender' in attrs:
                                        gender_info = attrs['gender']
                                        frame_summary += f"      성별: {gender_info.get('value', 'unknown')}\n"
                                    if 'age' in attrs:
                                        age_info = attrs['age']
                                        frame_summary += f"      나이: {age_info.get('value', 'unknown')}\n"
                            if frame.get('objects'):
                                frame_summary += f"  📦 객체 {len(frame['objects'])}개 감지\n"
                            
                            scene_attrs = frame.get('scene_attributes', {})
                            if scene_attrs:
                                scene_type = scene_attrs.get('scene_type', 'unknown')
                                lighting = scene_attrs.get('lighting', 'unknown')
                                frame_summary += f"  🏞️ 장면: {scene_type}, 조명: {lighting}\n"
                        
                        frame_summary += "\n💡 위 프레임들을 참고하여 영상에서 해당 내용을 확인해보세요."
                        optimal_response += frame_summary
                    
                    # 통합 응답 저장
                    optimal_message = VideoChatMessage.objects.create(
                        session=session,
                        message_type='ai_optimal',
                        content=optimal_response,
                        ai_model='optimal',
                        parent_message=user_message
                    )
                    
                except Exception as e:
                    print(f"통합 응답 생성 실패: {str(e)}")
                    optimal_response = f"통합 응답 생성 중 오류가 발생했습니다: {str(e)}"
            elif ai_responses and len(ai_responses) == 1:
                # AI 응답이 하나만 있는 경우
                optimal_response = list(ai_responses.values())[0]
            
            # 응답 데이터 구성
            response_data = {
                'session_id': str(session.id),
                'user_message': {
                    'id': str(user_message.id),
                    'content': message,
                    'created_at': user_message.created_at
                },
                'ai_responses': {
                    'individual': [
                        {
                            'id': str(msg.id),
                            'model': msg.ai_model,
                            'content': msg.content,
                            'created_at': msg.created_at
                        } for msg in individual_messages
                    ],
                    'optimal': {
                        'content': optimal_response,
                        'created_at': individual_messages[0].created_at if individual_messages else None
                    } if optimal_response else None
                },
                'relevant_frames': relevant_frames  # 관련 프레임 정보 추가
            }
            
            # 디버깅: relevant_frames 확인
            print(f"🔍 응답 생성 시 relevant_frames: {len(relevant_frames)}")
            if relevant_frames:
                print(f"📸 첫 번째 프레임: {relevant_frames[0]}")
            else:
                print("❌ relevant_frames가 비어있음!")
            
            print(f"📤 응답에 포함될 프레임 수: {len(relevant_frames)}")
            if relevant_frames:
                print(f"📸 첫 번째 프레임: {relevant_frames[0]}")
            
            return Response(response_data)
            
        except Exception as e:
            import traceback
            print(f"❌ VideoChatView POST 오류: {str(e)}")
            print(f"❌ 오류 상세: {traceback.format_exc()}")
            return Response({
                'error': f'채팅 처리 중 오류 발생: {str(e)}',
                'traceback': traceback.format_exc()
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def _find_relevant_frames(self, message, analysis_json_data, video_id):
        """사용자 메시지에 따라 관련 프레임을 찾아서 이미지 URL과 함께 반환"""
        try:
            if not analysis_json_data or 'frame_results' not in analysis_json_data:
                print("❌ 분석 데이터 또는 프레임 결과가 없습니다.")
                return []
            
            relevant_frames = []
            message_lower = message.lower()
            
            # 프레임 결과에서 매칭되는 프레임 찾기
            frame_results = analysis_json_data.get('frame_results', [])
            print(f"🔍 검색할 프레임 수: {len(frame_results)}")
            
            # 색상 기반 검색
            color_keywords = {
                '빨간색': ['red', '빨강', '빨간색'],
                '파란색': ['blue', '파랑', '파란색'],
                '노란색': ['yellow', '노랑', '노란색'],
                '초록색': ['green', '녹색', '초록색'],
                '보라색': ['purple', '자주색', '보라색'],
                '분홍색': ['pink', '핑크', '분홍색'],
                '검은색': ['black', '검정', '검은색'],
                '흰색': ['white', '하양', '흰색'],
                '회색': ['gray', 'grey', '회색'],
                '주황색': ['orange', '오렌지', '주황색'],
                '갈색': ['brown', '브라운', '갈색'],
                '옷': ['clothing', 'clothes', 'dress', 'shirt', 'pants', 'jacket']
            }
            
            # 색상 검색 모드 확인
            is_color_search = False
            detected_colors = []
            for color_korean, color_terms in color_keywords.items():
                if any(term in message_lower for term in color_terms):
                    is_color_search = True
                    detected_colors.append(color_korean)
                    print(f"🎨 색상 검색 감지: {color_korean}")
            
            # 색상 검색 모드 (우선순위)
            if is_color_search:
                print(f"🎨 색상 검색 모드: {detected_colors}")
                print(f"🔍 검색할 프레임 수: {len(frame_results)}")
                for frame in frame_results:
                    persons = frame.get('persons', [])
                    
                    # 색상 분석 결과 확인
                    dominant_colors = frame.get('dominant_colors', [])
                    color_match_found = False
                    
                    # 요청된 색상과 매칭되는지 확인 (더 유연한 매칭)
                    for detected_color in detected_colors:
                        for color_info in dominant_colors:
                            color_name = color_info.get('color', '').lower()
                            detected_color_lower = detected_color.lower()
                            
                            # 색상 키워드 매핑을 통한 매칭
                            color_mapping = {
                                '분홍색': 'pink', '핑크': 'pink',
                                '빨간색': 'red', '빨강': 'red',
                                '파란색': 'blue', '파랑': 'blue',
                                '노란색': 'yellow', '노랑': 'yellow',
                                '초록색': 'green', '녹색': 'green',
                                '보라색': 'purple', '자주색': 'purple',
                                '검은색': 'black', '검정': 'black',
                                '흰색': 'white', '하양': 'white',
                                '회색': 'gray', 'grey': 'gray',
                                '주황색': 'orange', '오렌지': 'orange',
                                '갈색': 'brown', '브라운': 'brown'
                            }
                            
                            # 매핑된 색상으로 비교
                            mapped_color = color_mapping.get(detected_color_lower, detected_color_lower)
                            
                            # 정확한 매칭 또는 부분 매칭
                            if (mapped_color == color_name or 
                                detected_color_lower == color_name or 
                                detected_color_lower in color_name or 
                                color_name in detected_color_lower):
                                color_match_found = True
                                print(f"✅ 색상 매칭 발견: {detected_color} -> {color_info}")
                                break
                        if color_match_found:
                            break
                    
                    # 디버깅을 위한 로그 추가
                    print(f"🔍 프레임 {frame.get('image_id', 0)} 색상 분석:")
                    print(f"  - 요청된 색상: {detected_colors}")
                    print(f"  - 감지된 색상: {[c.get('color', '') for c in dominant_colors]}")
                    print(f"  - 매칭 결과: {color_match_found}")
                    
                    # 색상 검색의 경우 색상 매칭이 된 프레임만 포함
                    if color_match_found:
                        frame_image_path = frame.get('frame_image_path', '')
                        actual_image_path = None
                        if frame_image_path:
                            # 실제 파일 시스템 경로 생성
                            import os
                            from django.conf import settings
                            actual_image_path = os.path.join(settings.MEDIA_ROOT, frame_image_path)
                            if os.path.exists(actual_image_path):
                                print(f"✅ 실제 이미지 파일 존재: {actual_image_path}")
                            else:
                                print(f"❌ 실제 이미지 파일 없음: {actual_image_path}")
                        
                        frame_info = {
                            'image_id': frame.get('image_id', 0),
                            'timestamp': frame.get('timestamp', 0),
                            'frame_image_path': frame_image_path,
                            'image_url': f'/media/{frame_image_path}',
                            'actual_image_path': actual_image_path,  # 실제 파일 경로 추가
                            'persons': persons,
                            'objects': frame.get('objects', []),
                            'scene_attributes': frame.get('scene_attributes', {}),
                            'dominant_colors': dominant_colors,  # 색상 분석 결과 추가
                            'relevance_score': 2,  # 색상 매칭 시 높은 점수
                            'color_search_info': {
                                'requested_colors': detected_colors,
                                'color_info_available': len(dominant_colors) > 0,
                                'color_match_found': color_match_found,
                                'actual_image_available': actual_image_path is not None,
                                'message': f"색상 분석 결과: {dominant_colors} | 요청하신 색상: {', '.join(detected_colors)}"
                            }
                        }
                        relevant_frames.append(frame_info)
                        print(f"✅ 프레임 {frame_info['image_id']} 추가 (색상 매칭 성공)")
                    else:
                        print(f"❌ 프레임 {frame.get('image_id', 0)}: 색상 매칭 실패 - {detected_colors} vs {dominant_colors}")
            
            # 사람 검색 모드
            elif any(keyword in message_lower for keyword in ['사람', 'person', 'people', 'human', '찾아', '보여']):
                print("👤 사람 검색 모드")
                print(f"🔍 검색할 프레임 수: {len(frame_results)}")
                for frame in frame_results:
                    persons = frame.get('persons', [])
                    print(f"🔍 프레임 {frame.get('image_id', 0)}: persons = {persons}")
                    # 사람이 감지된 프레임만 포함
                    if persons and len(persons) > 0:
                        frame_info = {
                            'image_id': frame.get('image_id', 0),
                            'timestamp': frame.get('timestamp', 0),
                            'frame_image_path': frame.get('frame_image_path', ''),
                            'image_url': f'/media/{frame.get("frame_image_path", "")}',
                            'persons': persons,
                            'objects': frame.get('objects', []),
                            'scene_attributes': frame.get('scene_attributes', {}),
                            'relevance_score': len(persons) * 2  # 사람 수에 비례한 점수
                        }
                        relevant_frames.append(frame_info)
                        print(f"✅ 프레임 {frame_info['image_id']} 추가 (사람 {len(persons)}명 감지)")
                        print(f"✅ 프레임 상세 정보: {frame_info}")
                    else:
                        print(f"❌ 프레임 {frame.get('image_id', 0)}: 사람 감지 안됨")
            
            # 다른 키워드 검색
            else:
                search_keywords = {
                    '자동차': ['car', 'vehicle', 'automobile'],
                    '동물': ['animal', 'dog', 'cat', 'pet'],
                    '음식': ['food', 'meal', 'eat', 'drink'],
                    '옷': ['clothing', 'clothes', 'dress', 'shirt'],
                    '건물': ['building', 'house', 'structure'],
                    '자연': ['nature', 'tree', 'sky', 'mountain'],
                    '물체': ['object', 'item', 'thing']
                }
                
                # 한국어 키워드 추출
                matched_keywords = []
                for korean_key, english_keywords in search_keywords.items():
                    if korean_key in message_lower:
                        matched_keywords.extend(english_keywords)
                
                for frame in frame_results:
                    frame_score = 0
                    frame_info = {
                        'image_id': frame.get('image_id', 0),
                        'timestamp': frame.get('timestamp', 0),
                        'frame_image_path': frame.get('frame_image_path', ''),
                        'image_url': f'/media/{frame.get("frame_image_path", "")}',
                        'persons': frame.get('persons', []),
                        'objects': frame.get('objects', []),
                        'scene_attributes': frame.get('scene_attributes', {}),
                        'relevance_score': 0
                    }
                    
                    # 객체 검색
                    for obj in frame_info['objects']:
                        obj_class = obj.get('class', '').lower()
                        if any(keyword in obj_class for keyword in matched_keywords):
                            frame_score += 5
                    
                    # 장면 속성 검색
                    scene_attrs = frame_info['scene_attributes']
                    if 'outdoor' in message_lower and scene_attrs.get('scene_type') == 'outdoor':
                        frame_score += 3
                    if 'indoor' in message_lower and scene_attrs.get('scene_type') == 'indoor':
                        frame_score += 3
                    if 'bright' in message_lower and scene_attrs.get('lighting') == 'bright':
                        frame_score += 2
                    if 'dark' in message_lower and scene_attrs.get('lighting') == 'dark':
                        frame_score += 2
                    
                    if frame_score > 0:
                        frame_info['relevance_score'] = frame_score
                        relevant_frames.append(frame_info)
                        print(f"✅ 프레임 {frame_info['image_id']} 추가 (점수: {frame_score})")
            
            # 관련도 점수순으로 정렬하고 상위 3개만 반환
            relevant_frames.sort(key=lambda x: x['relevance_score'], reverse=True)
            result = relevant_frames[:3]
            print(f"🎯 최종 선택된 프레임 수: {len(result)}")
            print(f"🎯 최종 프레임 상세: {result}")
            return result
            
        except Exception as e:
            print(f"❌ 프레임 검색 실패: {e}")
            return []

class FrameImageView(APIView):
    """프레임 이미지 서빙"""
    permission_classes = [AllowAny]
    
    def get(self, request, video_id, frame_number):
        try:
            from django.conf import settings
            # 프레임 이미지 경로 생성
            frame_filename = f"video{video_id}_frame{frame_number}.jpg"
            frame_path = os.path.join(settings.MEDIA_ROOT, 'images', frame_filename)
            
            # 파일이 존재하는지 확인
            if not os.path.exists(frame_path):
                raise Http404("프레임 이미지를 찾을 수 없습니다")
            
            # 이미지 파일 읽기
            with open(frame_path, 'rb') as f:
                image_data = f.read()
            
            # HTTP 응답으로 이미지 반환
            response = HttpResponse(image_data, content_type='image/jpeg')
            response['Content-Disposition'] = f'inline; filename="{frame_filename}"'
            return response
            
        except Exception as e:
            return Response({
                'error': f'프레임 이미지 로드 실패: {str(e)}'
            }, status=status.HTTP_404_NOT_FOUND)
