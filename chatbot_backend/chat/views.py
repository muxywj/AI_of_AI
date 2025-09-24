from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from chat.serializers import UserSerializer
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
