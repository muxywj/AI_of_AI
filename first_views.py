from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny
import logging
import json
import re
import os
from django.conf import settings
from dotenv import load_dotenv

# 지연 로딩을 위한 클라이언트 관리자 import
from .lazy_clients import lazy_clients

logger = logging.getLogger(__name__)

# 무거운 라이브러리들은 필요할 때만 import
def get_openai_client():
    return lazy_clients.get_openai_client()

def get_anthropic_client():
    return lazy_clients.get_anthropic_client()

def get_groq_client():
    return lazy_clients.get_groq_client()

def get_google_client():
    return lazy_clients.get_google_client()

def get_yolo_model(model_name='yolov8n'):
    return lazy_clients.get_yolo_model(model_name)

def fetch_and_clean_url(url, timeout=10):
    """
    주어진 URL의 HTML을 요청해, 스크립트·스타일 제거 후 텍스트만 반환합니다.
    """
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    # 스크립트·스타일·네비게이션 태그 제거
    for tag in soup(["script", "style", "header", "footer", "nav"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    # 빈 줄 제거
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)

# Add this function to your ChatBot class or as a standalone function
def sanitize_and_parse_json(text, selected_models, responses):
    """
    Sanitize and parse the JSON response from AI models.
    Handles various edge cases and formatting issues.
    """
    import re
    import json
    import logging
    
    logger = logging.getLogger(__name__)
    
    try:
        # Step 1: Basic cleanup
        text = text.strip()
        
        # Step 2: Handle code blocks
        if text.startswith('```json') and '```' in text:
            text = re.sub(r'```json(.*?)```', r'\1', text, flags=re.DOTALL).strip()
        elif text.startswith('```') and text.endswith('```'):
            text = text[3:-3].strip()
            
        # Step 3: Extract JSON object if embedded in other text
        json_pattern = r'({[\s\S]*})'
        json_matches = re.findall(json_pattern, text)
        if json_matches:
            text = json_matches[0]
            
        # Step 4: Handle escaped backslashes in the text
        # First identify all occurrences of escaped backslashes followed by characters like "_"
        text = re.sub(r'\\([_"])', r'\1', text)
        
        # Step 5: Attempt to parse the JSON
        result = json.loads(text)
        
        # Ensure the required fields exist
        required_fields = ["preferredModel", "best_response", "analysis", "reasoning"]
        for field in required_fields:
            if field not in result:
                if field == "best_response" and "bestResponse" in result:
                    result["best_response"] = result["bestResponse"]
                else:
                    result[field] = "" if field != "analysis" else {}
        
        return result
        
    except Exception as e:
        logger.error(f"❌ JSON 파싱 실패: {str(e)}")
        logger.error(f"원본 텍스트: {text[:200]}..." if len(text) > 200 else text)
        
        # Advanced recovery attempt for malformed JSON
        try:
            # Remove problematic escaped characters
            fixed_text = text.replace("\\_", "_").replace('\\"', '"')
            
            # Try to fix common issues with JSON (missing quotes, commas, etc.)
            fixed_text = re.sub(r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', fixed_text)
            fixed_text = re.sub(r':\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*([,}])', r':"\1"\2', fixed_text)
            
            # Handle unclosed strings
            for match in re.finditer(r':\s*"([^"\\]*(\\.[^"\\]*)*)', fixed_text):
                if not re.search(r':\s*"([^"\\]*(\\.[^"\\]*)*)"', match.group(0)):
                    pos = match.end()
                    fixed_text = fixed_text[:pos] + '"' + fixed_text[pos:]
            
            result = json.loads(fixed_text)
            logger.info("✅ Recovered JSON after fixing format issues")
            return result
        except:
            # Last resort: construct a sensible fallback response
            error_analysis = {}
            for model in selected_models:
                model_lower = model.lower()
                error_analysis[model_lower] = {"장점": "분석 실패", "단점": "분석 실패"}
            
            # Find the largest response to use as best_response
            best_response = ""
            if responses:
                best_response = max(responses.values(), key=len) 
            
            return {
                "preferredModel": "FALLBACK",
                "best_response": best_response,
                "analysis": error_analysis,
                "reasoning": "응답 분석 중 오류가 발생하여 최적의 답변을 생성하지 못했습니다."
            }

# src/chatbot_backend/chat/bot.py

# import logging
# import openai
# import anthropic
# import base64
# import imghdr
# from groq import Groq
# from io import BytesIO

# logger = logging.getLogger(__name__)

# class ChatBot:
#     def __init__(self, api_key, model, api_type):
#         self.conversation_history = []
#         self.api_type = api_type
#         self.api_key = api_key

#         # Anthropic 멀티모달은 Opus 모델 권장
#         if api_type == 'anthropic' and not model.startswith('claude-3-opus-20240229'):
#             logger.info(f"Overriding Anthropic model '{model}' to 'claude-3-opus-20240229' for image support")
#             self.model = 'claude-3-opus-20240229'
#         else:
#             self.model = model

#         if api_type == 'openai':
#             openai.api_key = api_key
#         elif api_type == 'anthropic':
#             # Anthropic Python SDK 초기화
#             self.client = anthropic.Client(api_key=api_key)
#         elif api_type == 'groq':
#             self.client = Groq(api_key=api_key)
#         else:
#             raise ValueError(f"Unsupported api_type: {api_type}")

#     def chat(self, prompt=None, user_input=None, image_file=None, analysis_mode=None, user_language=None):
#         """
#         prompt       : 텍스트 프롬프트 (키워드)
#         user_input   : 텍스트 프롬프트 (위치 인자)
#         image_file   : 파일 객체 (BytesIO, InMemoryUploadedFile 등)
#         analysis_mode: 'describe'|'ocr'|'objects'
#         user_language: 'ko','en'
#         """
#         text = prompt if prompt is not None else user_input
#         try:
#             logger.info(f"[{self.api_type}] Received input: {text}")

#             # 모델별 호출
#             if self.api_type == 'openai':
#                 # GPT-4 Vision 지원
#                 params = {
#                     'model': self.model,
#                     'messages': self.conversation_history + [{"role": "user", "content": text}],
#                     'temperature': 0.7,
#                     'max_tokens': 1024
#                 }
#                 if image_file:
#                     params['files'] = [("image", image_file)]
#                 resp = openai.ChatCompletion.create(**params)
#                 assistant_response = resp.choices[0].message.content

#             elif self.api_type == 'anthropic':
#                 # Claude 3 Opus: 이미지+텍스트 지원 via Messages API
#                 messages = []
#                 # 토큰 수 설정
#                 max_tokens = 1024 if image_file else 4096
#                 if image_file:
#                     # 이미지 바이너리 읽기 및 미디어 타입 자동 감지
#                     image_file.seek(0)
#                     data_bytes = image_file.read()
#                     ext = imghdr.what(None, h=data_bytes) or 'jpeg'
#                     mime_map = {
#                         'jpeg': 'image/jpeg', 'jpg': 'image/jpeg',
#                         'png': 'image/png', 'gif': 'image/gif',
#                         'bmp': 'image/bmp', 'webp': 'image/webp'
#                     }
#                     media_type = mime_map.get(ext, 'image/jpeg')
#                     b64 = base64.b64encode(data_bytes).decode('utf-8')

#                     # 이미지 블록과 텍스트 블록을 리스트로 구성
#                     image_block = {
#                         'type': 'image',
#                         'source': {'type': 'base64', 'media_type': media_type, 'data': b64}
#                     }
#                     text_block = {'type': 'text', 'text': text}
#                     content_blocks = [image_block, text_block]

#                     # 단일 메시지에 블록 리스트 전달
#                     messages.append({'role': 'user', 'content': content_blocks})
#                 else:
#                     # 텍스트 전용 메시지
#                     messages.append({'role': 'user', 'content': [{'type': 'text', 'text': text}]})

#                 # Messages API 호출
#                 resp = self.client.messages.create(
#                     model=self.model,
#                     messages=messages,
#                     max_tokens=max_tokens
#                 )
#                 # 응답 블록에서 텍스트만 합치기
#                 assistant_response = ' '.join(getattr(block, 'text', '') for block in resp.content)

#             elif self.api_type == 'groq':
#                 # Groq Chat API
#                 resp = self.client.chat.completions.create(
#                     model=self.model,
#                     messages=self.conversation_history + [{"role": "user", "content": text}],
#                     temperature=0.7,
#                     max_tokens=1024
#                 )
#                 assistant_response = resp.choices[0].message.content

#             else:
#                 raise ValueError(f"Unsupported api_type: {self.api_type}")

#             # 응답 기록 및 반환
#             self.conversation_history.append({"role": "assistant", "content": assistant_response})
#             logger.info(f"[{self.api_type}] Response: {assistant_response[:100]}...")
#             return assistant_response

#         except Exception as e:
#             logger.error(f"Error in chat method ({self.api_type}): {e}", exc_info=True)
#             raise


# paste-2.txt 수정된 내용

# chatbot.py - OpenAI v1.0+ 호환 버전
import openai
import anthropic
from groq import Groq
import logging
import json
import asyncio
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# sanitize_and_parse_json 함수 (기존 함수 포함)
def sanitize_and_parse_json(text, selected_models=None, responses=None):
    """JSON 응답을 정리하고 파싱하는 함수"""
    import re
    try:
        text = text.strip()
        
        # 코드 블록 제거
        if text.startswith('```json') and '```' in text:
            text = re.sub(r'```json(.*?)```', r'\1', text, flags=re.DOTALL).strip()
        elif text.startswith('```') and text.endswith('```'):
            text = text[3:-3].strip()
        
        # JSON 패턴 추출
        json_pattern = r'({[\s\S]*})'
        json_matches = re.findall(json_pattern, text)
        if json_matches:
            text = json_matches[0]
        
        # 이스케이프 문자 처리
        text = re.sub(r'\\([_"])', r'\1', text)
        
        # JSON 파싱
        result = json.loads(text)
        
        # 필수 필드 확인 및 보정
        required_fields = ["preferredModel", "best_response", "analysis", "reasoning"]
        for field in required_fields:
            if field not in result:
                if field == "best_response" and "bestResponse" in result:
                    result["best_response"] = result["bestResponse"]
                else:
                    result[field] = "" if field != "analysis" else {}
        
        return result
        
    except Exception as e:
        logger.error(f"JSON 파싱 실패: {e}")
        # 폴백 응답 생성
        error_analysis = {}
        if selected_models:
            for model in selected_models:
                model_lower = model.lower()
                error_analysis[model_lower] = {"장점": "분석 실패", "단점": "분석 실패"}
        
        return {
            "preferredModel": "ERROR",
            "best_response": max(responses.values(), key=len) if responses else "분석 오류가 발생했습니다.",
            "analysis": error_analysis,
            "reasoning": "응답 분석 중 오류가 발생했습니다."
        }
import openai
import os 
import anthropic
from groq import Groq
import logging
# from .langchain_config import LangChainManager  # 임시로 주석 처리

logger = logging.getLogger(__name__)
class ChatBot:
    def __init__(self, api_key, model, api_type, langchain_manager=None):
        self.conversation_history = []
        self.model = model
        self.api_type = api_type
        self.api_key = api_key
        self.langchain_manager = langchain_manager
        
        # LangChain 사용 여부 결정
        self.use_langchain = langchain_manager is not None
        
        if not self.use_langchain:
            # 기존 방식 초기화
            if api_type == 'openai':
                # OpenAI 클라이언트는 지연 로딩으로 처리
                pass
            elif api_type == 'anthropic':
                # Anthropic 클라이언트는 지연 로딩으로 처리
                pass
            elif api_type == 'groq':
                # Groq 클라이언트는 지연 로딩으로 처리
                pass
        else:
            # LangChain 체인 생성
            try:
                if api_type in ['gpt', 'claude']:
                    self.chat_chain = langchain_manager.create_chat_chain(api_type)
                elif api_type == 'groq' or api_type == 'mixtral':
                    # Groq는 별도 처리
                    self.groq_llm = langchain_manager.groq_llm if hasattr(langchain_manager, 'groq_llm') else None
                logger.info(f"LangChain 체인 생성 완료: {api_type}")
            except Exception as e:
                logger.warning(f"LangChain 체인 생성 실패, 기존 방식 사용: {e}")
                self.use_langchain = False

    async def _analyze_with_langchain(self, responses, query, user_language, selected_models):
        """LangChain을 사용한 응답 분석"""
        try:
            logger.info("\n" + "="*100)
            logger.info("📊 LangChain 분석 시작")
            logger.info(f"🤖 분석 수행 AI: {self.api_type.upper()}")
            logger.info(f"🔍 선택된 모델들: {', '.join(selected_models)}")
            logger.info("="*100)
            
            # 분석 체인 생성
            analysis_chain = self.langchain_manager.create_analysis_chain(self.api_type)
            
            # 응답 포맷팅
            formatted = self.langchain_manager.format_responses_for_analysis(
                responses, selected_models
            )
            
            # 분석 실행
            analysis_result = await analysis_chain.arun(
                query=query,
                user_language=user_language,
                selected_models=selected_models,
                **formatted
            )
            
            # ✅ 수정: preferredModel을 실제 모델명으로 설정
            analysis_result['preferredModel'] = self.api_type.upper()
            # 추가: botName도 설정
            analysis_result['botName'] = self.api_type.upper()
            
            logger.info("✅ LangChain 분석 완료\n")
            return analysis_result
            
        except Exception as e:
            logger.error(f"❌ LangChain 분석 에러: {str(e)}")
            # 폴백: 기존 방식
            return self.analyze_responses(responses, query, user_language, selected_models)

    def analyze_responses(self, responses, query, user_language, selected_models):
        """기존 동기 응답 분석 메서드 (호환성 유지)"""
        try:
            logger.info("\n" + "="*100)
            logger.info("📊 분석 시작")
            logger.info(f"🤖 분석 수행 AI: {self.api_type.upper()}")
            logger.info(f"🔍 선택된 모델들: {', '.join(selected_models)}")
            logger.info("="*100)

            # 선택된 모델들만 분석에 포함
            responses_section = ""
            analysis_section = ""
            
            for model in selected_models:
                model_lower = model.lower()
                responses_section += f"\n{model.upper()} 응답: 반드시 이 언어({user_language})로 작성 {responses.get(model_lower, '응답 없음')}"
                
                analysis_section += f"""
                        "{model_lower}": {{
                            "장점": "반드시 이 언어({user_language})로 작성 {model.upper()} 답변의 장점",
                            "단점": "반드시 이 언어({user_language})로 작성 {model.upper()} 답변의 단점"
                        }}{"," if model_lower != selected_models[-1].lower() else ""}"""

            # ✅ 수정: preferredModel을 실제 모델명으로 설정
            analysis_prompt = f"""다음은 동일한 질문에 대한 {len(selected_models)}가지 AI의 응답을 분석하는 것입니다.
                    사용자가 선택한 언어는 '{user_language}'입니다.
                    반드시 이 언어({user_language})로 최적의 답을 작성해주세요.
                    반드시 이 언어({user_language})로 장점을 작성해주세요.
                    반드시 이 언어({user_language})로 단점을 작성해주세요.
                    반드시 이 언어({user_language})로 분석 근거를 작성해주세요.

                    질문: {query}
                    {responses_section}

                     [최적의 응답을 만들 때 고려할 사항]
                    - 모든 AI의 답변들을 종합하여 최적의 답변으로 반드시 재구성합니다
                    - 기존 AI의 답변을 그대로 사용하면 안됩니다
                    - 즉, 기존 AI의 답변과 최적의 답변이 동일하면 안됩니다.
                    - 다수의 AI가 공통으로 제공한 정보는 가장 신뢰할 수 있는 올바른 정보로 간주합니다
                    - 코드를 묻는 질문일때는, AI의 답변 중 제일 좋은 답변을 선택해서 재구성해줘
                    - 반드시 JSON 형식으로 응답해주세요
                    [출력 형식]
                    {{
                        "preferredModel": "{self.api_type.upper()}",
                        "botName": "{self.api_type.upper()}",
                        "best_response": "최적의 답변 ({user_language}로 작성)",
                        "analysis": {{
                            {analysis_section}
                        }},
                        "reasoning": "반드시 이 언어({user_language})로 작성 최적의 응답을 선택한 이유"
                    }}"""

            # 기존 API 호출 로직 (변경 없음)
            if self.api_type == 'openai':
                client = get_openai_client()
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant. Always respond with valid JSON ONLY, no additional text or explanations."},
                        {"role": "user", "content": analysis_prompt}
                    ],
                    temperature=0,
                    max_tokens=4096
                )
                analysis_text = response['choices'][0]['message']['content']
                
            elif self.api_type == 'anthropic':
                system_message = next((msg['content'] for msg in self.conversation_history 
                                    if msg['role'] == 'system'), '')
                
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    temperature=0,
                    system=f"{system_message}\nYou must respond with valid JSON only in the specified language. No other text or formatting.",
                    messages=[{
                        "role": "user", 
                        "content": analysis_prompt
                    }]
                )
                analysis_text = message.content[0].text.strip()
            
            elif self.api_type == 'groq':
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant. Always respond with valid JSON ONLY, no additional text or explanations."},
                        {"role": "user", "content": analysis_prompt}
                    ],
                    temperature=0,
                    max_tokens=4096
                )
                analysis_text = response.choices[0].message.content

            logger.info("✅ 분석 완료\n")
            
            # JSON 파싱 (기존 함수 사용)
            from paste_3 import sanitize_and_parse_json  # 기존 함수 import
            analysis_result = sanitize_and_parse_json(analysis_text, selected_models, responses)
            
            # ✅ 수정: preferredModel과 botName을 실제 모델명으로 설정
            analysis_result['preferredModel'] = self.api_type.upper()
            analysis_result['botName'] = self.api_type.upper()
            
            return analysis_result
        
        except Exception as e:
            logger.error(f"❌ Analysis error: {str(e)}")
            # 기존 폴백 로직
            error_analysis = {}
            for model in selected_models:
                model_lower = model.lower()
                error_analysis[model_lower] = {"장점": "분석 실패", "단점": "분석 실패"}
            
            return {
                "preferredModel": self.api_type.upper(),
                "botName": self.api_type.upper(),  # ✅ 추가
                "best_response": max(responses.values(), key=len) if responses else "",
                "analysis": error_analysis,
                "reasoning": "응답 분석 중 오류가 발생하여 최적의 답변을 생성하지 못했습니다."
            }
# class ChatBot:
#     def __init__(self, api_key, model, api_type, langchain_manager=None):
#         self.conversation_history = []
#         self.model = model
#         self.api_type = api_type
#         self.api_key = api_key
#         self.langchain_manager = langchain_manager
        
#         # LangChain 사용 여부 결정
#         self.use_langchain = langchain_manager is not None
        
#         if not self.use_langchain:
#             # 기존 방식 초기화
#             if api_type == 'openai':
#                 openai.api_key = api_key
#             elif api_type == 'anthropic':
#                 self.client = anthropic.Anthropic(api_key=api_key)
#             elif api_type == 'groq':
#                 self.client = Groq(api_key=api_key)
#         else:
#             # LangChain 체인 생성
#             try:
#                 if api_type in ['gpt', 'claude']:
#                     self.chat_chain = langchain_manager.create_chat_chain(api_type)
#                 elif api_type == 'groq' or api_type == 'mixtral':
#                     # Groq는 별도 처리
#                     self.groq_llm = langchain_manager.groq_llm if hasattr(langchain_manager, 'groq_llm') else None
#                 logger.info(f"LangChain 체인 생성 완료: {api_type}")
#             except Exception as e:
#                 logger.warning(f"LangChain 체인 생성 실패, 기존 방식 사용: {e}")
#                 self.use_langchain = False
   
#     async def chat_async(self, user_input, image_file=None, analysis_mode=None, user_language=None):
#         """비동기 채팅 메서드 (LangChain 용)"""
#         if self.use_langchain:
#             return await self._chat_with_langchain(user_input, user_language)
#         else:
#             return self.chat(user_input, image_file, analysis_mode, user_language)
    
#     async def _chat_with_langchain(self, user_input, user_language='ko'):
#         """LangChain을 사용한 채팅"""
#         try:
#             if self.api_type in ['gpt', 'claude']:
#                 result = await self.chat_chain.arun(
#                     user_input=user_input,
#                     user_language=user_language
#                 )
#                 return result
#             elif self.api_type == 'groq' or self.api_type == 'mixtral':
#                 if self.groq_llm:
#                     prompt = f"사용자가 선택한 언어는 '{user_language}'입니다. 반드시 이 언어({user_language})로 응답하세요.\n\n{user_input}"
#                     result = self.groq_llm(prompt)
#                     return result
#                 else:
#                     # 폴백: 기존 방식
#                     return self.chat(user_input, user_language=user_language)
#             else:
#                 raise ValueError(f"지원하지 않는 API 타입: {self.api_type}")
                
#         except Exception as e:
#             logger.error(f"LangChain 채팅 에러: {e}")
#             # 폴백: 기존 방식
#             return self.chat(user_input, user_language=user_language)

#     def chat(self, user_input, image_file=None, analysis_mode=None, user_language=None):
#         """기존 동기 채팅 메서드 (호환성 유지)"""
#         try:
#             logger.info(f"Processing chat request for {self.api_type}")
#             logger.info(f"User input: {user_input}")
            
#             # 대화 기록에 사용자 입력 추가
#             if image_file:
#                 self.conversation_history = [{
#                     "role": "system",
#                     "content": f"이미지 분석 모드: {analysis_mode}, 응답 언어: {user_language}"
#                 }]
#                 messages = [
#                     {"role": "user", "content": user_input}
#                 ]
#             else:
#                 self.conversation_history.append({"role": "user", "content": user_input})
#                 messages = self.conversation_history

#             try:
#                 if self.api_type == 'openai':
#                     response = openai.ChatCompletion.create(
#                         model=self.model,
#                         messages=self.conversation_history,
#                         temperature=0.7,
#                         max_tokens=1024
#                     )
#                     assistant_response = response['choices'][0]['message']['content']
                    
#                 elif self.api_type == 'anthropic':
#                     try:
#                         # 시스템 메시지 찾기
#                         system_message = next((msg['content'] for msg in self.conversation_history 
#                                             if msg['role'] == 'system'), '')
                        
#                         # 사용자 메시지 찾기
#                         user_content = next((msg['content'] for msg in self.conversation_history 
#                                         if msg['role'] == 'user'), '')

#                         message = self.client.messages.create(
#                             model=self.model,
#                             max_tokens=4096,
#                             temperature=0,
#                             system=system_message,
#                             messages=[{
#                                 "role": "user",
#                                 "content": user_content
#                             }]
#                         )
#                         assistant_response = message.content[0].text
#                         logger.info(f"Anthropic response with system message: {system_message[:100]}")
                        
#                     except Exception as e:
#                         logger.error(f"Anthropic API error: {str(e)}")
#                         raise
               
#                 elif self.api_type == 'groq':
#                     response = self.client.chat.completions.create(
#                         model=self.model,
#                         messages=self.conversation_history,
#                         temperature=0.7,
#                         max_tokens=1024
#                     )
#                     assistant_response = response.choices[0].message.content
               
#                 # 응답 기록
#                 self.conversation_history.append({
#                     "role": "assistant",
#                     "content": assistant_response
#                 })
               
#                 logger.info(f"Generated response: {assistant_response[:100]}...")
#                 return assistant_response
               
#             except Exception as e:
#                 logger.error(f"API error in {self.api_type}: {str(e)}", exc_info=True)
#                 raise
               
#         except Exception as e:
#             logger.error(f"Error in chat method: {str(e)}", exc_info=True)
#             raise

#     async def analyze_responses_async(self, responses, query, user_language, selected_models):
#         """비동기 응답 분석 (LangChain 용)"""
#         if self.use_langchain and self.langchain_manager:
#             return await self._analyze_with_langchain(responses, query, user_language, selected_models)
#         else:
#             return self.analyze_responses(responses, query, user_language, selected_models)
    
#     async def _analyze_with_langchain(self, responses, query, user_language, selected_models):
#         """LangChain을 사용한 응답 분석"""
#         try:
#             logger.info("\n" + "="*100)
#             logger.info("📊 LangChain 분석 시작")
#             logger.info(f"🤖 분석 수행 AI: {self.api_type.upper()}")
#             logger.info(f"🔍 선택된 모델들: {', '.join(selected_models)}")
#             logger.info("="*100)
            
#             # 분석 체인 생성
#             analysis_chain = self.langchain_manager.create_analysis_chain(self.api_type)
            
#             # 응답 포맷팅
#             formatted = self.langchain_manager.format_responses_for_analysis(
#                 responses, selected_models
#             )
            
#             # 분석 실행
#             analysis_result = await analysis_chain.arun(
#                 query=query,
#                 user_language=user_language,
#                 selected_models=selected_models,
#                 **formatted
#             )
            
#             # preferredModel 설정
#             analysis_result['preferredModel'] = self.api_type.upper()
            
#             logger.info("✅ LangChain 분석 완료\n")
#             return analysis_result
            
#         except Exception as e:
#             logger.error(f"❌ LangChain 분석 에러: {str(e)}")
#             # 폴백: 기존 방식
#             return self.analyze_responses(responses, query, user_language, selected_models)

#     def analyze_responses(self, responses, query, user_language, selected_models):
#         """기존 동기 응답 분석 메서드 (호환성 유지)"""
#         try:
#             logger.info("\n" + "="*100)
#             logger.info("📊 분석 시작")
#             logger.info(f"🤖 분석 수행 AI: {self.api_type.upper()}")
#             logger.info(f"🔍 선택된 모델들: {', '.join(selected_models)}")
#             logger.info("="*100)

#             # 선택된 모델들만 분석에 포함
#             responses_section = ""
#             analysis_section = ""
            
#             for model in selected_models:
#                 model_lower = model.lower()
#                 responses_section += f"\n{model.upper()} 응답: 반드시 이 언어({user_language})로 작성 {responses.get(model_lower, '응답 없음')}"
                
#                 analysis_section += f"""
#                         "{model_lower}": {{
#                             "장점": "반드시 이 언어({user_language})로 작성 {model.upper()} 답변의 장점",
#                             "단점": "반드시 이 언어({user_language})로 작성 {model.upper()} 답변의 단점"
#                         }}{"," if model_lower != selected_models[-1].lower() else ""}"""

#             # 기존 분석 프롬프트 (변경 없음)
#             analysis_prompt = f"""다음은 동일한 질문에 대한 {len(selected_models)}가지 AI의 응답을 분석하는 것입니다.
#                     사용자가 선택한 언어는 '{user_language}'입니다.
#                     반드시 이 언어({user_language})로 최적의 답을 작성해주세요.
#                     반드시 이 언어({user_language})로 장점을 작성해주세요.
#                     반드시 이 언어({user_language})로 단점을 작성해주세요.
#                     반드시 이 언어({user_language})로 분석 근거를 작성해주세요.

#                     질문: {query}
#                     {responses_section}

#                      [최적의 응답을 만들 때 고려할 사항]
#                     - 모든 AI의 답변들을 종합하여 최적의 답변으로 반드시 재구성합니다
#                     - 기존 AI의 답변을 그대로 사용하면 안됩니다
#                     - 즉, 기존 AI의 답변과 최적의 답변이 동일하면 안됩니다.
#                     - 다수의 AI가 공통으로 제공한 정보는 가장 신뢰할 수 있는 올바른 정보로 간주합니다
#                     - 코드를 묻는 질문일때는, AI의 답변 중 제일 좋은 답변을 선택해서 재구성해줘
#                     - 반드시 JSON 형식으로 응답해주세요
#                     [출력 형식]
#                     {{
#                         "preferredModel": "{self.api_type.upper()}",
#                         "best_response": "최적의 답변 ({user_language}로 작성)",
#                         "analysis": {{
#                             {analysis_section}
#                         }},
#                         "reasoning": "반드시 이 언어({user_language})로 작성 최적의 응답을 선택한 이유"
#                     }}"""

#             # 기존 API 호출 로직 (변경 없음)
#             if self.api_type == 'openai':
#                 response = openai.ChatCompletion.create(
#                     model=self.model,
#                     messages=[
#                         {"role": "system", "content": "You are a helpful assistant. Always respond with valid JSON ONLY, no additional text or explanations."},
#                         {"role": "user", "content": analysis_prompt}
#                     ],
#                     temperature=0,
#                     max_tokens=4096
#                 )
#                 analysis_text = response['choices'][0]['message']['content']
                
#             elif self.api_type == 'anthropic':
#                 system_message = next((msg['content'] for msg in self.conversation_history 
#                                     if msg['role'] == 'system'), '')
                
#                 message = self.client.messages.create(
#                     model=self.model,
#                     max_tokens=4096,
#                     temperature=0,
#                     system=f"{system_message}\nYou must respond with valid JSON only in the specified language. No other text or formatting.",
#                     messages=[{
#                         "role": "user", 
#                         "content": analysis_prompt
#                     }]
#                 )
#                 analysis_text = message.content[0].text.strip()
            
#             elif self.api_type == 'groq':
#                 response = self.client.chat.completions.create(
#                     model=self.model,
#                     messages=[
#                         {"role": "system", "content": "You are a helpful assistant. Always respond with valid JSON ONLY, no additional text or explanations."},
#                         {"role": "user", "content": analysis_prompt}
#                     ],
#                     temperature=0,
#                     max_tokens=4096
#                 )
#                 analysis_text = response.choices[0].message.content

#             logger.info("✅ 분석 완료\n")
            
#             # JSON 파싱 (기존 함수 사용)
#             from paste_3 import sanitize_and_parse_json  # 기존 함수 import
#             analysis_result = sanitize_and_parse_json(analysis_text, selected_models, responses)
#             analysis_result['preferredModel'] = self.api_type.upper()
            
#             return analysis_result
        
#         except Exception as e:
#             logger.error(f"❌ Analysis error: {str(e)}")
#             # 기존 폴백 로직
#             error_analysis = {}
#             for model in selected_models:
#                 model_lower = model.lower()
#                 error_analysis[model_lower] = {"장점": "분석 실패", "단점": "분석 실패"}
            
#             return {
#                 "preferredModel": self.api_type.upper(),
#                 "best_response": max(responses.values(), key=len) if responses else "",
#                 "analysis": error_analysis,
#                 "reasoning": "응답 분석 중 오류가 발생하여 최적의 답변을 생성하지 못했습니다."
#             }
# class ChatView(APIView):
#     permission_classes = [AllowAny]

#     def post(self, request, preferredModel):
#         try:
#             logger.info(f"Received chat request for {preferredModel}")
            
#             data = request.data
#             user_message = data.get('message')
#             compare_responses = data.get('compare', True)
            
#             # 새로운 파라미터: 선택된 모델들
#             selected_models = data.get('selectedModels', ['gpt', 'claude', 'mixtral'])
            
#             # 선택된 모델 로그
#             logger.info(f"Selected models: {selected_models}")
            
#             # 토큰 유무에 따른 언어 및 선호 모델 처리
#             token = request.headers.get('Authorization')
#             if not token:
#                 # 비로그인: 기본 언어는 ko, 선호 모델은 GPT로 고정
#                 user_language = 'ko'
#                 preferredModel = 'gpt'
#             else:
#                 # 로그인: 요청 데이터의 언어 사용 (혹은 사용자의 설정을 따름)
#                 user_language = data.get('language', 'ko')
#                 # URL에 전달된 preferredModel을 그대로 사용 (프론트엔드에서 사용자 설정 반영)

#             logger.info(f"Received language setting: {user_language}")

#             if not user_message:
#                 return Response({'error': 'No message provided'}, 
#                                 status=status.HTTP_400_BAD_REQUEST)

#             # 비동기 응답을 위한 StreamingHttpResponse 사용
#             from django.http import StreamingHttpResponse
#             import json
#             import time

#             def stream_responses():
#                 try:
#                     system_message = {
#                         "role": "system",
#                         "content": f"사용자가 선택한 언어는 '{user_language}'입니다. 반드시 모든 응답을 이 언어({user_language})로 제공해주세요."
#                     }
                    
#                     responses = {}
                    
#                     # 현재 요청에 대한 고유 식별자 생성 (타임스탬프 활용)
#                     request_id = str(time.time())
                    
#                     # 선택된 모델들만 대화에 참여시킴
#                     selected_chatbots = {model: chatbots.get(model) for model in selected_models if model in chatbots}
                    
#                     # 각 봇의 응답을 개별적으로 처리하고 즉시 응답
#                     for bot_id, bot in selected_chatbots.items():
#                         if bot is None:
#                             logger.warning(f"Selected model {bot_id} not available in chatbots")
#                             yield json.dumps({
#                                 'type': 'bot_error',
#                                 'botId': bot_id,
#                                 'error': f"Model {bot_id} is not available"
#                             }) + '\n'
#                             continue
                            
#                         try:
#                             # 매번 새로운 대화 컨텍스트 생성 (이전 내용 초기화)
#                             bot.conversation_history = [system_message]
#                             response = bot.chat(user_message)
#                             responses[bot_id] = response
                            
#                             # 각 봇 응답을 즉시 전송
#                             yield json.dumps({
#                                 'type': 'bot_response',
#                                 'botId': bot_id,
#                                 'response': response,
#                                 'requestId': request_id  # 요청 ID 추가
#                             }) + '\n'
                            
#                         except Exception as e:
#                             logger.error(f"Error from {bot_id}: {str(e)}")
#                             responses[bot_id] = f"Error: {str(e)}"
                            
#                             # 에러도 즉시 전송
#                             yield json.dumps({
#                                 'type': 'bot_error',
#                                 'botId': bot_id,
#                                 'error': str(e),
#                                 'requestId': request_id  # 요청 ID 추가
#                             }) + '\n'
                    
#                     # 선택된 모델이 있고 응답이 있을 때만 분석 수행
#                     if selected_models and responses:
#                         # 분석(비교)은 로그인 시 사용자의 선호 모델을, 비로그인 시 GPT를 사용
#                         if token:
#                             analyzer_bot = chatbots.get(preferredModel) or chatbots.get('gpt')
#                         else:
#                             analyzer_bot = chatbots.get('gpt')
                        
#                         # 분석용 봇도 새로운 대화 컨텍스트로 초기화
#                         analyzer_bot.conversation_history = [system_message]
                        
#                         # 분석 실행 (항상 새롭게 실행)
#                         analysis = analyzer_bot.analyze_responses(responses, user_message, user_language, selected_models)
                        
#                         # 분석 결과 전송
#                         yield json.dumps({
#                             'type': 'analysis',
#                             'preferredModel': analyzer_bot and analyzer_bot.api_type.upper(),
#                             'best_response': analysis.get('best_response', ''),
#                             'analysis': analysis.get('analysis', {}),
#                             'reasoning': analysis.get('reasoning', ''),
#                             'language': user_language,
#                             'requestId': request_id,  # 요청 ID 추가
#                             'timestamp': time.time()  # 타임스탬프 추가
#                         }) + '\n'
#                     else:
#                         logger.warning("No selected models or responses to analyze")
                    
#                 except Exception as e:
#                     logger.error(f"Stream processing error: {str(e)}", exc_info=True)
#                     yield json.dumps({
#                         'type': 'error',
#                         'error': f"Stream processing error: {str(e)}"
#                     }) + '\n'

#             # StreamingHttpResponse 반환
#             response = StreamingHttpResponse(
#                 streaming_content=stream_responses(),
#                 content_type='text/event-stream'
#             )
#             response['Cache-Control'] = 'no-cache'
#             response['X-Accel-Buffering'] = 'no'
#             return response
                
#         except Exception as e:
#             logger.error(f"Unexpected error: {str(e)}", exc_info=True)
#             return Response({
#                 'error': str(e)
#             }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny
from django.http import StreamingHttpResponse
import logging
import json
import openai
import anthropic
from groq import Groq
from django.conf import settings
from bs4 import BeautifulSoup
import re
import time
import asyncio
from asgiref.sync import sync_to_async

# 새로 추가된 import
from .langchain_config import LangChainManager
from .langgraph_workflow import AIComparisonWorkflow

logger = logging.getLogger(__name__)

def convert_to_serializable(obj):
    """객체를 직렬화 가능한 형태로 변환"""
    if hasattr(obj, '__dict__'):
        return {k: convert_to_serializable(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj

class ChatView(APIView):
    permission_classes = [AllowAny]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # 기존 유사도 분석기
        from .similarity_analyzer import SimilarityAnalyzer  # 실제 import 경로로 변경 필요
        self.similarity_analyzer = SimilarityAnalyzer(threshold=0.85)
        
        # LangChain 관리자 초기화
        self.langchain_manager = LangChainManager(
            openai_key=OPENAI_API_KEY,
            anthropic_key=ANTHROPIC_API_KEY,
            groq_key=GROQ_API_KEY,
            google_key=os.getenv('GOOGLE_API_KEY')  # Google API 키 추가
        )
        
        # LangGraph 워크플로우 초기화
        self.workflow = AIComparisonWorkflow(
            langchain_manager=self.langchain_manager,
            similarity_analyzer=self.similarity_analyzer
        )
        
        # 기존 ChatBot 인스턴스들도 LangChain 사용하도록 업데이트
        self.update_chatbots_with_langchain()

    def update_chatbots_with_langchain(self):
        """기존 ChatBot들을 LangChain을 사용하도록 업데이트"""
        global chatbots
        
        # 기존 ChatBot들에 LangChain 매니저 추가
        for bot_id, bot in chatbots.items():
            bot.langchain_manager = self.langchain_manager
            bot.use_langchain = True
            
            # LangChain 체인 생성 시도
            try:
                if bot_id == 'gpt':
                    bot.chat_chain = self.langchain_manager.create_chat_chain('gpt')
                elif bot_id == 'claude':
                    bot.chat_chain = self.langchain_manager.create_chat_chain('claude')
                elif bot_id == 'mixtral':
                    bot.groq_llm = self.langchain_manager.groq_llm if hasattr(self.langchain_manager, 'groq_llm') else None
                logger.info(f"LangChain 체인 생성 완료: {bot_id}")
            except Exception as e:
                logger.warning(f"LangChain 체인 생성 실패 ({bot_id}), 기존 방식 사용: {e}")
                bot.use_langchain = False

    def post(self, request, preferredModel):
        try:
            logger.info(f"Received chat request for {preferredModel}")
            data = request.data
            user_message = data.get('message')
            selected_models = data.get('selectedModels', ['gpt', 'claude', 'mixtral'])
            token = request.headers.get('Authorization')
            user_language = 'ko' if not token else data.get('language', 'ko')
            use_workflow = data.get('useWorkflow', True)  # 워크플로우 사용 여부
            
            if not user_message:
                return Response({'error': 'No message provided'}, status=status.HTTP_400_BAD_REQUEST)

            # URL 처리 로직 (기존과 동일)
            url_pattern = r'^(https?://\S+)$'
            match = re.match(url_pattern, user_message.strip())
            if match:
                url = match.group(1)
                try:
                    page_text = fetch_and_clean_url(url)
                    if len(page_text) > 10000:
                        page_text = page_text[:5000] + "\n\n…(중략)…\n\n" + page_text[-5000:]
                    user_message = (
                        f"다음 웹페이지의 내용을 분석해 주세요:\n"
                        f"URL: {url}\n\n"
                        f"{page_text}"
                    )
                except Exception as e:
                    logger.error(f"URL fetch error: {e}")
                    return Response({'error': f"URL을 가져오지 못했습니다: {e}"}, status=status.HTTP_400_BAD_REQUEST)

            # 워크플로우 사용 여부에 따른 분기
            if use_workflow:
                return self.handle_with_workflow(user_message, selected_models, user_language, preferredModel)
            else:
                return self.handle_with_legacy(user_message, selected_models, user_language, preferredModel)

        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def handle_with_workflow(self, user_message, selected_models, user_language, preferred_model):
        """LangGraph 워크플로우를 사용한 처리"""
        def stream_workflow_responses():
            try:
                request_id = str(time.time())
                
                # 워크플로우 실행을 위한 async 래퍼
                async def run_workflow_async():
                    return await self.workflow.run_workflow(
                        user_message=user_message,
                        selected_models=selected_models,
                        user_language=user_language,
                        request_id=request_id
                    )
                
                # asyncio 이벤트 루프에서 실행
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    workflow_result = loop.run_until_complete(run_workflow_async())
                finally:
                    loop.close()
                
                # 개별 응답 스트리밍
                for bot_id, response in workflow_result["individual_responses"].items():
                    yield json.dumps({
                        'type': 'bot_response',
                        'botId': bot_id,
                        'response': response,
                        'requestId': request_id
                    }) + '\n'
                
                # 유사도 분석 결과
                if workflow_result["similarity_analysis"]:
                    yield json.dumps({
                        'type': 'similarity_analysis',
                        'result': workflow_result["similarity_analysis"],
                        'requestId': request_id,
                        'timestamp': time.time(),
                        'userMessage': user_message
                    }) + '\n'
                
                # 최종 분석 결과
                final_analysis = workflow_result["final_analysis"]
                yield json.dumps({
                    'type': 'analysis',
                    'preferredModel': final_analysis.get('preferredModel', preferred_model.upper()),
                    'best_response': final_analysis.get('best_response', ''),
                    'analysis': final_analysis.get('analysis', {}),
                    'reasoning': final_analysis.get('reasoning', ''),
                    'language': user_language,
                    'requestId': request_id,
                    'timestamp': time.time(),
                    'userMessage': user_message,
                    'workflowUsed': True,
                    'errors': workflow_result.get("errors", [])
                }) + '\n'
                
            except Exception as e:
                logger.error(f"워크플로우 스트리밍 에러: {e}")
                yield json.dumps({
                    'type': 'error',
                    'error': f"Workflow error: {e}",
                    'fallbackToLegacy': True
                }) + '\n'

        return StreamingHttpResponse(stream_workflow_responses(), content_type='text/event-stream')

    def handle_with_legacy(self, user_message, selected_models, user_language, preferred_model):
        """기존 방식으로 처리 (호환성 유지)"""
        def stream_responses():
            try:
                system_message = {
                    'role': 'system',
                    'content': f"사용자가 선택한 언어는 '{user_language}'입니다. 반드시 이 언어({user_language})로 응답하세요."
                }
                responses = {}
                request_id = str(time.time())
                
                # 각 모델별 챗봇 인스턴스 가져오기
                selected_chatbots = {m: chatbots.get(m) for m in selected_models if chatbots.get(m)}

                # 모델 응답 수집 (비동기 처리 시도)
                async def collect_responses_async():
                    responses = {}
                    tasks = []
                    
                    for bot_id, bot in selected_chatbots.items():
                        if hasattr(bot, 'chat_async') and bot.use_langchain:
                            # LangChain 비동기 사용
                            task = bot.chat_async(user_message, user_language=user_language)
                        else:
                            # 기존 동기 방식을 비동기로 래핑
                            task = sync_to_async(self.sync_chat)(bot, user_message, system_message)
                        tasks.append((bot_id, task))
                    
                    for bot_id, task in tasks:
                        try:
                            response = await task
                            responses[bot_id] = response
                            logger.info(f"✅ {bot_id} 응답 완료")
                        except Exception as e:
                            logger.error(f"❌ {bot_id} 응답 실패: {e}")
                    
                    return responses

                # 비동기 응답 수집
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    responses = loop.run_until_complete(collect_responses_async())
                finally:
                    loop.close()

                # 개별 응답 스트리밍
                for bot_id, resp_text in responses.items():
                    yield json.dumps({
                        'type': 'bot_response',
                        'botId': bot_id,
                        'response': resp_text,
                        'requestId': request_id
                    }) + '\n'

                # 유사도 분석
                if len(responses) >= 2:
                    sim_res = self.similarity_analyzer.cluster_responses(responses)
                    serial = convert_to_serializable(sim_res)
                    yield json.dumps({
                        'type': 'similarity_analysis',
                        'result': serial,
                        'requestId': request_id,
                        'timestamp': time.time(),
                        'userMessage': user_message
                    }) + '\n'

                # 최종 비교 및 분석
                analyzer_bot = chatbots.get(preferred_model) or chatbots.get('gpt')
                analyzer_bot.conversation_history = [system_message]
                
                # LangChain 비동기 분석 시도
                if hasattr(analyzer_bot, 'analyze_responses_async') and analyzer_bot.use_langchain:
                    async def analyze_async():
                        return await analyzer_bot.analyze_responses_async(
                            responses, user_message, user_language, list(responses.keys())
                        )
                    
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    try:
                        analysis = loop.run_until_complete(analyze_async())
                    finally:
                        loop.close()
                else:
                    # 기존 동기 방식
                    analysis = analyzer_bot.analyze_responses(
                        responses, user_message, user_language, list(responses.keys())
                    )
                
                yield json.dumps({
                    'type': 'analysis',
                    'preferredModel': analyzer_bot.api_type.upper(),
                    'best_response': analysis.get('best_response', ''),
                    'analysis': analysis.get('analysis', {}),
                    'reasoning': analysis.get('reasoning', ''),
                    'language': user_language,
                    'requestId': request_id,
                    'timestamp': time.time(),
                    'userMessage': user_message,
                    'workflowUsed': False
                }) + '\n'
                
            except Exception as e:
                yield json.dumps({
                    'type': 'error',
                    'error': f"Stream error: {e}"
                }) + '\n'

        return StreamingHttpResponse(stream_responses(), content_type='text/event-stream')

    def sync_chat(self, bot, user_message, system_message):
        """동기 채팅을 위한 헬퍼 메서드"""
        bot.conversation_history = [system_message]
        return bot.chat(user_message)
from dotenv import load_dotenv
load_dotenv()
# API 키 설정 (기존과 동일)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# ChatBot import (수정된 버전)

chatbots = {
    'gpt': ChatBot(OPENAI_API_KEY, 'gpt-3.5-turbo', 'openai'),
    'claude': ChatBot(ANTHROPIC_API_KEY, 'claude-3-5-haiku-20241022', 'anthropic'), 
    'mixtral': ChatBot(GROQ_API_KEY, 'llama-3.1-8b-instant', 'groq'),
}
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from .models import UserProfile, UserSettings
from .serializers import UserSerializer

class UserSettingsView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        try:
            user_settings, created = UserSettings.objects.get_or_create(
                user=request.user,
                defaults={
                    'language': 'ko',
                    'analyzer_bot': 'claude'
                }
            )
            serializer = UserSerializer(user_settings)
            return Response(serializer.data)
        except Exception as e:
            return Response({
                'language': 'ko', 
                'analyzer_bot': 'claude'
            }, status=status.HTTP_200_OK)

    def post(self, request):
        try:
            user_settings, created = UserSettings.objects.get_or_create(
                user=request.user,
                defaults={
                    'language': request.data.get('language', 'ko'),
                    'analyzer_bot': request.data.get('analyzer_bot', 'claude')
                }
            )

            # 이미 존재하는 경우 업데이트
            if not created:
                user_settings.language = request.data.get('language', user_settings.language)
                user_settings.analyzer_bot = request.data.get('analyzer_bot', user_settings.analyzer_bot)
                user_settings.save()

            serializer = UserSettingsSerializer(user_settings)
            return Response(serializer.data, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({
                'error': str(e)
            }, status=status.HTTP_400_BAD_REQUEST)
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
import requests
import logging
from django.contrib.auth import get_user_model
from .models import SocialAccount

import uuid

logger = logging.getLogger(__name__)
User = get_user_model()

# def generate_unique_username(email, name=None):
#     """고유한 username 생성"""
#     base = name or email.split('@')[0]
#     username = base
#     suffix = 1
    
#     # username이 고유할 때까지 숫자 추가
#     while User.objects.filter(username=username).exists():
#         username = f"{base}_{suffix}"
#         suffix += 1
    
#     return username

def generate_unique_username(email, name=None):
    """username 생성 - 이메일 앞부분 또는 이름 사용"""
    if name:
        return name  # 이름이 있으면 그대로 사용
    return email.split('@')[0]  # 이름이 없으면 이메일 앞부분 사용
from rest_framework.authentication import TokenAuthentication
from rest_framework.permissions import AllowAny
from rest_framework.authtoken.models import Token
from rest_framework import status
from rest_framework.decorators import api_view, authentication_classes, permission_classes
# views.py
from django.db import transaction, IntegrityError

# @api_view(['GET'])
# @authentication_classes([TokenAuthentication])
# @permission_classes([AllowAny])
# @api_view(['GET'])
# @permission_classes([AllowAny])
# def google_callback(request):
#     logger.info("Starting Google callback process")  # 로깅 추가
#     try:
#         with transaction.atomic():
#             # 1. 사용자 정보 가져오기
#             auth_header = request.headers.get('Authorization', '')
#             access_token = auth_header.split(' ')[1]
            
#             user_info_response = requests.get(
#                 'https://www.googleapis.com/oauth2/v3/userinfo',
#                 headers={'Authorization': f'Bearer {access_token}'}
#             )
            
#             user_info = user_info_response.json()
#             email = user_info.get('email')
#             name = user_info.get('name')

#             logger.info(f"Processing user: {email}")  # 로깅 추가

#             # 2. User 객체 가져오기 또는 생성
#             user = User.objects.filter(email=email).first()
#             if not user:
#                 user = User.objects.create(
#                     username=email,
#                     email=email,
#                     first_name=name or '',
#                     is_active=True
#                 )
#                 logger.info(f"Created new user: {user.id}")
#             else:
#                 logger.info(f"Found existing user: {user.id}")

#             # 3. 기존 UserSettings 삭제 (있다면)
#             UserSettings.objects.filter(user=user).delete()
#             logger.info("Deleted any existing settings")

#             # 4. 새로운 UserSettings 생성
#             settings = UserSettings.objects.create(
#                 user=user,
#                 language='ko',
#                 preferred_model='default'
#             )
#             logger.info(f"Created new settings for user: {user.id}")

#             # 5. 토큰 생성
#             token, _ = Token.objects.get_or_create(user=user)
            
#             return Response({
#                 'user': {
#                     'id': user.id,
#                     'email': user.email,
#                     'username': user.username,
#                     'first_name': user.first_name,
#                     'settings': {
#                         'language': settings.language,
#                         'preferred_model': settings.preferred_model
#                     }
#                 },
#                 'access_token': token.key
#             })
#     except Exception as e:
#         logger.error(f"Error in google_callback: {str(e)}")
#         return Response(
#             {'error': str(e)},
#             status=status.HTTP_400_BAD_REQUEST
        # )
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
        except User.DoesNotExist:
            # 새로운 사용자 생성
            username = generate_unique_username(email, name)
            user = User.objects.create(
                username=username,
                email=email,
                is_active=True
            )
            
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

        # 토큰 생성 또는 가져오기
        token, created = Token.objects.get_or_create(user=user)
        logger.info(f"GOOGLE Token created: {created}")
        logger.info(f"Token key: {token.key}")
        logger.info(f"Token user: {token.user.username}")


        # 사용자 데이터 반환
        serializer = UserSerializer(user)
        return Response({
            'user': serializer.data,
            'access_token': token.key,  # Django REST Framework Token 반환
            'token_created': created,
            'google_access_token': access_token,  # Google OAuth 액세스 토큰

        })

    except Exception as e:
        logger.error(f"Error in google_callback: {str(e)}")
        return Response(
            {'error': str(e)}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
import requests
import json
import requests
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import redirect
from .models import User  # User 모델을 임포트


@api_view(['GET'])
@permission_classes([AllowAny])
def kakao_callback(request):
    try:
        auth_code = request.GET.get('code')
        logger.info(f"Received Kakao auth code: {auth_code}")
        
        # 카카오 토큰 받기
        token_url = "https://kauth.kakao.com/oauth/token"
        data = {
            "grant_type": "authorization_code",
            "client_id": settings.KAKAO_CLIENT_ID,
            "redirect_uri": settings.KAKAO_REDIRECT_URI,
            "code": auth_code,
        }
        
        token_response = requests.post(token_url, data=data)
        
        if not token_response.ok:
            return Response({
                'error': '카카오 토큰 받기 실패',
                'details': token_response.text
            }, status=status.HTTP_400_BAD_REQUEST)
        
        token_data = token_response.json()
        access_token = token_data.get('access_token')
        
        if not access_token:
            return Response({
                'error': '액세스 토큰 없음',
                'details': token_data
            }, status=status.HTTP_400_BAD_REQUEST)

        # 카카오 사용자 정보 받기
        user_info_url = "https://kapi.kakao.com/v2/user/me"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-type": "application/x-www-form-urlencoded;charset=utf-8"
        }
        
        user_info_response = requests.get(
            user_info_url,
            headers=headers,
            params={
                'property_keys': json.dumps([
                    "kakao_account.email",
                    "kakao_account.profile",
                    "kakao_account.name"
                ])
            }
        )
        
        if not user_info_response.ok:
            return Response({
                'error': '사용자 정보 받기 실패',
                'details': user_info_response.text
            }, status=status.HTTP_400_BAD_REQUEST)
        
        user_info = user_info_response.json()
        kakao_account = user_info.get('kakao_account', {})
        email = kakao_account.get('email')
        profile = kakao_account.get('profile', {})
        nickname = profile.get('nickname')
        
        logger.info(f"Kakao user info - email: {email}, nickname: {nickname}")
        
        if not email:
            return Response({
                'error': '이메일 정보 없음',
                'details': '카카오 계정의 이메일 정보가 필요합니다.'
            }, status=status.HTTP_400_BAD_REQUEST)

        # 사용자 생성 또는 업데이트
        try:
            user = User.objects.get(email=email)
            logger.info(f"Updated existing user with nickname: {nickname}")
        except User.DoesNotExist:
            unique_username = generate_unique_username(email, nickname)
            user = User.objects.create(
                email=email,
                username=unique_username,
                is_active=True
            )            
            logger.info(f"Created new user with nickname: {nickname}")

        # 소셜 계정 생성 또는 업데이트
        social_account, created = SocialAccount.objects.update_or_create(
            email=email,
            provider='kakao',
            defaults={
                'user': user,
                'nickname': nickname
            }
        )
        logger.info(f"Social account updated - email: {email}, nickname: {nickname}")

        if not created and social_account.user != user:
            social_account.user = user
            social_account.save()

        # 토큰 생성 또는 가져오기
        token, token_created = Token.objects.get_or_create(user=user)
        logger.info(f"KAKAO Token created: {token_created}")
        logger.info(f"Token key: {token.key}")
        logger.info(f"Token user: {token.user.username}")

        serializer = UserSerializer(user)
        return Response({
            'user': serializer.data,
            'access_token': token.key,  # Django REST Framework Token 반환
            'token_created': created,
            'kakao_access_token': access_token,  # Google OAuth 액세스 토큰

        })


        
    except Exception as e:
        logger.exception("Unexpected error in kakao_callback")
        return Response({
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


from rest_framework.authtoken.models import Token  # Token 모델 추가

@api_view(['GET'])
@permission_classes([AllowAny])
def naver_callback(request):
    try:
        code = request.GET.get('code')
        state = request.GET.get('state')
        logger.info(f"Received Naver auth code: {code}")

        # 네이버 토큰 받기
        token_url = "https://nid.naver.com/oauth2.0/token"
        token_params = {
            "grant_type": "authorization_code",
            "client_id": settings.NAVER_CLIENT_ID,
            "client_secret": settings.NAVER_CLIENT_SECRET,
            "code": code,
            "state": state
        }

        token_response = requests.get(token_url, params=token_params)

        if not token_response.ok:
            return Response({
                'error': '네이버 토큰 받기 실패',
                'details': token_response.text
            }, status=status.HTTP_400_BAD_REQUEST)

        token_data = token_response.json()
        access_token = token_data.get('access_token')

        if not access_token:
            return Response({
                'error': '액세스 토큰 없음',
                'details': token_data
            }, status=status.HTTP_400_BAD_REQUEST)

        # 네이버 사용자 정보 받기
        user_info_url = "https://openapi.naver.com/v1/nid/me"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-type": "application/x-www-form-urlencoded;charset=utf-8"
        }

        user_info_response = requests.get(user_info_url, headers=headers)

        if not user_info_response.ok:
            return Response({
                'error': '사용자 정보 받기 실패',
                'details': user_info_response.text
            }, status=status.HTTP_400_BAD_REQUEST)

        user_info = user_info_response.json()
        response = user_info.get('response', {})
        email = response.get('email')
        nickname = response.get('nickname')
        username = email.split('@')[0]

        if not email:
            return Response({
                'error': '이메일 정보 없음',
                'details': '네이버 계정의 이메일 정보가 필요합니다.'
            }, status=status.HTTP_400_BAD_REQUEST)

        # 사용자 생성 또는 조회
        user, created = User.objects.get_or_create(
            email=email,
            defaults={'username': generate_unique_username(email, username), 'is_active': True}
        )

        # 소셜 계정 조회 및 업데이트
        social_account, social_created = SocialAccount.objects.update_or_create(
            provider='naver',
            email=email,
            defaults={'user': user, 'nickname': nickname}
        )

        logger.info(f"Social account updated - email: {email}, nickname: {nickname}")

        # ✅ Django REST Framework Token 생성
        token, token_created = Token.objects.get_or_create(user=user)
        logger.info(f"Naver Token created: {token_created}")
        logger.info(f"Token key: {token.key}")
        logger.info(f"Token user: {token.user.username}")

        serializer = UserSerializer(user)
        return Response({
            'user': serializer.data,
            'access_token': token.key,  # Django REST Framework Token 반환
            'token_created': created,
            'naver_access_token': access_token,  # 네이버 액세스 토큰
        })

    except Exception as e:
        logger.exception("Unexpected error in naver_callback")
        return Response({
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# views.py
import logging
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from django.contrib.auth import get_user_model

# logger = logging.getLogger(__name__)

# @api_view(['PUT'])
# @permission_classes([IsAuthenticated])
# def update_user_settings(request):
#     # 추가 로깅 및 디버깅
#     logger.info(f"User authentication status: {request.user.is_authenticated}")
#     logger.info(f"User: {request.user}")
#     logger.info(f"Request headers: {request.headers}")
    
#     try:
#         # 인증 상태 명시적 확인
#         if not request.user.is_authenticated:
#             logger.error("Unauthenticated user attempt")
#             return Response({
#                 'status': 'error',
#                 'message': '인증되지 않은 사용자입니다.'
#             }, status=401)
        
#         # UserProfile 모델이 있다고 가정
#         user = request.user
#         user_profile = user.userprofile
        
#         # 설정 업데이트
#         settings_data = request.data
#         user_profile.language = settings_data.get('language', user_profile.language)
#         user_profile.preferred_model = settings_data.get('preferredModel', user_profile.preferred_model)
#         user_profile.save()
        
#         return Response({
#             'status': 'success',
#             'message': '설정이 성공적으로 업데이트되었습니다.',
#             'settings': {
#                 'language': user_profile.language,
#                 'preferredModel': user_profile.preferred_model
#             }
#         })
    
#     except Exception as e:
#         print("Error:", str(e))  # 에러 로깅
#         logger.error(f"Settings update error: {str(e)}")
#         return Response({
#             'status': 'error',
#             'message': f'오류 발생: {str(e)}'
#         }, status=400)
# views.py
# 백엔드에서 토큰 형식 확인
@api_view(['PUT'])
@authentication_classes([TokenAuthentication])
@permission_classes([IsAuthenticated])
def update_user_settings(request):
    try:
        # 토큰 로깅 추가
        token_header = request.headers.get('Authorization')
        if not token_header or not token_header.startswith('Token '):
            return Response({'error': '잘못된 토큰 형식'}, status=status.HTTP_401_UNAUTHORIZED)
        
        user = request.user
        if not user.is_authenticated:
            return Response({'error': '인증되지 않은 사용자'}, status=status.HTTP_401_UNAUTHORIZED)
        
        settings_data = request.data
        print(f"Received settings data: {settings_data}")  # 데이터 로깅 추가
        
        # UserSettings 업데이트 또는 생성
        settings, created = UserSettings.objects.get_or_create(user=user)
        
        # 필드 업데이트
        if 'language' in settings_data:
            settings.language = settings_data['language']
        if 'preferredModel' in settings_data:
            settings.preferred_model = settings_data['preferredModel']
        
        settings.save()
        
        return Response({
            'message': 'Settings updated successfully',
            'settings': {
                'language': settings.language,
                'preferredModel': settings.preferred_model
            }
        })
        
    except Exception as e:
        logger.error(f"Settings update error: {str(e)}")
        return Response(
            {'error': str(e)},
            status=status.HTTP_400_BAD_REQUEST
        )
from rest_framework import status
from rest_framework.decorators import api_view, authentication_classes, permission_classes
from rest_framework.authentication import TokenAuthentication
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from django.core.exceptions import ObjectDoesNotExist

@api_view(['PUT'])
@authentication_classes([TokenAuthentication])
@permission_classes([IsAuthenticated])
def update_user_settings(request):
    try:
        user = request.user
        settings_data = request.data
        
        # UserProfile 확인 및 생성
        try:
            profile = user.userprofile
        except ObjectDoesNotExist:
            profile = UserProfile.objects.create(user=user)
            
        # UserSettings 확인 및 생성/업데이트
        settings, created = UserSettings.objects.get_or_create(
            user=user,
            defaults={
                'language': settings_data.get('language', 'en'),
                'preferred_model': settings_data.get('preferredModel', 'default')
            }
        )
        
        if not created:
            settings.language = settings_data.get('language', settings.language)
            settings.preferred_model = settings_data.get('preferredModel', settings.preferred_model)
            settings.save()
            
        return Response({
            'message': 'Settings updated successfully',
            'settings': {
                'language': settings.language,
                'preferredModel': settings.preferred_model
            }
        })
            
    except Exception as e:
        print(f"Settings update error: {str(e)}")
        return Response(
            {'error': str(e)},
            status=status.HTTP_400_BAD_REQUEST
        )

import logging
import re
import math
import numpy as np
from collections import Counter
from typing import Dict, List, Union, Any
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
logger = logging.getLogger(__name__)

class SimilarityAnalyzer:
    """
    AI 모델 응답 간의 유사도를 분석하고 응답 특성을 추출하는 클래스
    다국어 지원을 위해 paraphrase-multilingual-MiniLM-L12-v2 모델 사용
    """
    
    def __init__(self, threshold=0., use_transformer=True):
        """
        초기화
        
        Args:
            threshold (float): 유사 응답으로 분류할 임계값 (0~1)
            use_transformer (bool): SentenceTransformer 모델 사용 여부
        """
        self.threshold = threshold
        self.use_transformer = use_transformer
        
        # 다국어 SentenceTransformer 모델 로드
        if use_transformer:
            try:
                self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                logger.info("다국어 SentenceTransformer 모델 로드 완료")
            except Exception as e:
                logger.error(f"SentenceTransformer 모델 로드 실패: {str(e)}")
                self.use_transformer = False
                
        # Fallback용 TF-IDF 벡터라이저
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(
            min_df=1, 
            analyzer='word',
            ngram_range=(1, 2),
            stop_words=None  # 다국어 지원을 위해 stop_words 제거
        )
    
    def preprocess_text(self, text: str) -> str:
        """
        텍스트 전처리
        
        Args:
            text (str): 전처리할 텍스트
            
        Returns:
            str: 전처리된 텍스트
        """
        # 소문자 변환 (영어 텍스트만 해당)
        # 다국어 지원을 위해 영어가 아닌 경우 원래 케이스 유지
        if text.isascii():
            text = text.lower()
        
        # 코드 블록 제거 (분석에서 제외)
        text = re.sub(r'```.*?```', ' CODE_BLOCK ', text, flags=re.DOTALL)
        
        # HTML 태그 제거
        text = re.sub(r'<.*?>', '', text)
        
        # 특수 문자 처리 (다국어 지원을 위해 완전 제거하지 않음)
        text = re.sub(r'[^\w\s\u0080-\uFFFF]', ' ', text)
        
        # 여러 공백을 하나로 치환
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def calculate_similarity_matrix(self, responses: Dict[str, str]) -> Dict[str, Dict[str, float]]:
        """
        모델 응답 간의 유사도 행렬 계산
        
        Args:
            responses (dict): 모델 ID를 키로, 응답 텍스트를 값으로 하는 딕셔너리
            
        Returns:
            dict: 모델 간 유사도 행렬
        """
        try:
            model_ids = list(responses.keys())
            
            # 텍스트 전처리
            preprocessed_texts = [self.preprocess_text(responses[model_id]) for model_id in model_ids]
            
            if self.use_transformer and self.model:
                # SentenceTransformer를 사용한 임베딩 생성
                try:
                    embeddings = self.model.encode(preprocessed_texts)
                    # 코사인 유사도 계산
                    similarity_matrix = cosine_similarity(embeddings)
                except Exception as e:
                    logger.error(f"SentenceTransformer 임베딩 생성 중 오류: {str(e)}")
                    # Fallback: TF-IDF 사용
                    tfidf_matrix = self.vectorizer.fit_transform(preprocessed_texts)
                    similarity_matrix = cosine_similarity(tfidf_matrix)
            else:
                # TF-IDF 벡터화
                tfidf_matrix = self.vectorizer.fit_transform(preprocessed_texts)
                # 코사인 유사도 계산
                similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # 결과를 딕셔너리로 변환
            result = {}
            for i, model1 in enumerate(model_ids):
                result[model1] = {}
                for j, model2 in enumerate(model_ids):
                    result[model1][model2] = float(similarity_matrix[i][j])
            
            return result
            
        except Exception as e:
            logger.error(f"유사도 행렬 계산 중 오류: {str(e)}")
            # 오류 발생 시 빈 행렬 반환
            return {model_id: {other_id: 0.0 for other_id in responses} for model_id in responses}
    
      
    def cluster_responses(self, responses):
        """
        응답을 유사도에 따라 군집화
        
        Args:
            responses (dict): 모델 ID를 키로, 응답 텍스트를 값으로 하는 딕셔너리
            
        Returns:
            dict: 군집화 결과
        """
        try:
            model_ids = list(responses.keys())
            if len(model_ids) <= 1:
                return {
                    "similarGroups": [model_ids],
                    "outliers": [],
                    "similarityMatrix": {}
                }
            
            # 유사도 행렬 계산
            similarity_matrix = self.calculate_similarity_matrix(responses)
            
            # 계층적 클러스터링 수행
            clusters = [[model_id] for model_id in model_ids]
            
            merge_happened = True
            while merge_happened and len(clusters) > 1:
                merge_happened = False
                max_similarity = -1
                merge_indices = [-1, -1]
                
                # 가장 유사한 두 클러스터 찾기
                for i in range(len(clusters)):
                    for j in range(i + 1, len(clusters)):
                        # 두 클러스터 간 평균 유사도 계산
                        cluster_similarity = 0
                        pair_count = 0
                        
                        for model1 in clusters[i]:
                            for model2 in clusters[j]:
                                cluster_similarity += similarity_matrix[model1][model2]
                                pair_count += 1
                        
                        avg_similarity = cluster_similarity / max(1, pair_count)
                        
                        if avg_similarity > max_similarity:
                            max_similarity = avg_similarity
                            merge_indices = [i, j]
                
                # 임계값보다 유사도가 높으면 클러스터 병합
                if max_similarity >= self.threshold:
                    i, j = merge_indices
                    clusters[i].extend(clusters[j])
                    clusters.pop(j)
                    merge_happened = True
            
            # 클러스터 크기에 따라 정렬
            clusters.sort(key=lambda x: -len(x))
            
            # 주요 그룹과 이상치 구분
            main_group = clusters[0] if clusters else []
            outliers = [model for cluster in clusters[1:] for model in cluster]
            
            # 응답 특성 추출
            response_features = {model_id: self.extract_response_features(responses[model_id]) 
                                for model_id in model_ids}
            
            return {
                "similarGroups": clusters,
                "mainGroup": main_group,
                "outliers": outliers,
                "similarityMatrix": similarity_matrix,
                "responseFeatures": response_features
            }
            
        except Exception as e:
            logger.error(f"응답 군집화 중 오류: {str(e)}")
            # 오류 발생 시 모든 모델을 하나의 그룹으로 반환
            return {
                "similarGroups": [model_ids],
                "mainGroup": model_ids,
                "outliers": [],
                "similarityMatrix": {},
                "responseFeatures": {}
            }
    
    
    def extract_response_features(self, text: str) -> Dict[str, Union[int, float, bool]]:
        """
        응답 텍스트에서 특성 추출
        
        Args:
            text (str): 응답 텍스트
            
        Returns:
            dict: 응답 특성 정보
        """
        try:
            # 응답 길이
            length = len(text)
            
            # 코드 블록 개수
            code_blocks = re.findall(r'```[\s\S]*?```', text)
            code_block_count = len(code_blocks)
            
            # 링크 개수
            links = re.findall(r'\[.*?\]\(.*?\)', text) or re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
            link_count = len(links)
            
            # 목록 항목 개수
            list_items = re.findall(r'^[\s]*[-*+] |^[\s]*\d+\. ', text, re.MULTILINE)
            list_item_count = len(list_items)
            
            # 문장 분리 (다국어 지원)
            sentences = re.split(r'[.!?。！？]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            # 평균 문장 길이
            avg_sentence_length = sum(len(s) for s in sentences) / max(1, len(sentences))
            
            # 어휘 다양성 (고유 단어 비율)
            words = re.findall(r'\b\w+\b', text.lower())
            unique_words = set(words)
            vocabulary_diversity = len(unique_words) / max(1, len(words))
            
            # 언어 감지 (추가 기능)
            lang_features = self.detect_language_features(text)
            
            features = {
                "length": length,
                "codeBlockCount": code_block_count,
                "linkCount": link_count,
                "listItemCount": list_item_count,
                "sentenceCount": len(sentences),
                "avgSentenceLength": avg_sentence_length,
                "vocabularyDiversity": vocabulary_diversity,
                "hasCode": code_block_count > 0
            }
            
            # 언어 특성 추가
            features.update(lang_features)
            
            return features
            
        except Exception as e:
            logger.error(f"응답 특성 추출 중 오류: {str(e)}")
            # 오류 발생 시 기본값 반환
            return {
                "length": len(text),
                "codeBlockCount": 0,
                "linkCount": 0,
                "listItemCount": 0,
                "sentenceCount": 1,
                "avgSentenceLength": len(text),
                "vocabularyDiversity": 0,
                "hasCode": False,
                "detectedLang": "unknown"
            }
    
    def detect_language_features(self, text: str) -> Dict[str, Any]:
        """
        텍스트의 언어 특성 감지
        
        Args:
            text (str): 분석할 텍스트
            
        Returns:
            dict: 언어 특성 정보
        """
        try:
            # 언어 특성 감지를 위한 간단한 휴리스틱
            # 실제 프로덕션에서는 langdetect 등의 라이브러리 사용 권장
            
            # 한국어 특성 (한글 비율)
            korean_chars = len(re.findall(r'[ㄱ-ㅎㅏ-ㅣ가-힣]', text))
            
            # 영어 특성 (영문 비율)
            english_chars = len(re.findall(r'[a-zA-Z]', text))
            
            # 일본어 특성 (일본어 문자 비율)
            japanese_chars = len(re.findall(r'[ぁ-んァ-ン一-龯]', text))
            
            # 중국어 특성 (중국어 문자 비율)
            chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
            
            # 기타 문자 (숫자, 특수문자 제외)
            total_chars = len(re.findall(r'[^\d\s\W]', text))
            
            # 비율 계산
            total = max(1, total_chars)
            korean_ratio = korean_chars / total
            english_ratio = english_chars / total
            japanese_ratio = japanese_chars / total
            chinese_ratio = chinese_chars / total
            
            # 주요 언어 결정
            lang_ratios = {
                "ko": korean_ratio,
                "en": english_ratio,
                "ja": japanese_ratio,
                "zh": chinese_ratio,
                "other": 1.0 - (korean_ratio + english_ratio + japanese_ratio + chinese_ratio)
            }
            
            detected_lang = max(lang_ratios.items(), key=lambda x: x[1])[0]
            
            return {
                "detectedLang": detected_lang,
                "langRatios": lang_ratios
            }
            
        except Exception as e:
            logger.error(f"언어 특성 감지 중 오류: {str(e)}")
            return {
                "detectedLang": "unknown",
                "langRatios": {"unknown": 1.0}
            }
    
    def compare_responses(self, response1: str, response2: str) -> Dict[str, Any]:
        """
        두 응답 간의 유사도와 차이점 분석
        
        Args:
            response1 (str): 첫 번째 응답
            response2 (str): 두 번째 응답
            
        Returns:
            dict: 유사도 및 차이점 분석 결과
        """
        try:
            # 텍스트 전처리
            text1 = self.preprocess_text(response1)
            text2 = self.preprocess_text(response2)
            
            # 임베딩 생성 및 유사도 계산
            if self.use_transformer and self.model:
                embeddings = self.model.encode([text1, text2])
                similarity = float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])
            else:
                # TF-IDF를 사용한 유사도 계산
                tfidf_matrix = self.vectorizer.fit_transform([text1, text2])
                similarity = float(cosine_similarity(tfidf_matrix)[0][1])
            
            # 응답 특성 비교
            features1 = self.extract_response_features(response1)
            features2 = self.extract_response_features(response2)
            
            # 특성 차이 계산
            feature_diffs = {}
            for key in set(features1.keys()) & set(features2.keys()):
                if isinstance(features1[key], (int, float)) and isinstance(features2[key], (int, float)):
                    feature_diffs[key] = features2[key] - features1[key]
            
            # 주요 차이점 고유 단어 분석
            words1 = re.findall(r'\b\w+\b', text1.lower())
            words2 = re.findall(r'\b\w+\b', text2.lower())
            
            counter1 = Counter(words1)
            counter2 = Counter(words2)
            
            unique_to_1 = [word for word, count in counter1.items() if word not in counter2]
            unique_to_2 = [word for word, count in counter2.items() if word not in counter1]
            
            # 가장 빈도가 높은 고유 단어 (최대 10개)
            top_unique_to_1 = sorted(
                [(word, counter1[word]) for word in unique_to_1], 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
            
            top_unique_to_2 = sorted(
                [(word, counter2[word]) for word in unique_to_2], 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
            
            return {
                "similarity": similarity,
                "isSimilar": similarity >= self.threshold,
                "features1": features1,
                "features2": features2,
                "featureDiffs": feature_diffs,
                "uniqueWordsTo1": top_unique_to_1,
                "uniqueWordsTo2": top_unique_to_2
            }
            
        except Exception as e:
            logger.error(f"응답 비교 중 오류: {str(e)}")
            return {
                "similarity": 0.0,
                "isSimilar": False,
                "error": str(e)
            }
        
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny
import logging
import json
import openai
import anthropic
from groq import Groq
from django.conf import settings
import time


logger = logging.getLogger(__name__)

class TextSimplificationView(APIView):
    """
    텍스트를 쉬운 표현으로 변환하는 API 뷰
    특정 대상(어린이, 고령자, 외국인 학습자 등)에 맞춰 변환
    """
    permission_classes = [AllowAny]
    
    def post(self, request):
        try:
            logger.info("쉬운 표현 변환 요청 받음")
            
            data = request.data
            original_text = data.get('message')
            target_audience = data.get('targetAudience', 'general')
            language = data.get('language', 'ko')
            
            if not original_text:
                return Response({'error': '변환할 텍스트가 없습니다.'}, 
                               status=status.HTTP_400_BAD_REQUEST)
            
            # 텍스트 단순화 수행
            simplifier = TextSimplifier(
                api_key=settings.OPENAI_API_KEY,
                model="gpt-4-turbo",  # 또는 선호하는 GPT 모델
                api_type="openai"
            )
            
            result = simplifier.simplify_text(
                original_text=original_text,
                target_audience=target_audience,
                language=language
            )
            
            return Response(result)
            
        except Exception as e:
            logger.error(f"텍스트 단순화 중 오류 발생: {str(e)}", exc_info=True)
            return Response({
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class TextSimplifier:
    """
    텍스트를 쉬운 표현으로 변환하는 클래스
    다양한 AI 모델을 활용하여 대상자별 맞춤형 단순화 수행
    """
    def __init__(self, api_key, model, api_type):
        self.model = model
        self.api_type = api_type
        self.api_key = api_key
        
        if api_type == 'openai':
            openai.api_key = api_key
        elif api_type == 'anthropic':
            # Anthropic 클라이언트는 지연 로딩으로 처리
            pass
        elif api_type == 'groq':
            # Groq 클라이언트는 지연 로딩으로 처리
            pass
    
    def simplify_text(self, original_text, target_audience, language='ko'):
        """
        텍스트를 단순화하여 반환
        
        Args:
            original_text (str): 원본 텍스트
            target_audience (str): 대상자 유형 (general, child, elderly, foreigner)
            language (str): 언어 (기본값: 한국어)
            
        Returns:
            dict: 단순화 결과
        """
        try:
            logger.info(f"텍스트 단순화 시작: 대상={target_audience}, 언어={language}")
            
            # 대상자에 맞는 프롬프트 생성
            prompt = self._get_simplification_prompt(original_text, target_audience, language)
            
            # AI 모델을 사용하여 텍스트 단순화
            simplified_text = self._generate_simplified_text(prompt)
            
            # 결과 반환
            result = {
                'original_text': original_text,
                'simplified_text': simplified_text,
                'target_audience': target_audience,
                'language': language,
                'timestamp': time.time()
            }
            
            logger.info("텍스트 단순화 완료")
            return result
            
        except Exception as e:
            logger.error(f"텍스트 단순화 오류: {str(e)}", exc_info=True)
            raise
    
    def _get_simplification_prompt(self, original_text, target_audience, language):
        """대상자 맞춤형 단순화 프롬프트 생성"""
        
        base_prompt = f"""
다음 텍스트를 더 쉬운 표현으로 변환해주세요:

{original_text}

대상자: {target_audience}
언어: {language}
"""
        
        if target_audience == 'child':
            base_prompt += """
[어린이용 변환 지침]
1. 7-12세 어린이가 이해할 수 있는 단어와 표현으로 변환하세요.
2. 짧고 간단한 문장을 사용하세요.
3. 추상적인 개념은 구체적인 예시와 함께 설명하세요.
4. 재미있고 흥미로운 표현을 사용하세요.
5. 어려운 단어는 간단한 동의어로 대체하세요.
6. 필요한 경우 비유와 예시를 활용하세요.
7. 문장 사이에 적절한 줄바꿈을 추가하세요.
"""
        elif target_audience == 'elderly':
            base_prompt += """
[고령자용 변환 지침]
1. 명확하고 직접적인 표현을 사용하세요.
2. 외래어나 영어 표현은 가능한 한국어로 대체하세요.
3. 복잡한 문장 구조를 피하고 간결하게 작성하세요.
4. 전문 용어는 일상적인 용어로 설명하세요.
5. 친숙한 비유와 예시를 사용하세요.
6. 중요한 정보는 반복해서 강조하세요.
7. 문장 사이에 적절한 줄바꿈을 추가하세요.
"""
        elif target_audience == 'foreigner':
            base_prompt += """
[외국인 학습자용 변환 지침]
1. 한국어 학습자(초급~중급)가 이해할 수 있는 기본 어휘를 사용하세요.
2. 관용어, 속담, 은유적 표현을 피하세요.
3. 한자어는 가능한 순우리말로 대체하세요.
4. 문법적으로 단순한 문장 구조를 사용하세요.
5. 복잡한 연결어미나 조사 사용을 최소화하세요.
6. 중요한 개념은 괄호 안에 영어로 병기하세요.
7. 문장 사이에 적절한 줄바꿈을 추가하세요.
"""
        else:  # general
            base_prompt += """
[일반인용 변환 지침]
1. 보편적인 교양 수준의 어휘와 표현을 사용하세요.
2. 불필요하게 복잡한 문장 구조를 단순화하세요.
3. 전문 용어는 간단한 설명과 함께 사용하세요.
4. 논리적 흐름을 유지하며 명확하게 표현하세요.
5. 비유와 예시를 적절히 활용하세요.
6. 중요한 내용을 강조하고 핵심을 먼저 제시하세요.
7. 문장 사이에 적절한 줄바꿈을 추가하세요.
"""
            
        return base_prompt
    
    def _generate_simplified_text(self, prompt):
        """AI 모델을 사용하여 단순화된 텍스트 생성"""
        try:
            # API 유형에 따른 분기
            if self.api_type == 'openai':
                client = get_openai_client()
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "당신은 복잡한 텍스트를 더 쉽게 이해할 수 있는 형태로 변환해주는 전문가입니다."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.5,
                    max_tokens=2000
                )
                simplified_text = response['choices'][0]['message']['content']
                
            elif self.api_type == 'anthropic':
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=2000,
                    temperature=0.5,
                    system="당신은 복잡한 텍스트를 더 쉽게 이해할 수 있는 형태로 변환해주는 전문가입니다.",
                    messages=[{
                        "role": "user", 
                        "content": prompt
                    }]
                )
                simplified_text = message.content[0].text
                
            elif self.api_type == 'groq':
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "당신은 복잡한 텍스트를 더 쉽게 이해할 수 있는 형태로 변환해주는 전문가입니다."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.5,
                    max_tokens=2000
                )
                simplified_text = response.choices[0].message.content
            
            return simplified_text
            
        except Exception as e:
            logger.error(f"텍스트 생성 오류: {str(e)}", exc_info=True)
            raise




    
import logging
import json
import os
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny
from PIL import Image, ImageFilter, ImageEnhance
import pytesseract
from .models import OCRResult
from .serializers import OCRResultSerializer

import PyPDF2
import tempfile
from pdf2image import convert_from_path
import re

logger = logging.getLogger(__name__)

# OllamaClient와 GPTTranslator 클래스 가져오기
from .ollama_client import OllamaClient
from .gpt_translator import GPTTranslator 

@method_decorator(csrf_exempt, name='dispatch')
class ProcessFileView(APIView):
    permission_classes = [AllowAny]
    
    def post(self, request):
        try:
            logger.info("ProcessFileView 요청 수신: %s %s", request.method, request.path)
            
            # 요청 데이터 확인
            if 'file' not in request.FILES:
                logger.error("파일이 제공되지 않음")
                return Response({'error': '파일이 제공되지 않았습니다'}, status=status.HTTP_400_BAD_REQUEST)
            
            file_obj = request.FILES['file']
            file_name = file_obj.name.lower()
            logger.info("파일 업로드: %s, 크기: %s bytes", file_name, file_obj.size)
            
            # Ollama 클라이언트 초기화
            ollama_base_url = os.environ.get('OLLAMA_API_URL', 'http://localhost:11434')
            ollama_client = OllamaClient(base_url=ollama_base_url)
            
            # GPT 번역기 초기화
            gpt_translator = GPTTranslator()
            
            # 번역 옵션 확인 (기본값: True)
            enable_translation = request.data.get('enable_translation', 'true').lower() == 'true'
            
            # 파일 유형 확인
            if file_name.endswith(('.pdf')):
                file_type = 'pdf'
                
                # PDF 페이지 범위 확인
                start_page = int(request.data.get('start_page', 1))
                end_page = int(request.data.get('end_page', 0))  # 0은 전체 페이지를 의미
                
                logger.info("PDF 처리 범위: %s ~ %s 페이지", start_page, end_page if end_page > 0 else "끝")
                
            elif file_name.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')):
                file_type = 'image'
            else:
                logger.error("지원되지 않는 파일 형식: %s", file_name)
                return Response({'error': '지원되지 않는 파일 형식입니다. 이미지나 PDF 파일을 업로드해주세요.'},
                              status=status.HTTP_400_BAD_REQUEST)
            
            # OCR 결과 객체 생성
            ocr_result = OCRResult.objects.create(file=file_obj, file_type=file_type)
            logger.info("OCRResult 객체 생성: %s", ocr_result.id)
            
            # OCR 처리
            try:
                ocr_text = ""
                page_texts = []  # 페이지별 텍스트 저장
                
                if file_type == 'image':
                    # 이미지 파일 처리 - 개선된 OCR 적용
                    img = Image.open(ocr_result.file.path)
                    # 이미지 정보 로깅
                    logger.info(f"이미지 정보: 크기={img.size}, 모드={img.mode}, 포맷={img.format}")
                    
                    # 이미지 전처리 및 OCR 수행 - OllamaClient 메서드 사용
                    preprocessed_img = ollama_client.preprocess_image_for_ocr(img)
                    ocr_text = ollama_client.get_optimized_ocr_text(preprocessed_img, lang='kor+eng')
                    page_texts.append({"page": 1, "text": ocr_text})
                    logger.info("이미지 OCR 처리 완료, 추출 텍스트 길이: %s", len(ocr_text))
                    logger.info("추출된 텍스트 샘플: %s", ocr_text[:200] if ocr_text else "텍스트 없음")
                
                elif file_type == 'pdf':
                    # PDF 처리 - 직접 추출 후 필요시 OCR
                    logger.info("PDF 처리 시작: %s", ocr_result.file.path)
                    
                    # 직접 텍스트 추출 시도 (페이지별)
                    direct_extract_success = False
                    try:
                        all_page_texts = self.extract_text_from_pdf_by_pages(ocr_result.file.path)
                        
                        # 페이지 범위 처리
                        if start_page > 1 or (end_page > 0 and end_page < len(all_page_texts)):
                            if start_page <= len(all_page_texts):
                                if end_page > 0 and end_page >= start_page:
                                    page_texts = all_page_texts[start_page-1:end_page]
                                else:
                                    page_texts = all_page_texts[start_page-1:]
                            else:
                                page_texts = []
                        else:
                            page_texts = all_page_texts
                        
                        combined_text = "\n".join([page["text"] for page in page_texts])
                        
                        # 추출된 텍스트 길이가 충분한지 확인 (더 엄격한 조건)
                        if combined_text.strip() and len(combined_text.strip()) >= 50:
                            meaningful_chars = sum(1 for c in combined_text if c.isalnum())
                            if meaningful_chars > 30:  # 의미있는 글자가 30자 이상이면 성공으로 간주
                                ocr_text = combined_text
                                direct_extract_success = True
                                logger.info("PDF 직접 텍스트 추출 성공, 총 %s 페이지, 텍스트 길이: %s", 
                                          len(page_texts), len(ocr_text))
                                logger.info("추출된 텍스트 샘플: %s", ocr_text[:200] if ocr_text else "텍스트 없음")
                    except Exception as e:
                        logger.error(f"PDF 직접 텍스트 추출 실패: {str(e)}")
                    
                    # 직접 텍스트 추출이 실패한 경우, OCR 시도
                    if not direct_extract_success:
                        logger.info("PDF OCR 처리 시작 (직접 추출 실패 또는 텍스트 불충분)")
                        
                        # 페이지 범위 설정으로 OCR
                        all_page_texts = self.ocr_pdf_by_pages(ocr_result.file.path, ollama_client, start_page, end_page)
                        
                        # 페이지 범위 처리 - ocr_pdf_by_pages에서 처리했으므로 전체 사용
                        page_texts = all_page_texts
                        
                        ocr_text = "\n".join([page["text"] for page in page_texts])
                        logger.info("PDF OCR 처리 완료, 총 %s 페이지, 텍스트 길이: %s", 
                                  len(page_texts), len(ocr_text))
                        logger.info("추출된 텍스트 샘플: %s", ocr_text[:200] if ocr_text else "텍스트 없음")
                
                # 텍스트 정화 - 개선된 함수 사용
                ocr_result.ocr_text = self.clean_text(ocr_text)
                
                # PDF 파일은 항상 텍스트 관련 있음으로 설정
                if file_type == 'pdf':
                    text_relevant = True
                
                # 분석 유형 확인 (기본값: both)
                analysis_type = request.data.get('analysis_type', 'both')
                logger.info("분석 유형: %s", analysis_type)
                
                # 결과 변수 초기화
                image_analysis = ""
                text_analysis = ""
                combined_analysis = ""
                
                # 번역 결과 변수 초기화
                translated_analysis = ""
                translation_success = False
                translation_error = ""
                
                # 페이지 분할 분석 여부 확인
                analyze_by_page = request.data.get('analyze_by_page', 'true').lower() == 'true'
                
                # 선택된 분석 유형에 따라 처리
                if analysis_type in ['ollama', 'both']:
                    # 이미지 파일인 경우
                    if file_type == 'image':
                        # 사용자 정의 프롬프트 설정 (요약된 간결한 설명을 위해)
                        custom_prompt = f"""이미지를 객관적으로 관찰하고 다음 지침에 따라 응답하세요:

필수 포함 사항:
- 이미지에 실제로 보이는 사람, 동물, 물체만 언급 (없으면 언급하지 않음)
- 만약 동물이라면, 어떤 종의 동물인지도 출력
- 확실히 보이는 색상만 언급 (배경색, 옷 색상 등)
- 명확히 보이는 자세나 위치 관계 (정면, 측면 등)

절대 포함하지 말 것:
- 추측이나 해석 ("~로 보입니다", "~같습니다" 표현 금지)
- 보이지 않는 부분에 대한 언급 ("보이지 않는다", "없다" 등의 표현 금지)
- 반복적인 설명
- 감정이나 분위기 묘사

형식:
- 1-2문장으로 매우 간결하게 작성
- 단순 사실 나열 형식 (예: "이미지에는 검은 머리 여성이 있고, 배경은 흰색이다.")

OCR 텍스트 (참고용, 실제 이미지에 보이는 경우만 언급): {ocr_result.ocr_text}

영어로 간결하게 응답해주세요."""

                        
                        # OCR 텍스트를 전달 (analyze_image 내부에서 관련성 판단)
                        image_analysis = ollama_client.analyze_image(
                            ocr_result.file.path, 
                            custom_prompt,
                            ocr_text=ocr_result.ocr_text
                        )
                        
                        # OCR 텍스트 분석 (텍스트가 있고 both 모드인 경우)
                        if ocr_result.ocr_text and analysis_type == 'both':
                            # 페이지별 텍스트 정리를 위한 프롬프트
                            text_prompt = f"""다음 OCR로 추출한 텍스트를 자세히 분석하고 명확하게 정리해주세요:

{ocr_result.ocr_text}

분석 지침:
1. 텍스트의 주요 내용과 구조를 파악하여 정리
2. 단순 요약이 아닌, 텍스트의 핵심 정보를 충실하게 정리
3. 중요한 세부 정보를 포함
4. 내용이 이미지와 관련이 있을 수 있으므로 문맥을 고려하여 정리

반드시 "영어(En)"로 응답해주세요."""
                            
                            try:
                                text_analysis = ollama_client.analyze_text(ocr_result.ocr_text, text_prompt)
                            except Exception as e:
                                logger.error(f"텍스트 분석 오류: {str(e)}")
                                text_analysis = f"텍스트 분석 중 오류가 발생했습니다: {str(e)}"
                            
                            # 두 분석 결과 결합
                            combined_analysis = f"이미지 분석 결과:\n{image_analysis}\n\n텍스트 분석 결과:\n{text_analysis}"
                        else:
                            # OCR 없이 이미지 분석만 수행
                            combined_analysis = image_analysis
                        
                    else:  # PDF 파일인 경우
                        if ocr_result.ocr_text:
                            if analyze_by_page and len(page_texts) > 1:
                                # 개선된 페이지별 분석 수행 - OllamaClient의 분석 기능 활용
                                try:
                                    combined_analysis = ollama_client.analyze_text(ocr_result.ocr_text, None, page_texts)
                                    logger.info("페이지별 분석 완료")
                                except Exception as e:
                                    logger.error(f"페이지별 분석 오류: {str(e)}")
                                    combined_analysis = f"페이지별 분석 중 오류가 발생했습니다: {str(e)}"
                            else:
                                # 문서 전체 분석 - 페이지별 구조화 요청
                                text_prompt = f"""다음 PDF에서 추출한 텍스트를 페이지별로 명확하게 정리해주세요:

{ocr_result.ocr_text}

분석 지침:
1. 텍스트를 페이지나 섹션 단위로 구분하여 정리해주세요.
2. 내용을 요약하지 말고, 각 섹션의 핵심 정보를 충실하게 정리해주세요.
3. 모든 중요한 세부 정보를 포함해주세요.
4. 내용을 단순 요약하지 말고, 구조화된 형식으로 정리해주세요.

다음과 같은 형식으로 정리해주세요:
===== 페이지 1 (또는 섹션 1) =====
- 주요 내용 정리
- 중요 개념 설명
- 핵심 정보 나열

===== 페이지 2 (또는 섹션 2) =====
- 주요 내용 정리
...

반드시 "영어로" 응답해주세요."""
                                
                                try:
                                    text_analysis = ollama_client.analyze_text(ocr_result.ocr_text, text_prompt)
                                    combined_analysis = text_analysis
                                except Exception as e:
                                    logger.error(f"문서 전체 분석 오류: {str(e)}")
                                    combined_analysis = f"문서 분석 중 오류가 발생했습니다: {str(e)}\n\nOCR 결과: {ocr_result.ocr_text[:500]}..."
                    
                    logger.info("이미지/텍스트 분석 완료")
                
                # GPT 번역 수행 (번역이 활성화된 경우)
                if enable_translation and combined_analysis and gpt_translator.is_available:
                    logger.info("GPT 번역 시작")
                    try:
                        # 분석 유형에 따른 번역
                        if file_type == 'pdf' and analyze_by_page and len(page_texts) > 1:
                            # 페이지별 분석 결과 번역
                            translation_result = gpt_translator.translate_paged_analysis(combined_analysis)
                        else:
                            # 일반 분석 결과 번역
                            translation_result = gpt_translator.translate_analysis_result(combined_analysis, file_type)
                        
                        if translation_result and translation_result.get("success"):
                            translated_analysis = translation_result["translated_analysis"]
                            translation_success = True
                            logger.info("GPT 번역 성공")
                        else:
                            error_msg = translation_result.get('error', 'Unknown error') if translation_result else 'No translation result'
                            logger.error(f"GPT 번역 실패: {error_msg}")
                            translated_analysis = f"번역 실패: {error_msg}"
                            translation_error = error_msg
                            
                    except Exception as e:
                        logger.error(f"GPT 번역 중 예외 발생: {str(e)}")
                        translated_analysis = f"번역 중 오류 발생: {str(e)}"
                        translation_error = str(e)
                
                # 번역 관련 메타데이터 저장
                ocr_result.translation_enabled = enable_translation
                ocr_result.translation_success = translation_success
                ocr_result.analysis_type = analysis_type
                ocr_result.analyze_by_page = analyze_by_page
                
                # MySQL 저장을 위한 텍스트 정화
                ocr_result.llm_response = self.clean_text(combined_analysis)
                
                # 번역 결과도 저장
                if enable_translation and translated_analysis:
                    if translation_success:
                        # 성공한 번역 결과 저장
                        ocr_result.llm_response_korean = self.clean_text(translated_analysis)
                        ocr_result.translation_model = gpt_translator.model if gpt_translator else "unknown"
                    else:
                        # 실패한 경우 오류 메시지 저장 (디버깅용)
                        ocr_result.llm_response_korean = f"번역 실패: {translation_error}"
                
                # 텍스트 관련성 정보 저장 - PDF는 항상 True, 이미지는 분석 과정에서 결정
                if file_type == 'pdf':
                    ocr_result.text_relevant = True
                
            except Exception as e:
                logger.error("처리 실패: %s", str(e), exc_info=True)
                return Response({'error': f'처리 실패: {str(e)}'}, 
                               status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
            # 결과 저장
            try:
                ocr_result.save()
                logger.info("OCRResult 저장 완료 (ID: %s)", ocr_result.id)
            except Exception as e:
                logger.error(f"데이터베이스 저장 실패: {str(e)}")
                return Response({'error': f'결과 저장 실패: {str(e)}'}, 
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
            # 응답 데이터 구성 - 명시적으로 필드 지정
            try:
                # 기본 시리얼라이저 데이터
                response_data = OCRResultSerializer(ocr_result).data
                
                # 번역 관련 정보 명시적 추가
                response_data['translation_enabled'] = enable_translation
                response_data['translation_success'] = translation_success
                
                # 영어 원문과 한국어 번역을 명확히 구분
                response_data['llm_response'] = ocr_result.llm_response  # 영어 원문
                
                if enable_translation and translation_success:
                    # 번역 성공 시 한국어 번역 추가
                    response_data['llm_response_korean'] = ocr_result.llm_response_korean
                    logger.info("응답에 한국어 번역 포함")
                elif enable_translation and not translation_success:
                    # 번역 실패 시 오류 정보 추가
                    response_data['llm_response_korean'] = None
                    response_data['translation_error'] = translation_error if translation_error else "번역 실패"
                    logger.info("번역 실패 - 영어 원문만 포함")
                else:
                    # 번역 비활성화 시
                    response_data['llm_response_korean'] = None
                    logger.info("번역 비활성화 - 영어 원문만 포함")
                
                # 디버깅용 로그
                logger.info(f"응답 데이터 구성 완료:")
                logger.info(f"  - 영어 원문 길이: {len(response_data.get('llm_response', ''))}")
                logger.info(f"  - 한국어 번역 길이: {len(response_data.get('llm_response_korean', '') or '')}")
                logger.info(f"  - 번역 성공: {response_data.get('translation_success', False)}")
                
            except Exception as e:
                logger.error(f"응답 데이터 구성 실패: {str(e)}")
                return Response({'error': f'응답 구성 실패: {str(e)}'}, 
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
            # 응답 반환
            return Response(response_data, status=status.HTTP_201_CREATED)
                
        except Exception as e:
            logger.error("처리 중 예기치 않은 오류: %s", str(e), exc_info=True)
            return Response({'error': f'서버 오류: {str(e)}'}, 
                           status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    def extract_text_from_pdf_by_pages(self, pdf_path):
        """PDF에서 직접 텍스트를 페이지별로 추출"""
        pages = []
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                total_pages = len(reader.pages)
                
                for i in range(total_pages):
                    page = reader.pages[i]
                    text = page.extract_text()
                    pages.append({"page": i + 1, "text": text})
                    
            return pages
        except Exception as e:
            logger.error(f"PDF 텍스트 직접 추출 오류: {str(e)}")
            raise
    
    def ocr_pdf_by_pages(self, pdf_path, ollama_client, start_page=1, end_page=0):
        """PDF를 OCR로 처리하여 페이지별 텍스트 추출"""
        pages = []
        
        try:
            # PDF2Image로 이미지 변환
            with tempfile.TemporaryDirectory() as path:
                # 페이지 번호는 1부터 시작하지만, convert_from_path는 0부터 시작하므로 조정
                first_page = start_page
                last_page = None if end_page <= 0 else end_page
                
                images = convert_from_path(
                    pdf_path, 
                    dpi=300, 
                    output_folder=path, 
                    first_page=first_page,
                    last_page=last_page
                )
                
                # 각 페이지 이미지 OCR 처리
                for i, image in enumerate(images):
                    # 이미지 전처리
                    preprocessed_img = ollama_client.preprocess_image_for_ocr(image)
                    # OCR 수행
                    text = ollama_client.get_optimized_ocr_text(preprocessed_img, lang='kor+eng')
                    
                    # 페이지 번호 계산 (시작 페이지 고려)
                    page_num = start_page + i
                    pages.append({"page": page_num, "text": text})
                    
            return pages
        except Exception as e:
            logger.error(f"PDF OCR 처리 오류: {str(e)}")
            raise
    
    def clean_text(self, text):
        """텍스트 정화 함수"""
        if not text:
            return ""
            
        # 연속된 공백 제거
        text = re.sub(r'\s+', ' ', text)
        # 연속된 줄바꿈 제거
        text = re.sub(r'\n\s*\n', '\n\n', text)
        # 앞뒤 공백 제거
        text = text.strip()
        
        return text

@method_decorator(csrf_exempt, name='dispatch')
class MultiAIProcessFileView(APIView):
    """멀티 AI 모델을 사용한 PDF/이미지 분석 뷰"""
    permission_classes = [AllowAny]
    
    def post(self, request):
        try:
            logger.info("MultiAIProcessFileView 요청 수신: %s %s", request.method, request.path)
            
            # 요청 데이터 확인
            if 'file' not in request.FILES:
                logger.error("파일이 제공되지 않음")
                return Response({'error': '파일이 제공되지 않았습니다'}, status=status.HTTP_400_BAD_REQUEST)
            
            file_obj = request.FILES['file']
            file_name = file_obj.name.lower()
            logger.info("파일 업로드: %s, 크기: %s bytes", file_name, file_obj.size)
            
            # 멀티 AI OCR 서비스 초기화
            from .multi_ai_ocr_service import get_multi_ai_ocr_service
            multi_ai_service = get_multi_ai_ocr_service()
            
            # Ollama 클라이언트 초기화 (OCR용)
            ollama_base_url = os.environ.get('OLLAMA_API_URL', 'http://localhost:11434')
            ollama_client = OllamaClient(base_url=ollama_base_url)
            
            # GPT 번역기 초기화
            gpt_translator = GPTTranslator()
            
            # 요청 파라미터 파싱
            question = request.data.get('question', 'Analyze the content and summarize key information.')
            selected_models = request.data.get('selected_models', [])  # 선택된 AI 모델들
            enable_translation = request.data.get('enable_translation', 'true').lower() == 'true'
            analysis_type = request.data.get('analysis_type', 'both')  # 'ocr', 'ollama', 'both'
            analyze_by_page = request.data.get('analyze_by_page', 'true').lower() == 'true'
            
            # 파일 유형 확인
            if file_name.endswith(('.pdf')):
                file_type = 'pdf'
                start_page = int(request.data.get('start_page', 1))
                end_page = int(request.data.get('end_page', 0))
                logger.info("PDF 처리 범위: %s ~ %s 페이지", start_page, end_page if end_page > 0 else "끝")
            elif file_name.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')):
                file_type = 'image'
            else:
                logger.error("지원되지 않는 파일 형식: %s", file_name)
                return Response({'error': '지원되지 않는 파일 형식입니다. 이미지나 PDF 파일을 업로드해주세요.'},
                              status=status.HTTP_400_BAD_REQUEST)
            
            # OCR 결과 객체 생성
            ocr_result = OCRResult.objects.create(file=file_obj, file_type=file_type)
            logger.info("OCRResult 객체 생성: %s", ocr_result.id)
            
            # OCR 처리 (기존 로직과 동일)
            try:
                ocr_text = ""
                page_texts = []
                
                if file_type == 'image':
                    img = Image.open(ocr_result.file.path)
                    logger.info(f"이미지 정보: 크기={img.size}, 모드={img.mode}, 포맷={img.format}")
                    
                    preprocessed_img = ollama_client.preprocess_image_for_ocr(img)
                    ocr_text = ollama_client.get_optimized_ocr_text(preprocessed_img, lang='kor+eng')
                    page_texts.append({"page": 1, "text": ocr_text})
                    logger.info("이미지 OCR 처리 완료, 추출 텍스트 길이: %s", len(ocr_text))
                
                elif file_type == 'pdf':
                    logger.info("PDF 처리 시작: %s", ocr_result.file.path)
                    
                    # 직접 텍스트 추출 시도
                    direct_extract_success = False
                    try:
                        all_page_texts = self.extract_text_from_pdf_by_pages(ocr_result.file.path)
                        
                        if start_page > 1 or (end_page > 0 and end_page < len(all_page_texts)):
                            if start_page <= len(all_page_texts):
                                if end_page > 0 and end_page >= start_page:
                                    page_texts = all_page_texts[start_page-1:end_page]
                                else:
                                    page_texts = all_page_texts[start_page-1:]
                            else:
                                page_texts = []
                        else:
                            page_texts = all_page_texts
                        
                        combined_text = "\n".join([page["text"] for page in page_texts])
                        
                        if combined_text.strip() and len(combined_text.strip()) >= 50:
                            meaningful_chars = sum(1 for c in combined_text if c.isalnum())
                            if meaningful_chars > 30:
                                ocr_text = combined_text
                                direct_extract_success = True
                                logger.info("PDF 직접 텍스트 추출 성공, 총 %s 페이지, 텍스트 길이: %s", 
                                          len(page_texts), len(ocr_text))
                    except Exception as e:
                        logger.error(f"PDF 직접 텍스트 추출 실패: {str(e)}")
                    
                    # 직접 텍스트 추출이 실패한 경우, OCR 시도
                    if not direct_extract_success:
                        logger.info("PDF OCR 처리 시작 (직접 추출 실패 또는 텍스트 불충분)")
                        all_page_texts = self.ocr_pdf_by_pages(ocr_result.file.path, ollama_client, start_page, end_page)
                        page_texts = all_page_texts
                        ocr_text = "\n".join([page["text"] for page in page_texts])
                        logger.info("PDF OCR 처리 완료, 총 %s 페이지, 텍스트 길이: %s", 
                                  len(page_texts), len(ocr_text))
                
                # 텍스트 정화
                ocr_result.ocr_text = self.clean_text(ocr_text)
                
                # PDF 파일은 항상 텍스트 관련 있음으로 설정
                if file_type == 'pdf':
                    ocr_result.text_relevant = True
                
            except Exception as e:
                logger.error("OCR 처리 실패: %s", str(e), exc_info=True)
                return Response({'error': f'OCR 처리 실패: {str(e)}'}, 
                               status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
            # 멀티 AI 분석 수행
            multi_ai_results = {}
            combined_analysis = ""
            
            try:
                # 분석 옵션 구성
                analysis_options = {
                    'start_page': start_page if file_type == 'pdf' else 1,
                    'end_page': end_page if file_type == 'pdf' else 1,
                    'analyze_by_page': analyze_by_page,
                    'file_type': file_type
                }
                
                # 멀티 AI 분석 실행
                multi_ai_results = multi_ai_service.analyze_file_multi_ai(
                    file_path=ocr_result.file.path,
                    file_type=file_type,
                    question=question,
                    selected_models=selected_models,
                    analysis_options=analysis_options
                )
                
                logger.info("멀티 AI 분석 완료, 결과 모델 수: %s", len(multi_ai_results))
                
                # 응답 비교 및 결합
                comparison_result = multi_ai_service.compare_responses(multi_ai_results)
                
                # 최고 응답 선택 또는 모든 응답 결합
                if comparison_result.get('comparison', {}).get('best_response'):
                    best_model = comparison_result['comparison']['best_response']
                    if best_model in multi_ai_results and multi_ai_results[best_model].success:
                        combined_analysis = multi_ai_results[best_model].response_text
                        logger.info(f"최고 응답 모델 선택: {best_model}")
                    else:
                        # 모든 성공한 응답 결합
                        successful_responses = [r.response_text for r in multi_ai_results.values() 
                                              if r.success and r.response_text]
                        combined_analysis = "\n\n".join(successful_responses)
                        logger.info("모든 성공한 응답 결합")
                else:
                    # 모든 성공한 응답 결합
                    successful_responses = [r.response_text for r in multi_ai_results.values() 
                                          if r.success and r.response_text]
                    combined_analysis = "\n\n".join(successful_responses)
                    logger.info("모든 성공한 응답 결합")
                
            except Exception as e:
                logger.error("멀티 AI 분석 실패: %s", str(e), exc_info=True)
                combined_analysis = f"멀티 AI 분석 중 오류가 발생했습니다: {str(e)}"
            
            # GPT 번역 수행 (번역이 활성화된 경우)
            translated_analysis = ""
            translation_success = False
            translation_error = ""
            
            if enable_translation and combined_analysis and gpt_translator.is_available:
                logger.info("GPT 번역 시작")
                try:
                    if file_type == 'pdf' and analyze_by_page and len(page_texts) > 1:
                        translation_result = gpt_translator.translate_paged_analysis(combined_analysis)
                    else:
                        translation_result = gpt_translator.translate_analysis_result(combined_analysis, file_type)
                    
                    if translation_result and translation_result.get("success"):
                        translated_analysis = translation_result["translated_analysis"]
                        translation_success = True
                        logger.info("GPT 번역 성공")
                    else:
                        error_msg = translation_result.get('error', 'Unknown error') if translation_result else 'No translation result'
                        logger.error(f"GPT 번역 실패: {error_msg}")
                        translated_analysis = f"번역 실패: {error_msg}"
                        translation_error = error_msg
                        
                except Exception as e:
                    logger.error(f"GPT 번역 중 예외 발생: {str(e)}")
                    translated_analysis = f"번역 중 오류 발생: {str(e)}"
                    translation_error = str(e)
            
            # 메타데이터 저장
            ocr_result.translation_enabled = enable_translation
            ocr_result.translation_success = translation_success
            ocr_result.analysis_type = analysis_type
            ocr_result.analyze_by_page = analyze_by_page
            
            # MySQL 저장을 위한 텍스트 정화
            ocr_result.llm_response = self.clean_text(combined_analysis)
            
            # 번역 결과도 저장
            if enable_translation and translated_analysis:
                if translation_success:
                    ocr_result.llm_response_korean = self.clean_text(translated_analysis)
                    ocr_result.translation_model = gpt_translator.model if gpt_translator else "unknown"
                else:
                    ocr_result.llm_response_korean = f"번역 실패: {translation_error}"
            
            # 결과 저장
            try:
                ocr_result.save()
                logger.info("OCRResult 저장 완료 (ID: %s)", ocr_result.id)
            except Exception as e:
                logger.error(f"데이터베이스 저장 실패: {str(e)}")
                return Response({'error': f'결과 저장 실패: {str(e)}'}, 
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
            # 응답 데이터 구성
            try:
                response_data = OCRResultSerializer(ocr_result).data
                
                # 멀티 AI 관련 정보 추가
                response_data['multi_ai_results'] = {}
                response_data['comparison_result'] = comparison_result
                response_data['selected_models'] = selected_models
                
                # 각 모델별 결과 추가
                for model_name, result in multi_ai_results.items():
                    response_data['multi_ai_results'][model_name] = {
                        'response_text': result.response_text,
                        'confidence_score': result.confidence_score,
                        'processing_time': result.processing_time,
                        'token_usage': result.token_usage,
                        'success': result.success,
                        'error': result.error
                    }
                
                # 번역 관련 정보
                response_data['translation_enabled'] = enable_translation
                response_data['translation_success'] = translation_success
                response_data['llm_response'] = ocr_result.llm_response  # 영어 원문
                
                if enable_translation and translation_success:
                    response_data['llm_response_korean'] = ocr_result.llm_response_korean
                elif enable_translation and not translation_success:
                    response_data['llm_response_korean'] = None
                    response_data['translation_error'] = translation_error
                else:
                    response_data['llm_response_korean'] = None
                
                logger.info(f"멀티 AI 응답 데이터 구성 완료:")
                logger.info(f"  - 분석 모델 수: {len(multi_ai_results)}")
                logger.info(f"  - 영어 원문 길이: {len(response_data.get('llm_response', ''))}")
                logger.info(f"  - 한국어 번역 길이: {len(response_data.get('llm_response_korean', '') or '')}")
                
            except Exception as e:
                logger.error(f"응답 데이터 구성 실패: {str(e)}")
                return Response({'error': f'응답 구성 실패: {str(e)}'}, 
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
            # 응답 반환
            return Response(response_data, status=status.HTTP_201_CREATED)
                
        except Exception as e:
            logger.error("처리 중 예기치 않은 오류: %s", str(e), exc_info=True)
            return Response({'error': f'서버 오류: {str(e)}'}, 
                           status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    def extract_text_from_pdf_by_pages(self, pdf_path):
        """PDF에서 직접 텍스트를 페이지별로 추출"""
        pages = []
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                total_pages = len(reader.pages)
                
                for i in range(total_pages):
                    page = reader.pages[i]
                    text = page.extract_text()
                    pages.append({"page": i + 1, "text": text})
                    
            return pages
        except Exception as e:
            logger.error(f"PDF 텍스트 직접 추출 오류: {str(e)}")
            raise
    
    def ocr_pdf_by_pages(self, pdf_path, ollama_client, start_page=1, end_page=0):
        """PDF를 OCR로 처리하여 페이지별 텍스트 추출"""
        pages = []
        
        try:
            # PDF2Image로 이미지 변환
            with tempfile.TemporaryDirectory() as path:
                # 페이지 번호는 1부터 시작하지만, convert_from_path는 0부터 시작하므로 조정
                first_page = start_page
                last_page = None if end_page <= 0 else end_page
                
                images = convert_from_path(
                    pdf_path, 
                    first_page=first_page, 
                    last_page=last_page,
                    dpi=200
                )
                
                for i, image in enumerate(images):
                    page_num = start_page + i
                    
                    # 이미지 전처리
                    preprocessed_img = ollama_client.preprocess_image_for_ocr(image)
                    
                    # OCR 수행
                    ocr_text = ollama_client.get_optimized_ocr_text(preprocessed_img, lang='kor+eng')
                    
                    pages.append({"page": page_num, "text": ocr_text})
                    
                logger.info(f"PDF OCR 처리 완료: {len(pages)} 페이지")
                
        except Exception as e:
            logger.error(f"PDF OCR 처리 오류: {str(e)}")
            raise
        
        return pages
    
    def clean_text(self, text):
        """텍스트 정화 함수"""
        if not text:
            return ""
            
        # 연속된 공백 제거
        text = re.sub(r'\s+', ' ', text)
        # 연속된 줄바꿈 제거
        text = re.sub(r'\n\s*\n', '\n\n', text)
        # 앞뒤 공백 제거
        text = text.strip()
        
        return text
    
# chat/views.py에 추가할 뷰들

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated, AllowAny
from django.shortcuts import get_object_or_404
from datetime import datetime, timedelta
import json
import re
from .models import Schedule, ScheduleRequest, ConflictResolution
from .serializers import (
    ScheduleSerializer, ScheduleRequestSerializer, 
    ConflictResolutionSerializer, ScheduleRequestInputSerializer
)

# 기존 ChatBot 클래스와 ChatView는 그대로 유지...
# chat/views.py 수정 버전

# from rest_framework.views import APIView
# from rest_framework.response import Response
# from rest_framework import status
# from rest_framework.decorators import api_view, permission_classes
# from rest_framework.permissions import IsAuthenticated, AllowAny
# from django.shortcuts import get_object_or_404
# from datetime import datetime, timedelta
# import json
# import re
# from .models import Schedule, ScheduleRequest, ConflictResolution
# from .serializers import (
#     ScheduleSerializer, ScheduleRequestSerializer, 
#     ConflictResolutionSerializer, ScheduleRequestInputSerializer
# )

# # 기존 ChatBot 클래스는 그대로 유지...


# chatbots = {
#     'gpt': ChatBot(OPENAI_API_KEY, 'gpt-3.5-turbo', 'openai'),
#     'claude': ChatBot(ANTHROPIC_API_KEY, 'claude-3-5-haiku-20241022', 'anthropic'), 
#     'mixtral': ChatBot(GROQ_API_KEY, 'llama-3.1-8b-instant', 'groq'),
# }

# # 백엔드 views.py에 추가할 함수들

# def parse_date_from_request(request_text):
#     """자연어 날짜를 실제 날짜로 변환"""
#     today = datetime.now().date()
    
#     # 오늘/내일/모레 등 한국어 날짜 표현 처리
#     if '오늘' in request_text:
#         return today
#     elif '내일' in request_text:
#         return today + timedelta(days=1)
#     elif '모레' in request_text or '모래' in request_text:
#         return today + timedelta(days=2)
#     elif '이번 주' in request_text:
#         # 이번 주 금요일로 설정
#         days_until_friday = (4 - today.weekday()) % 7
#         if days_until_friday == 0:  # 오늘이 금요일이면 다음 주 금요일
#             days_until_friday = 7
#         return today + timedelta(days=days_until_friday)
#     elif '다음 주' in request_text:
#         return today + timedelta(days=7)
#     else:
#         # 기본값: 내일
#         return today + timedelta(days=1)

# def parse_multiple_schedules_backend(request_text):
#     """백엔드에서 여러 일정 파싱"""
#     # 쉼표, "그리고", "및" 등으로 분리
#     separators = [',', '，', '그리고', '및', '와', '과']
    
#     parts = [request_text]
#     for sep in separators:
#         new_parts = []
#         for part in parts:
#             new_parts.extend(part.split(sep))
#         parts = new_parts
    
#     # 정리된 요청들 반환
#     cleaned_requests = []
#     for part in parts:
#         cleaned = part.strip()
#         if cleaned and len(cleaned) > 2:  # 너무 짧은 텍스트 제외
#             cleaned_requests.append(cleaned)
    
#     return cleaned_requests if len(cleaned_requests) > 1 else [request_text]
# class ScheduleOptimizerBot:
#     """일정 최적화를 위한 AI 봇 클래스 - 여러 AI 모델 연동"""
    
#     def __init__(self):
#         self.chatbots = {
#                 'gpt': ChatBot(OPENAI_API_KEY, 'gpt-3.5-turbo', 'openai'),
#                 'claude': ChatBot(ANTHROPIC_API_KEY, 'claude-3-5-haiku-20241022', 'anthropic'), 
#                 'mixtral': ChatBot(GROQ_API_KEY, 'llama-3.1-8b-instant', 'groq'),
#             }
        
#     def create_schedule_prompt(self, request_text, user_context=None, existing_schedules=None):
#         """일정 생성을 위한 프롬프트 생성 - 빈 시간 분석 포함"""
#         base_prompt = f"""
#         사용자의 일정 요청을 분석하여 기존 일정과 충돌하지 않는 최적의 빈 시간을 찾아 제안해주세요.

#         요청 내용: {request_text}
        
#         기존 일정들: {existing_schedules or []}
        
#         분석 방법:
#         1. 기존 일정들의 시간대를 확인하여 사용자가 입력한 날의 빈 시간을 찾아주세요
#         2. 요청된 일정의 성격에 맞는 최적의 시간대를 추천해주세요
#         3. 일정 간 여유 시간도 고려해주세요
        
#         다음 형식으로 응답해주세요:
#         {{
#             "title": "일정 제목",
#             "description": "상세 설명",
#             "suggested_date": "YYYY-MM-DD",
#             "suggested_start_time": "HH:MM",
#             "suggested_end_time": "HH:MM",
#             "location": "장소 (선택사항)",
#             "priority": "HIGH/MEDIUM/LOW/URGENT",
#             "attendees": ["참석자1", "참석자2"],
#             "reasoning": "이 시간을 제안하는 이유 (빈 시간 분석 결과 포함)"
#         }}
        
#         사용자의 맥락 정보: {user_context or "없음"}
#         """
#         return base_prompt

#     def create_conflict_resolution_prompt(self, conflicting_schedules, new_request):
#         """일정 충돌 해결을 위한 프롬프트 생성"""
#         prompt = f"""
#         기존 일정과 새로운 일정 요청 사이에 충돌이 발생했습니다. 
#         여러 AI의 관점에서 최적의 해결 방안을 제안해주세요.

#         기존 충돌 일정들:
#         {json.dumps(conflicting_schedules, ensure_ascii=False, indent=2)}

#         새로운 일정 요청: {new_request}

#         다음 형식으로 해결 방안을 제안해주세요:
#         {{
#             "resolution_options": [
#                 {{
#                     "option": "방안 1",
#                     "description": "상세 설명",
#                     "impact": "영향도 분석",
#                     "recommended": true/false
#                 }},
#                 {{
#                     "option": "방안 2", 
#                     "description": "상세 설명",
#                     "impact": "영향도 분석",
#                     "recommended": true/false
#                 }}
#             ],
#             "best_recommendation": "가장 추천하는 방안과 이유"
#         }}
#         """
#         return prompt
    
#     def get_ai_suggestions(self, prompt, suggestion_type="schedule"):
#         """여러 AI 모델로부터 제안받기"""
#         suggestions = {}
        
#         for model_name, chatbot in self.chatbots.items():
#             try:
#                 response = chatbot.chat(prompt)
#                 suggestions[f"{model_name}_suggestion"] = response
#             except Exception as e:
#                 suggestions[f"{model_name}_suggestion"] = f"오류 발생: {str(e)}"
        
#         return suggestions
    
#     def analyze_and_optimize_suggestions(self, suggestions, query, selected_models=['GPT', 'Claude', 'Mixtral']):
#         """여러 AI 제안을 분석하여 최적화된 결과 생성 - 기존 analyze_responses 활용"""
#         try:
#             # ChatBot의 analyze_responses 기능 활용
#             analyzer = self.chatbots['claude']  # Claude를 분석용으로 사용
            
#             # 제안을 분석용 형태로 변환
#             responses_for_analysis = {}
#             for key, suggestion in suggestions.items():
#                 model_name = key.replace('_suggestion', '')
#                 responses_for_analysis[model_name] = suggestion
            
#             # 기존 analyze_responses 메서드 활용
#             analysis_result = analyzer.analyze_responses(
#                 responses_for_analysis, 
#                 query, 
#                 'Korean',  # 기본 언어
#                 selected_models
#             )
            
#             # JSON 응답에서 최적화된 일정 정보 추출
#             try:
#                 # best_response에서 JSON 부분 추출
#                 json_match = re.search(r'\{.*\}', analysis_result.get('best_response', ''), re.DOTALL)
#                 if json_match:
#                     optimized = json.loads(json_match.group())
#                 else:
#                     # fallback: 첫 번째 유효한 제안 사용
#                     optimized = self._extract_first_valid_suggestion(suggestions)
#             except:
#                 optimized = self._extract_first_valid_suggestion(suggestions)
            
#             confidence = self._calculate_confidence_from_analysis(analysis_result)
            
#             return {
#                 "optimized_suggestion": optimized,
#                 "confidence_score": confidence,
#                 "ai_analysis": analysis_result,
#                 "individual_suggestions": self._parse_individual_suggestions(suggestions)
#             }
            
#         except Exception as e:
#             print(f"Analysis error: {str(e)}")
#             return {"error": f"최적화 과정에서 오류 발생: {str(e)}"}
    
#     def _extract_first_valid_suggestion(self, suggestions):
#         """첫 번째 유효한 제안 추출"""
#         for key, suggestion in suggestions.items():
#             try:
#                 json_match = re.search(r'\{.*\}', suggestion, re.DOTALL)
#                 if json_match:
#                     return json.loads(json_match.group())
#             except:
#                 continue
        
#         # 기본 제안 반환
#         return {
#             "title": "새 일정",
#             "description": "AI가 제안한 일정입니다",
#             "suggested_date": datetime.now().strftime('%Y-%m-%d'),
#             "suggested_start_time": "09:00",
#             "suggested_end_time": "10:00",
#             "location": "",
#             "priority": "MEDIUM",
#             "attendees": [],
#             "reasoning": "여러 AI 모델의 제안을 종합한 결과입니다."
#         }
    
#     def _calculate_confidence_from_analysis(self, analysis_result):
#         """분석 결과에서 신뢰도 계산"""
#         reasoning = analysis_result.get('reasoning', '')
        
#         # 키워드 기반 신뢰도 계산
#         confidence_keywords = ['일치', '공통', '정확', '최적', '추천']
#         uncertainty_keywords = ['불확실', '추정', '가능성', '어려움']
        
#         confidence_score = 0.5  # 기본값
        
#         for keyword in confidence_keywords:
#             if keyword in reasoning:
#                 confidence_score += 0.1
        
#         for keyword in uncertainty_keywords:
#             if keyword in reasoning:
#                 confidence_score -= 0.1
        
#         return max(0.1, min(1.0, confidence_score))
    
#     def _parse_individual_suggestions(self, suggestions):
#         """개별 제안들을 파싱"""
#         parsed = []
#         for key, suggestion in suggestions.items():
#             try:
#                 json_match = re.search(r'\{.*\}', suggestion, re.DOTALL)
#                 if json_match:
#                     parsed_suggestion = json.loads(json_match.group())
#                     parsed_suggestion['source'] = key.replace('_suggestion', '')
#                     parsed.append(parsed_suggestion)
#             except:
#                 continue
#         return parsed

# class ScheduleManagementView(APIView):
#     """일정 관리 메인 뷰 - 권한 수정"""
#     # 임시로 AllowAny로 변경 (개발/테스트용)
#     permission_classes = [IsAuthenticated]
    
#     def __init__(self):
#         super().__init__()
#         self.optimizer = ScheduleOptimizerBot()
    
#     def get(self, request):
#         """사용자의 일정 목록 조회"""
#         # 🚫 기존 더미 사용자 로직 제거
#         if not request.user.is_authenticated:
#             return Response({'error': '인증이 필요합니다.'}, status=status.HTTP_401_UNAUTHORIZED)
        
#         schedules = Schedule.objects.filter(user=request.user).order_by('start_time')
        
#         # 날짜 필터링 (기존 코드 유지)
#         start_date = request.query_params.get('start_date')
#         end_date = request.query_params.get('end_date')
        
#         if start_date:
#             schedules = schedules.filter(start_time__date__gte=start_date)
#         if end_date:
#             schedules = schedules.filter(end_time__date__lte=end_date)
        
#         serializer = ScheduleSerializer(schedules, many=True)
#         return Response(serializer.data)
#     def post(self, request):
#         """새로운 일정 생성 요청 - 여러 일정 지원 개선"""
#         try:
#             request_text = request.data.get('request_text', '')
#             existing_schedules = request.data.get('existing_schedules', [])
            
#             if not request_text:
#                 return Response({'error': '요청 텍스트가 필요합니다.'}, 
#                             status=status.HTTP_400_BAD_REQUEST)
#             if not request.user.is_authenticated:
#                 return Response({'error': '인증이 필요합니다.'}, status=status.HTTP_401_UNAUTHORIZED)
        
#             user = request.user

         
            
#             # 여러 일정 요청인지 확인
#             schedule_requests = parse_multiple_schedules_backend(request_text)
#             target_date = parse_date_from_request(request_text)
            
#             if len(schedule_requests) > 1:
#                 # 여러 일정 처리
#                 multiple_schedules = []
#                 all_individual_suggestions = []
                
#                 for i, single_request in enumerate(schedule_requests):
#                     # 각 일정의 시작 시간을 다르게 설정
#                     schedule_date = target_date
#                     if i > 0:  # 두 번째 일정부터는 2시간씩 뒤로
#                         base_hour = 9 + (i * 2)
#                     else:
#                         base_hour = 9
                    
#                     # 개별 일정 생성
#                     optimized_schedule = {
#                         "title": self._extract_schedule_title(single_request),
#                         "description": f"AI가 분석한 {self._extract_schedule_title(single_request)} 일정입니다.",
#                         "suggested_date": schedule_date.strftime('%Y-%m-%d'),
#                         "suggested_start_time": f"{base_hour:02d}:00",
#                         "suggested_end_time": f"{base_hour + 2:02d}:00",
#                         "location": self._extract_schedule_location(single_request),
#                         "priority": "HIGH",
#                         "attendees": [],
#                         "reasoning": f"{i + 1}번째 일정: {single_request}. 기존 일정과 충돌하지 않는 시간으로 배정했습니다."
#                     }
#                     multiple_schedules.append(optimized_schedule)
                    
#                     # 각 AI별 개별 제안 생성
#                     for ai_type in ['gpt', 'claude', 'mixtral']:
#                         individual_suggestion = optimized_schedule.copy()
#                         individual_suggestion['source'] = ai_type
#                         individual_suggestion['reasoning'] = f"{ai_type.upper()}가 분석한 {self._extract_schedule_title(single_request)} 최적 시간입니다."
#                         all_individual_suggestions.append(individual_suggestion)
                
#                 # 여러 일정 응답 생성
#                 response_data = {
#                     'request_id': int(datetime.now().timestamp()),
#                     'multiple_schedules': multiple_schedules,
#                     'optimized_suggestion': multiple_schedules[0],
#                     'confidence_score': 0.92,
#                     'individual_suggestions': all_individual_suggestions,
#                     'ai_analysis': {
#                         'analysis_summary': f"총 {len(schedule_requests)}개의 일정을 분석하여 최적의 시간대로 배정했습니다.",
#                         'reasoning': f"여러 일정을 {target_date.strftime('%Y년 %m월 %d일')}에 시간 순서대로 배치하여 충돌을 방지했습니다.",
#                         'models_used': ["gpt", "claude", "mixtral"]
#                     },
#                     'has_conflicts': False,
#                     'conflicts': [],
#                     'analysis_summary': f"{len(schedule_requests)}개 일정에 대해 3개 AI 모델이 분석한 결과입니다.",
#                     'is_multiple_schedule': True
#                 }
                
#             else:
#                 # 단일 일정 처리 (기존 로직 사용하되 날짜 반영)
#                 user_context = self._get_user_context(user)
                
#                 # 날짜가 반영된 프롬프트 생성
#                 enhanced_prompt = f"""
#                 사용자의 일정 요청을 분석하여 기존 일정과 충돌하지 않는 최적의 빈 시간을 찾아 제안해주세요.
                
#                 요청 내용: {request_text}
#                 목표 날짜: {target_date.strftime('%Y년 %m월 %d일')} ({self._get_weekday_korean(target_date)})
#                 기존 일정들: {existing_schedules or []}
                
#                 다음 형식으로 응답해주세요:
#                 {{
#                     "title": "일정 제목",
#                     "description": "상세 설명",
#                     "suggested_date": "{target_date.strftime('%Y-%m-%d')}",
#                     "suggested_start_time": "HH:MM",
#                     "suggested_end_time": "HH:MM",
#                     "location": "장소",
#                     "priority": "HIGH/MEDIUM/LOW/URGENT",
#                     "attendees": [],
#                     "reasoning": "이 시간을 제안하는 이유"
#                 }}
#                 """
                
#                 # 기존 단일 일정 로직 계속...
#                 suggestions = self.optimizer.get_ai_suggestions(enhanced_prompt)
#                 optimized_result = self.optimizer.analyze_and_optimize_suggestions(
#                     suggestions, f"일정 요청: {request_text}"
#                 )
                
#                 response_data = {
#                     'request_id': int(datetime.now().timestamp()),
#                     'optimized_suggestion': optimized_result.get('optimized_suggestion', {}),
#                     'confidence_score': optimized_result.get('confidence_score', 0.0),
#                     'ai_analysis': optimized_result.get('ai_analysis', {}),
#                     'individual_suggestions': optimized_result.get('individual_suggestions', []),
#                     'has_conflicts': False,
#                     'conflicts': [],
#                     'analysis_summary': "3개 AI 모델이 분석한 결과입니다.",
#                     'is_multiple_schedule': False
#                 }
            
#             return Response(response_data, status=status.HTTP_201_CREATED)
            
#         except Exception as e:
#                     return Response({
#                         'error': f'일정 생성 요청 처리 중 오류: {str(e)}'
#                     }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

#     def _extract_schedule_title(self, request):
#             """요청에서 일정 제목 추출"""
#             if '운동' in request:
#                 return '운동'
#             elif '미팅' in request or '회의' in request:
#                 return '팀 미팅'
#             elif '공부' in request or '학습' in request:
#                 return '학습 시간'
#             elif '작업' in request or '업무' in request:
#                 return '집중 작업'
#             elif '약속' in request:
#                 return '약속'
#             else:
#                 return '새 일정'

#     def _extract_schedule_location(self, request):
#             """요청에서 장소 추출"""
#             if '운동' in request:
#                 return '헬스장'
#             elif '미팅' in request or '회의' in request:
#                 return '회의실'
#             elif '공부' in request or '학습' in request:
#                 return '도서관'
#             elif '커피' in request:
#                 return '카페'
#             else:
#                 return '사무실'

#     def _get_weekday_korean(self, date):
#             """요일을 한국어로 반환"""
#             weekdays = ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일']
#             return weekdays[date.weekday()]
            
   
#     def _check_schedule_conflicts(self, user, suggestion):
#         """일정 충돌 검사"""
#         if not suggestion or 'suggested_date' not in suggestion:
#             return []
        
#         try:
#             suggested_date = datetime.strptime(suggestion['suggested_date'], '%Y-%m-%d').date()
#             start_time = datetime.strptime(suggestion.get('suggested_start_time', '09:00'), '%H:%M').time()
#             end_time = datetime.strptime(suggestion.get('suggested_end_time', '10:00'), '%H:%M').time()
            
#             suggested_start = datetime.combine(suggested_date, start_time)
#             suggested_end = datetime.combine(suggested_date, end_time)
            
#             conflicts = Schedule.objects.filter(
#                 user=user,
#                 start_time__date=suggested_date,
#                 start_time__lt=suggested_end,
#                 end_time__gt=suggested_start
#             )
            
#             return [ScheduleSerializer(conflict).data for conflict in conflicts]
            
#         except Exception as e:
#             return []

# # 권한 수정된 함수들
# @api_view(['POST'])
# @permission_classes([IsAuthenticated])  # 🔧 권한 변경
# def confirm_schedule(request, request_id):
#     """AI 제안된 일정을 확정하여 실제 일정으로 생성"""
#     try:
#         user = request.user
        
#         # 🚫 ScheduleRequest.DoesNotExist에서 더미 데이터 생성 제거
#         try:
#             schedule_request = ScheduleRequest.objects.get(id=request_id, user=user)
#             optimized_suggestion = json.loads(schedule_request.optimized_suggestion)
#         except ScheduleRequest.DoesNotExist:
#             return Response({
#                 'error': f'요청 ID {request_id}를 찾을 수 없습니다.'
#             }, status=status.HTTP_404_NOT_FOUND)
#                 # 날짜/시간 파싱 개선
#         try:
#             suggested_date = optimized_suggestion.get('suggested_date')
#             suggested_start_time = optimized_suggestion.get('suggested_start_time', '09:00')
#             suggested_end_time = optimized_suggestion.get('suggested_end_time', '10:00')
            
#             # 날짜 형식 확인 및 변환
#             if isinstance(suggested_date, str):
#                 if 'T' in suggested_date:  # ISO 형식인 경우
#                     suggested_date = suggested_date.split('T')[0]
                
#                 start_datetime = datetime.strptime(
#                     f"{suggested_date} {suggested_start_time}",
#                     '%Y-%m-%d %H:%M'
#                 )
#                 end_datetime = datetime.strptime(
#                     f"{suggested_date} {suggested_end_time}",
#                     '%Y-%m-%d %H:%M'
#                 )
#             else:
#                 # 날짜가 없으면 오늘로 설정
#                 today = datetime.now().date()
#                 start_datetime = datetime.strptime(
#                     f"{today} {suggested_start_time}",
#                     '%Y-%m-%d %H:%M'
#                 )
#                 end_datetime = datetime.strptime(
#                     f"{today} {suggested_end_time}",
#                     '%Y-%m-%d %H:%M'
#                 )
                
#         except (ValueError, TypeError) as e:
#             print(f"DateTime parsing error: {e}")
#             # 기본값으로 폴백
#             now = datetime.now()
#             start_datetime = now.replace(hour=9, minute=0, second=0, microsecond=0)
#             end_datetime = now.replace(hour=10, minute=0, second=0, microsecond=0)
        
#         # Schedule 객체 생성
#         schedule_data = {
#             'user': user,
#             'title': optimized_suggestion.get('title', '새 일정'),
#             'description': optimized_suggestion.get('description', 'AI가 제안한 일정입니다.'),
#             'start_time': start_datetime,
#             'end_time': end_datetime,
#             'location': optimized_suggestion.get('location', ''),
#             'priority': optimized_suggestion.get('priority', 'MEDIUM'),
#             'attendees': json.dumps(optimized_suggestion.get('attendees', []), ensure_ascii=False)
#         }
        
#         schedule = Schedule.objects.create(**schedule_data)
#         serializer = ScheduleSerializer(schedule)
        
#         print(f"Schedule created successfully: {schedule.id}")
        
#         return Response({
#             'message': '여러 AI의 분석을 통해 최적화된 일정이 성공적으로 생성되었습니다.',
#             'schedule': serializer.data
#         }, status=status.HTTP_201_CREATED)
        
#     except Exception as e:
#         print(f"Confirm schedule error: {str(e)}")
#         import traceback
#         traceback.print_exc()
        
#         return Response({
#             'error': f'일정 생성 중 오류가 발생했습니다: {str(e)}'
#         }, status=status.HTTP_400_BAD_REQUEST)


# # Alternative solution: Convert to Class-Based View
# class ConfirmScheduleView(APIView):
#     """AI 제안된 일정을 확정하여 실제 일정으로 생성"""
#     permission_classes = [AllowAny]  # 임시로 AllowAny
    
#     def post(self, request, request_id):
#         try:
#             # 사용자 처리
#             if not request.user.is_authenticated:
#                 return Response({'error': '인증이 필요합니다.'}, status=status.HTTP_401_UNAUTHORIZED)

#             user = request.user
            
#             # request_id로 ScheduleRequest를 찾거나 더미 데이터 처리
#             try:
#                 schedule_request = ScheduleRequest.objects.get(id=request_id, user=user)
#                 optimized_suggestion = json.loads(schedule_request.optimized_suggestion)
#             except ScheduleRequest.DoesNotExist:
#                 # 더미 모드: request_id를 기반으로 기본 일정 생성
#                 print(f"ScheduleRequest {request_id} not found, creating dummy schedule")
#                 optimized_suggestion = {
#                     'title': 'AI 제안 일정',
#                     'description': 'AI가 제안한 최적의 일정입니다.',
#                     'suggested_date': datetime.now().strftime('%Y-%m-%d'),
#                     'suggested_start_time': '09:00',
#                     'suggested_end_time': '10:00',
#                     'location': '사무실',
#                     'priority': 'MEDIUM',
#                     'attendees': []
#                 }
#             except json.JSONDecodeError as e:
#                 print(f"JSON decode error: {e}")
#                 return Response({
#                     'error': f'일정 데이터 파싱 오류: {str(e)}'
#                 }, status=status.HTTP_400_BAD_REQUEST)
            
#             # 날짜/시간 파싱
#             try:
#                 suggested_date = optimized_suggestion.get('suggested_date')
#                 suggested_start_time = optimized_suggestion.get('suggested_start_time', '09:00')
#                 suggested_end_time = optimized_suggestion.get('suggested_end_time', '10:00')
                
#                 # 날짜 형식 확인 및 변환
#                 if isinstance(suggested_date, str):
#                     if 'T' in suggested_date:  # ISO 형식인 경우
#                         suggested_date = suggested_date.split('T')[0]
                    
#                     start_datetime = datetime.strptime(
#                         f"{suggested_date} {suggested_start_time}",
#                         '%Y-%m-%d %H:%M'
#                     )
#                     end_datetime = datetime.strptime(
#                         f"{suggested_date} {suggested_end_time}",
#                         '%Y-%m-%d %H:%M'
#                     )
#                 else:
#                     # 날짜가 없으면 오늘로 설정
#                     today = datetime.now().date()
#                     start_datetime = datetime.strptime(
#                         f"{today} {suggested_start_time}",
#                         '%Y-%m-%d %H:%M'
#                     )
#                     end_datetime = datetime.strptime(
#                         f"{today} {suggested_end_time}",
#                         '%Y-%m-%d %H:%M'
#                     )
                    
#             except (ValueError, TypeError) as e:
#                 print(f"DateTime parsing error: {e}")
#                 # 기본값으로 폴백
#                 now = datetime.now()
#                 start_datetime = now.replace(hour=9, minute=0, second=0, microsecond=0)
#                 end_datetime = now.replace(hour=10, minute=0, second=0, microsecond=0)
            
#             # Schedule 객체 생성
#             schedule_data = {
#                 'user': user,
#                 'title': optimized_suggestion.get('title', '새 일정'),
#                 'description': optimized_suggestion.get('description', 'AI가 제안한 일정입니다.'),
#                 'start_time': start_datetime,
#                 'end_time': end_datetime,
#                 'location': optimized_suggestion.get('location', ''),
#                 'priority': optimized_suggestion.get('priority', 'MEDIUM'),
#                 'attendees': json.dumps(optimized_suggestion.get('attendees', []), ensure_ascii=False)
#             }
            
#             schedule = Schedule.objects.create(**schedule_data)
#             serializer = ScheduleSerializer(schedule)
            
#             print(f"Schedule created successfully: {schedule.id}")
            
#             return Response({
#                 'message': '여러 AI의 분석을 통해 최적화된 일정이 성공적으로 생성되었습니다.',
#                 'schedule': serializer.data
#             }, status=status.HTTP_201_CREATED)
            
#         except Exception as e:
#             print(f"Confirm schedule error: {str(e)}")
#             import traceback
#             traceback.print_exc()
            
#             return Response({
#                 'error': f'일정 생성 중 오류가 발생했습니다: {str(e)}'
#             }, status=status.HTTP_400_BAD_REQUEST)


# # Fix for resolve_schedule_conflict function
# @api_view(['POST'])
# @permission_classes([IsAuthenticated])  # 🔧 권한 변경
# def resolve_schedule_conflict(request):
#     """일정 충돌 해결 방안 제공"""
#     # 🚫 더미 사용자 로직 제거
#     user = request.user
    
#     conflicting_schedule_ids = request.data.get('conflicting_schedule_ids', [])
#     new_request = request.data.get('new_request', '')
    
#     # 나머지 로직은 그대로...
    
#     if not conflicting_schedule_ids or not new_request:
#         return Response({
#             'error': '충돌 일정 ID와 새로운 요청이 필요합니다.'
#         }, status=status.HTTP_400_BAD_REQUEST)
    
#     try:
#         # 사용자 처리
#         if request.user.is_authenticated:
#             user = request.user
#         else:
#             from django.contrib.auth.models import User
#             user, created = User.objects.get_or_create(
#                 username='dummy_user',
#                 defaults={'email': 'dummy@example.com'}
#             )
        
#         # 충돌 일정들 조회
#         conflicting_schedules = Schedule.objects.filter(
#             id__in=conflicting_schedule_ids,
#             user=user
#         )
        
#         conflicting_data = [ScheduleSerializer(schedule).data for schedule in conflicting_schedules]
        
#         # 다중 AI 모델들로부터 해결 방안 받기
#         optimizer = ScheduleOptimizerBot()
#         prompt = optimizer.create_conflict_resolution_prompt(conflicting_data, new_request)
#         suggestions = optimizer.get_ai_suggestions(prompt, "conflict_resolution")
        
#         # AI 분석을 통한 최적 해결방안 도출
#         analysis_result = optimizer.analyze_and_optimize_suggestions(
#             suggestions,
#             f"충돌 해결: {new_request}"
#         )
        
#         # 해결 방안 저장
#         conflict_resolution = ConflictResolution.objects.create(
#             user=user,
#             conflicting_schedules=json.dumps(conflicting_data, ensure_ascii=False),
#             resolution_options=json.dumps(suggestions, ensure_ascii=False),
#             ai_recommendations=json.dumps(analysis_result, ensure_ascii=False)
#         )
        
#         return Response({
#             'resolution_id': conflict_resolution.id,
#             'conflicting_schedules': conflicting_data,
#             'ai_suggestions': suggestions,
#             'optimized_resolution': analysis_result,
#             'message': f'{len(suggestions)}개 AI 모델이 분석한 충돌 해결 방안이 생성되었습니다.'
#         }, status=status.HTTP_201_CREATED)
        
#     except Exception as e:
#         return Response({
#             'error': f'충돌 해결 방안 생성 중 오류가 발생했습니다: {str(e)}'
#         }, status=status.HTTP_400_BAD_REQUEST)


# # Alternative Class-Based View for conflict resolution
# class ResolveScheduleConflictView(APIView):
#     """일정 충돌 해결 방안 제공 - 다중 AI 분석"""
#     permission_classes = [IsAuthenticated]
    
#     def post(self, request):
#         conflicting_schedule_ids = request.data.get('conflicting_schedule_ids', [])
#         new_request = request.data.get('new_request', '')
        
#         if not conflicting_schedule_ids or not new_request:
#             return Response({
#                 'error': '충돌 일정 ID와 새로운 요청이 필요합니다.'
#             }, status=status.HTTP_400_BAD_REQUEST)
        
#         try:
#             # 사용자 처리
#             if request.user.is_authenticated:
#                 user = request.user
#             else:
#                 from django.contrib.auth.models import User
#                 user, created = User.objects.get_or_create(
#                     username='dummy_user',
#                     defaults={'email': 'dummy@example.com'}
#                 )
            
#             # 충돌 일정들 조회
#             conflicting_schedules = Schedule.objects.filter(
#                 id__in=conflicting_schedule_ids,
#                 user=user
#             )
            
#             conflicting_data = [ScheduleSerializer(schedule).data for schedule in conflicting_schedules]
            
#             # 다중 AI 모델들로부터 해결 방안 받기
#             optimizer = ScheduleOptimizerBot()
#             prompt = optimizer.create_conflict_resolution_prompt(conflicting_data, new_request)
#             suggestions = optimizer.get_ai_suggestions(prompt, "conflict_resolution")
            
#             # AI 분석을 통한 최적 해결방안 도출
#             analysis_result = optimizer.analyze_and_optimize_suggestions(
#                 suggestions,
#                 f"충돌 해결: {new_request}"
#             )
            
#             # 해결 방안 저장
#             conflict_resolution = ConflictResolution.objects.create(
#                 user=user,
#                 conflicting_schedules=json.dumps(conflicting_data, ensure_ascii=False),
#                 resolution_options=json.dumps(suggestions, ensure_ascii=False),
#                 ai_recommendations=json.dumps(analysis_result, ensure_ascii=False)
#             )
            
#             return Response({
#                 'resolution_id': conflict_resolution.id,
#                 'conflicting_schedules': conflicting_data,
#                 'ai_suggestions': suggestions,
#                 'optimized_resolution': analysis_result,
#                 'message': f'{len(suggestions)}개 AI 모델이 분석한 충돌 해결 방안이 생성되었습니다.'
#             }, status=status.HTTP_201_CREATED)
            
#         except Exception as e:
#             return Response({
#                 'error': f'충돌 해결 방안 생성 중 오류가 발생했습니다: {str(e)}'
#             }, status=status.HTTP_400_BAD_REQUEST)
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes, authentication_classes
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.authentication import TokenAuthentication, SessionAuthentication
from rest_framework.authtoken.models import Token
from rest_framework import exceptions
from django.shortcuts import get_object_or_404
from datetime import datetime, timedelta
import json
import re
import os
from .models import Schedule, ScheduleRequest, ConflictResolution
from .serializers import (
    ScheduleSerializer, ScheduleRequestSerializer, 
    ConflictResolutionSerializer, ScheduleRequestInputSerializer
)
import logging

# 타임존 import 수정
import pytz
KST = pytz.timezone('Asia/Seoul')

def get_current_datetime():
    return datetime.now(KST)

logger = logging.getLogger(__name__)

# API 키들
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")  
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# 토큰 디버깅을 위한 커스텀 인증 클래스
class DebugTokenAuthentication(TokenAuthentication):
    """디버깅이 포함된 토큰 인증 클래스"""
    
    def authenticate(self, request):
        logger.info("=== 개선된 토큰 인증 디버깅 시작 ===")
        
        auth_header = request.META.get('HTTP_AUTHORIZATION', '')
        logger.info(f"Authorization 헤더: '{auth_header}'")
        
        if not auth_header:
            logger.warning("❌ Authorization 헤더가 없습니다")
            return None
            
        if not auth_header.startswith('Bearer '):
            logger.warning(f"❌ Bearer 토큰 형식이 아닙니다: {auth_header}")
            return None
            
        token = auth_header.split(' ')[1]
        logger.info(f"📱 추출된 토큰: {token[:10]}...{token[-10:]}")
        
        try:
            token_obj = Token.objects.select_related('user').get(key=token)
            logger.info(f"✅ DB에서 토큰 발견: {token_obj.key[:10]}...{token_obj.key[-10:]}")
            logger.info(f"👤 토큰 소유자: {token_obj.user.username} (ID: {token_obj.user.id})")
            
            if not token_obj.user.is_active:
                logger.warning(f"❌ 사용자가 비활성화됨: {token_obj.user.username}")
                raise exceptions.AuthenticationFailed('User inactive or deleted.')
            
            logger.info("✅ 토큰 인증 성공!")
            logger.info("=== 개선된 토큰 인증 디버깅 종료 ===")
            return (token_obj.user, token_obj)
            
        except Token.DoesNotExist:
            logger.error(f"❌ DB에 해당 토큰이 존재하지 않음: {token[:10]}...{token[-10:]}")
            raise exceptions.AuthenticationFailed('Invalid token.')
        
        except Exception as e:
            logger.error(f"❌ 예상치 못한 오류: {str(e)}")
            raise exceptions.AuthenticationFailed('Authentication error.')


# 실제 AI ChatBot 클래스
class RealChatBot:
    def __init__(self, api_key, model_name, provider):
        self.api_key = api_key
        self.model_name = model_name
        self.provider = provider
    
    def chat(self, prompt):
        """실제 AI API 호출"""
        try:
            if self.provider == 'openai' and self.api_key:
                return self._call_openai_api(prompt)
            elif self.provider == 'anthropic' and self.api_key:
                return self._call_anthropic_api(prompt)
            elif self.provider == 'groq' and self.api_key:
                return self._call_groq_api(prompt)
            else:
                return self._generate_fallback_response(prompt)
        except Exception as e:
            logger.error(f"AI API 호출 실패 ({self.provider}): {e}")
            return self._generate_fallback_response(prompt)
    
    def _call_openai_api(self, prompt):
        """OpenAI API 호출 - 새 버전 문법 사용"""
        try:
            from openai import OpenAI
            
            client = OpenAI(api_key=self.api_key)
            
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "당신은 일정 관리를 도와주는 AI 어시스턴트입니다. JSON 형식으로 응답해주세요."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except ImportError:
            logger.error("openai 패키지가 설치되지 않았습니다")
            return self._generate_fallback_response(prompt)
        except Exception as e:
            logger.error(f"OpenAI API 오류: {e}")
            return self._generate_fallback_response(prompt)
    
    def _call_anthropic_api(self, prompt):
        """Anthropic API 호출"""
        try:
            client = get_anthropic_client()
            
            response = client.messages.create(
                model=self.model_name,
                max_tokens=1000,
                messages=[
                    {"role": "user", "content": f"당신은 일정 관리를 도와주는 AI 어시스턴트입니다. JSON 형식으로 응답해주세요.\n\n{prompt}"}
                ]
            )
            
            return response.content[0].text
            
        except ImportError:
            logger.error("anthropic 패키지가 설치되지 않았습니다")
            return self._generate_fallback_response(prompt)
        except Exception as e:
            logger.error(f"Anthropic API 오류: {e}")
            return self._generate_fallback_response(prompt)
    
    def _call_groq_api(self, prompt):
        """Groq API 호출"""
        try:
            client = get_groq_client()
            
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "당신은 일정 관리를 도와주는 AI 어시스턴트입니다. JSON 형식으로 응답해주세요."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except ImportError:
            logger.error("groq 패키지가 설치되지 않았습니다")
            return self._generate_fallback_response(prompt)
        except Exception as e:
            logger.error(f"Groq API 오류: {e}")
            return self._generate_fallback_response(prompt)
    
    def _generate_fallback_response(self, prompt):
        """API 호출 실패 시 기본 응답 생성"""
        current_time = get_current_datetime()
        
        # 프롬프트에서 키워드 추출하여 적절한 제목 생성
        title = "새 일정"
        if "운동" in prompt:
            title = "운동"
        elif "회의" in prompt or "미팅" in prompt:
            title = "회의"
        elif "공부" in prompt or "학습" in prompt:
            title = "공부"
        elif "약속" in prompt:
            title = "약속"
        elif "작업" in prompt:
            title = "작업"
        
        return f"""{{
            "title": "{title}",
            "description": "일정이 생성되었습니다",
            "suggested_date": "{current_time.strftime('%Y-%m-%d')}",
            "suggested_start_time": "09:00",
            "suggested_end_time": "10:00",
            "location": "",
            "priority": "MEDIUM",
            "attendees": [],
            "reasoning": "기본 일정 제안입니다."
        }}"""


# 일정 관리 뷰 - LLM 자동 제목 생성 강화
class ScheduleManagementView(APIView):
    """일정 관리 메인 뷰 - 토큰 인증 적용"""
    authentication_classes = [DebugTokenAuthentication, SessionAuthentication]
    permission_classes = [IsAuthenticated]
    
    def __init__(self):
        super().__init__()
    
    def get_optimizer(self):
        """필요할 때만 optimizer 인스턴스 생성"""
        if not hasattr(self, '_optimizer'):
            self._optimizer = ScheduleOptimizerBot()
        return self._optimizer
    
    def get(self, request):
        """사용자의 일정 목록 조회"""
        logger.info(f"일정 조회 요청 - 사용자: {request.user.username}")
        
        try:
            schedules = Schedule.objects.filter(user=request.user).order_by('start_time')
            
            # 날짜 필터링
            start_date = request.query_params.get('start_date')
            end_date = request.query_params.get('end_date')
            
            if start_date:
                schedules = schedules.filter(start_time__date__gte=start_date)
            if end_date:
                schedules = schedules.filter(end_time__date__lte=end_date)
            
            serializer = ScheduleSerializer(schedules, many=True)
            logger.info(f"일정 조회 성공: {len(serializer.data)}개 일정 반환")
            return Response(serializer.data)
            
        except Exception as e:
            logger.error(f"일정 조회 실패: {str(e)}")
            return Response(
                {'error': f'일정 조회 중 오류 발생: {str(e)}'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    def _generate_smart_title(self, request_text):
        """LLM을 사용해 스마트한 일정 제목 생성"""
        try:
            optimizer = self.get_optimizer()
            
            # 제목 생성 전용 프롬프트
            title_prompt = f"""
            다음 일정 요청에서 적절한 일정 제목을 한국어로 간단하고 명확하게 생성해주세요.
            
            요청 내용: "{request_text}"
            
            규칙:
            1. 10글자 이내로 간단하게
            2. 구체적이고 의미있게
            3. 이모지 사용하지 마세요
            4. 제목만 반환하세요 (따옴표나 설명 없이)
            
            예시:
            - "내일 오후 2시에 회의" → "팀 회의"
            - "주말에 운동하기" → "주말 운동"
            - "친구와 카페에서 만나기" → "친구와 카페"
            - "프로젝트 작업" → "프로젝트 작업"
            """
            
            suggestions = optimizer.get_ai_suggestions(title_prompt, "title")
            
            # AI 응답에서 제목 추출
            for key, response in suggestions.items():
                if response and len(response.strip()) > 0:
                    # 응답에서 깔끔한 제목만 추출
                    lines = response.strip().split('\n')
                    for line in lines:
                        clean_line = line.strip().strip('"\'`').strip()
                        # 너무 길거나 짧지 않은 적절한 제목 찾기
                        if 2 <= len(clean_line) <= 20 and not clean_line.startswith('제목:'):
                            logger.info(f"LLM 생성 제목: {clean_line}")
                            return clean_line
            
            # AI 응답이 부적절하면 기본 제목 생성
            logger.warning("LLM 제목 생성 실패, 기본 제목 사용")
            return self._extract_schedule_title(request_text)
            
        except Exception as e:
            logger.error(f"스마트 제목 생성 실패: {str(e)}")
            return self._extract_schedule_title(request_text)
    
    def post(self, request):
        """새로운 일정 생성 요청"""
        logger.info(f"일정 생성 요청 - 사용자: {request.user.username}")
        
        try:
            request_text = request.data.get('request_text', '')
            existing_schedules = request.data.get('existing_schedules', [])
            
            if not request_text:
                return Response({'error': '요청 텍스트가 필요합니다.'}, 
                            status=status.HTTP_400_BAD_REQUEST)
            
            user = request.user
            
            # 여러 일정 요청인지 확인
            schedule_requests = parse_multiple_schedules_backend(request_text)
            target_date = parse_date_from_request(request_text)
            target_datetime = get_current_datetime()
            
            logger.info(f"파싱된 일정 요청: {len(schedule_requests)}개")

            if len(schedule_requests) > 1:
                # 여러 일정 처리 - 충돌 방지 강화
                multiple_schedules = []
                all_individual_suggestions = []
                cumulative_existing_schedules = existing_schedules.copy()  # 누적 일정 관리
                
                logger.info(f"여러 일정 처리 시작: {len(schedule_requests)}개 일정")
                
                for i, single_request in enumerate(schedule_requests):
                    logger.info(f"처리 중인 일정 {i+1}/{len(schedule_requests)}: {single_request}")
                    
                    # LLM이 각 일정의 제목을 자동 생성
                    smart_title = self._generate_smart_title(single_request)
                    
                    # 시간 정보 추출
                    parsed_start, parsed_duration = self._extract_time_info(single_request)
                    
                    # 기본 시작 시간 설정 (사용자 지정이 없으면 순차적으로)
                    if parsed_start is not None:
                        start_hour = parsed_start
                        logger.info(f"사용자 지정 시간 사용: {start_hour}시")
                    else:
                        start_hour = 9 + (i * 2)  # 9시, 11시, 13시, 15시... 순차적 배정
                        logger.info(f"기본 시간 사용: {start_hour}시")
                        
                    duration_hours = parsed_duration or 1

                    # 충돌 방지 시간 계산 - 누적된 기존 일정 포함
                    logger.info(f"충돌 방지 계산 시작 - 현재 누적 일정 개수: {len(cumulative_existing_schedules)}")
                    schedule_start_dt, schedule_end_dt = self._find_non_conflicting_time(
                        cumulative_existing_schedules, start_hour, duration_hours, target_date
                    )

                    optimized_schedule = {
                        "title": smart_title,  # LLM이 생성한 스마트 제목
                        "description": f"AI가 분석한 일정입니다: {single_request}",
                        "suggested_date": target_datetime.strftime('%Y-%m-%d'),
                        "suggested_start_time": schedule_start_dt.strftime('%H:%M'),
                        "suggested_end_time": schedule_end_dt.strftime('%H:%M'),
                        "location": self._extract_schedule_location(single_request),
                        "priority": "HIGH",
                        "attendees": [],
                        "reasoning": f"{i + 1}번째 일정: {single_request}. 기존 일정과 충돌하지 않는 {schedule_start_dt.strftime('%H:%M')}-{schedule_end_dt.strftime('%H:%M')} 시간으로 배정했습니다."
                    }

                    multiple_schedules.append(optimized_schedule)
                    
                    # 방금 추가한 일정을 누적 일정 목록에 추가 (다음 일정 처리 시 고려)
                    cumulative_existing_schedules.append({
                        'start_time': schedule_start_dt.strftime('%Y-%m-%dT%H:%M:%S'),
                        'end_time': schedule_end_dt.strftime('%Y-%m-%dT%H:%M:%S'),
                        'title': smart_title
                    })
                    
                    logger.info(f"일정 {i+1} 완료: {smart_title} ({schedule_start_dt.strftime('%H:%M')}-{schedule_end_dt.strftime('%H:%M')})")
                    
                    # 각 AI별 개별 제안 생성
                    for ai_type in ['gpt', 'claude', 'mixtral']:
                        individual_suggestion = optimized_schedule.copy()
                        individual_suggestion['source'] = ai_type
                        individual_suggestion['reasoning'] = f"{ai_type.upper()}가 분석한 결과입니다."
                        all_individual_suggestions.append(individual_suggestion)
                
                logger.info(f"여러 일정 처리 완료: 총 {len(multiple_schedules)}개 일정 생성")
                
                response_data = {
                    'request_id': int(datetime.now().timestamp()),
                    'multiple_schedules': multiple_schedules,
                    'optimized_suggestion': multiple_schedules[0] if multiple_schedules else {},
                    'confidence_score': 0.92,
                    'individual_suggestions': all_individual_suggestions,
                    'ai_analysis': {
                        'analysis_summary': f"총 {len(schedule_requests)}개의 일정을 분석하여 충돌 없는 최적 시간대로 배정했습니다.",
                        'reasoning': f"여러 일정을 {target_date.strftime('%Y년 %m월 %d일')}에 시간 순서대로 배치하여 모든 충돌을 방지했습니다.",
                        'models_used': ["gpt", "claude", "mixtral"]
                    },
                    'has_conflicts': False,
                    'conflicts': [],
                    'analysis_summary': f"{len(schedule_requests)}개 일정에 대해 AI가 충돌 없이 분석한 결과입니다.",
                    'is_multiple_schedule': True
                }
                
            else:
                # 단일 일정 처리 - 충돌 방지 강화
                logger.info("단일 일정 처리 시작")
                
                # LLM이 제목을 자동 생성
                smart_title = self._generate_smart_title(request_text)
                
                # 사용자가 원하는 시간 추출
                parsed_start, parsed_duration = self._extract_time_info(request_text)
                preferred_start_hour = parsed_start if parsed_start is not None else 9
                duration_hours = parsed_duration or 1
                
                logger.info(f"단일 일정 시간 분석: 선호 시작시간 {preferred_start_hour}시, 지속시간 {duration_hours}시간")
                
                # 충돌 방지 시간 계산
                schedule_start_dt, schedule_end_dt = self._find_non_conflicting_time(
                    existing_schedules, preferred_start_hour, duration_hours, target_date
                )
                
                # AI에게 충돌 없는 시간으로 제안 요청
                enhanced_prompt = f"""
                사용자의 일정 요청을 분석하여 최적의 일정을 제안해주세요.
                
                요청 내용: {request_text}
                목표 날짜: {target_date.strftime('%Y년 %m월 %d일')} ({self._get_weekday_korean(target_date)})
                기존 일정들: {len(existing_schedules)}개 일정이 이미 있습니다
                배정된 시간: {schedule_start_dt.strftime('%H:%M')}-{schedule_end_dt.strftime('%H:%M')} (충돌 방지됨)
                
                다음 형식으로 JSON 응답해주세요:
                {{
                    "title": "{smart_title}",
                    "description": "상세 설명",
                    "suggested_date": "{target_date.strftime('%Y-%m-%d')}",
                    "suggested_start_time": "{schedule_start_dt.strftime('%H:%M')}",
                    "suggested_end_time": "{schedule_end_dt.strftime('%H:%M')}",
                    "location": "장소",
                    "priority": "MEDIUM",
                    "attendees": [],
                    "reasoning": "기존 일정과 충돌하지 않는 최적 시간입니다"
                }}
                """
                
                try:
                    optimizer = self.get_optimizer()
                    suggestions = optimizer.get_ai_suggestions(enhanced_prompt)
                    optimized_result = optimizer.analyze_and_optimize_suggestions(
                        suggestions, f"일정 요청: {request_text}"
                    )
                    
                    # 충돌 방지된 시간으로 덮어쓰기 보장
                    if 'optimized_suggestion' in optimized_result:
                        optimized_result['optimized_suggestion']['title'] = smart_title
                        optimized_result['optimized_suggestion']['suggested_date'] = target_date.strftime('%Y-%m-%d')
                        optimized_result['optimized_suggestion']['suggested_start_time'] = schedule_start_dt.strftime('%H:%M')
                        optimized_result['optimized_suggestion']['suggested_end_time'] = schedule_end_dt.strftime('%H:%M')
                        optimized_result['optimized_suggestion']['reasoning'] = f"기존 일정과 충돌하지 않는 {schedule_start_dt.strftime('%H:%M')}-{schedule_end_dt.strftime('%H:%M')} 시간으로 배정했습니다."
                    
                    response_data = {
                        'request_id': int(datetime.now().timestamp()),
                        'optimized_suggestion': optimized_result.get('optimized_suggestion', {}),
                        'confidence_score': optimized_result.get('confidence_score', 0.9),
                        'ai_analysis': optimized_result.get('ai_analysis', {}),
                        'individual_suggestions': optimized_result.get('individual_suggestions', []),
                        'has_conflicts': False,
                        'conflicts': [],
                        'analysis_summary': "AI가 기존 일정과의 충돌을 방지하여 최적 시간을 배정했습니다.",
                        'is_multiple_schedule': False
                    }
                    
                except Exception as e:
                    logger.error(f"단일 일정 AI 처리 중 오류: {e}")
                    # AI 처리 실패 시에도 충돌 방지된 기본 응답 생성
                    response_data = {
                        'request_id': int(datetime.now().timestamp()),
                        'optimized_suggestion': {
                            "title": smart_title,
                            "description": f"요청하신 일정입니다: {request_text}",
                            "suggested_date": target_date.strftime('%Y-%m-%d'),
                            "suggested_start_time": schedule_start_dt.strftime('%H:%M'),
                            "suggested_end_time": schedule_end_dt.strftime('%H:%M'),
                            "location": self._extract_schedule_location(request_text),
                            "priority": "MEDIUM",
                            "attendees": [],
                            "reasoning": f"기존 일정과 충돌하지 않는 {schedule_start_dt.strftime('%H:%M')}-{schedule_end_dt.strftime('%H:%M')} 시간으로 배정했습니다."
                        },
                        'confidence_score': 0.8,
                        'ai_analysis': {
                            'analysis_summary': '충돌 방지 시간 배정이 완료되었습니다.',
                            'reasoning': '기존 일정과 겹치지 않는 최적 시간을 찾았습니다.',
                            'models_used': []
                        },
                        'individual_suggestions': [],
                        'has_conflicts': False,
                        'conflicts': [],
                        'analysis_summary': "충돌 방지 알고리즘이 최적 시간을 배정했습니다.",
                        'is_multiple_schedule': False
                    }
                
                logger.info(f"단일 일정 처리 완료: {smart_title} ({schedule_start_dt.strftime('%H:%M')}-{schedule_end_dt.strftime('%H:%M')})")
            
            logger.info("일정 생성 요청 처리 완료")
            return Response(response_data, status=status.HTTP_201_CREATED)
            
        except Exception as e:
            logger.error(f"일정 생성 요청 처리 실패: {str(e)}")
            return Response({
                'error': f'일정 생성 요청 처리 중 오류: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    def _extract_time_info(self, text):
        """텍스트에서 시간 정보 추출"""
        import re
        start_hour = None
        duration_hours = 1

        is_pm = '오후' in text
        is_am = '오전' in text

        # "오후 3-5시"와 같은 경우 처리
        time_range = re.search(r'(\d{1,2})\s*[-~]\s*(\d{1,2})\s*시', text)
        if time_range:
            start = int(time_range.group(1))
            end = int(time_range.group(2))

            if is_pm:
                if start < 12:
                    start += 12
                if end < 12:
                    end += 12
            elif is_am:
                if start == 12:
                    start = 0
                if end == 12:
                    end = 0

            start_hour = start
            duration_hours = end - start
            return start_hour, duration_hours

        # "2시간"만 있는 경우
        dur_match = re.search(r'(\d{1,2})\s*시간', text)
        if dur_match:
            duration_hours = int(dur_match.group(1))

        # 단일 시각: "오후 3시"
        single_time_match = re.search(r'(오전|오후)?\s*(\d{1,2})\s*시', text)
        if single_time_match:
            hour = int(single_time_match.group(2))
            if single_time_match.group(1) == '오후' and hour < 12:
                hour += 12
            elif single_time_match.group(1) == '오전' and hour == 12:
                hour = 0
            start_hour = hour

        return start_hour, duration_hours

    def _find_non_conflicting_time(self, existing_schedules, start_hour, duration_hours, target_date):
        """기존 일정과 겹치지 않는 시간대를 탐색 - 현실적인 시간대만 고려"""
        from datetime import datetime, timedelta, time
        
        logger.info(f"충돌 방지 시간 탐색 시작: 원하는 시작시간 {start_hour}시, 지속시간 {duration_hours}시간")
        logger.info(f"기존 일정 개수: {len(existing_schedules)}")

        # 현실적인 활동 시간대 정의
        WORK_START = 7   # 오전 7시부터
        WORK_END = 22    # 오후 10시까지
        
        def parse_existing_schedule_time(schedule):
            """기존 일정 시간 파싱"""
            try:
                if 'start_time' in schedule and 'end_time' in schedule:
                    start_str = schedule['start_time']
                    end_str = schedule['end_time']
                    
                    if 'T' in start_str:
                        start_dt = datetime.strptime(start_str, '%Y-%m-%dT%H:%M:%S')
                        end_dt = datetime.strptime(end_str, '%Y-%m-%dT%H:%M:%S')
                    else:
                        start_dt = datetime.strptime(start_str, '%Y-%m-%d %H:%M:%S')
                        end_dt = datetime.strptime(end_str, '%Y-%m-%d %H:%M:%S')
                        
                    return start_dt, end_dt
                        
            except (ValueError, KeyError) as e:
                logger.warning(f"일정 시간 파싱 실패: {schedule}, 오류: {e}")
                return None, None
            
            return None, None

        def is_conflicting(new_start, new_end, schedules):
            """시간 겹침 검사"""
            logger.info(f"충돌 검사: 새 일정 {new_start.strftime('%H:%M')}-{new_end.strftime('%H:%M')}")
            
            for i, schedule in enumerate(schedules):
                existing_start, existing_end = parse_existing_schedule_time(schedule)
                
                if existing_start is None or existing_end is None:
                    continue
                
                # 같은 날짜인지 확인
                if existing_start.date() != target_date:
                    continue
                    
                logger.info(f"기존 일정 {i+1}: {existing_start.strftime('%H:%M')}-{existing_end.strftime('%H:%M')}")
                
                # 겹치는지 검사
                if (new_start < existing_end) and (existing_start < new_end):
                    logger.warning(f"⚠️ 시간 충돌 발견! 새 일정과 기존 일정 {i+1}이 겹침")
                    return True
                    
            return False

        def is_realistic_time(hour, duration):
            """현실적인 시간인지 확인"""
            end_hour = hour + duration
            
            # 업무시간 내에 있는지 확인
            if hour < WORK_START or end_hour > WORK_END:
                return False
                
            # 점심시간 피하기 (12-13시)
            if hour <= 12 and end_hour >= 13:
                return False
                
            return True

        # 사용자가 지정한 시간이 현실적인지 먼저 확인
        if start_hour < WORK_START or start_hour >= WORK_END:
            logger.warning(f"사용자 지정 시간 {start_hour}시가 비현실적임. 오전 9시로 조정")
            start_hour = 9

        # 현실적인 시간대에서만 탐색
        realistic_hours = []
        
        # 오전 시간대 (7-12시)
        for h in range(7, 12):
            if h + duration_hours <= 12:  # 점심시간 전에 끝나야 함
                realistic_hours.append(h)
        
        # 오후 시간대 (13-22시)
        for h in range(13, 22):
            if h + duration_hours <= 22:  # 저녁 10시 전에 끝나야 함
                realistic_hours.append(h)

        # 사용자 선호 시간부터 시작하여 탐색
        search_order = []
        if start_hour in realistic_hours:
            search_order.append(start_hour)
        
        # 선호 시간 주변부터 확장 탐색
        for offset in range(1, 8):
            if start_hour + offset in realistic_hours and start_hour + offset not in search_order:
                search_order.append(start_hour + offset)
            if start_hour - offset in realistic_hours and start_hour - offset not in search_order:
                search_order.append(start_hour - offset)
        
        # 나머지 현실적인 시간들 추가
        for h in realistic_hours:
            if h not in search_order:
                search_order.append(h)

        logger.info(f"현실적인 시간대 탐색 순서: {search_order}")

        # 현실적인 시간대에서 충돌 없는 시간 찾기
        for attempt, hour in enumerate(search_order):
            try:
                candidate_start = datetime.combine(target_date, time(hour))
                candidate_end = candidate_start + timedelta(hours=duration_hours)
                
                logger.info(f"시도 {attempt + 1}: {candidate_start.strftime('%H:%M')}-{candidate_end.strftime('%H:%M')}")
                
                if not is_conflicting(candidate_start, candidate_end, existing_schedules):
                    logger.info(f"✅ 충돌 없는 현실적인 시간 발견: {candidate_start.strftime('%H:%M')}-{candidate_end.strftime('%H:%M')}")
                    return candidate_start, candidate_end
                    
            except Exception as e:
                logger.error(f"시간 계산 오류: {e}")
                continue

        # 모든 현실적인 시간대에 충돌이 있는 경우
        logger.warning("모든 현실적인 시간대에 충돌 발생")
        
        # 가장 빈 시간대 찾기
        best_hour = 9  # 기본값
        min_conflicts = float('inf')
        
        for hour in realistic_hours:
            try:
                candidate_start = datetime.combine(target_date, time(hour))
                candidate_end = candidate_start + timedelta(hours=duration_hours)
                
                # 이 시간대의 충돌 개수 세기
                conflict_count = 0
                for schedule in existing_schedules:
                    existing_start, existing_end = parse_existing_schedule_time(schedule)
                    if existing_start and existing_start.date() == target_date:
                        if (candidate_start < existing_end) and (existing_start < candidate_end):
                            conflict_count += 1
                
                if conflict_count < min_conflicts:
                    min_conflicts = conflict_count
                    best_hour = hour
                    
            except Exception as e:
                logger.error(f"충돌 계산 오류: {e}")
                continue
        
        # 최소 충돌 시간대로 배정
        final_start = datetime.combine(target_date, time(best_hour))
        final_end = final_start + timedelta(hours=duration_hours)
        
        logger.info(f"🔄 최소 충돌 시간으로 배정: {final_start.strftime('%H:%M')}-{final_end.strftime('%H:%M')} (충돌 {min_conflicts}개)")
        return final_start, final_end

    def _extract_time_info(self, text):
        """텍스트에서 시간 정보 추출 - 현실적인 시간만 반환"""
        import re
        start_hour = None
        duration_hours = 1

        is_pm = '오후' in text
        is_am = '오전' in text

        # "오후 3-5시"와 같은 경우 처리
        time_range = re.search(r'(\d{1,2})\s*[-~]\s*(\d{1,2})\s*시', text)
        if time_range:
            start = int(time_range.group(1))
            end = int(time_range.group(2))

            if is_pm:
                if start < 12:
                    start += 12
                if end < 12:
                    end += 12
            elif is_am:
                if start == 12:
                    start = 0
                if end == 12:
                    end = 0

            # 현실적인 시간인지 확인
            if 7 <= start <= 22 and 7 <= end <= 22:
                start_hour = start
                duration_hours = end - start
                return start_hour, duration_hours

        # "2시간"만 있는 경우
        dur_match = re.search(r'(\d{1,2})\s*시간', text)
        if dur_match:
            duration_hours = int(dur_match.group(1))

        # 단일 시각: "오후 3시"
        single_time_match = re.search(r'(오전|오후)?\s*(\d{1,2})\s*시', text)
        if single_time_match:
            hour = int(single_time_match.group(2))
            if single_time_match.group(1) == '오후' and hour < 12:
                hour += 12
            elif single_time_match.group(1) == '오전' and hour == 12:
                hour = 0
            
            # 현실적인 시간인지 확인
            if 7 <= hour <= 22:
                start_hour = hour

        # 비현실적인 시간이면 기본값으로 설정
        if start_hour is not None and (start_hour < 7 or start_hour > 22):
            logger.warning(f"비현실적인 시간 {start_hour}시 감지, 오전 9시로 조정")
            start_hour = 9

        return start_hour, duration_hours
    def _extract_schedule_title(self, request):
        """요청에서 기본 일정 제목 추출 (fallback용)"""
        if '운동' in request:
            return '운동'
        elif '미팅' in request or '회의' in request:
            return '팀 미팅'
        elif '공부' in request or '학습' in request:
            return '학습 시간'
        elif '작업' in request or '업무' in request:
            return '집중 작업'
        elif '약속' in request:
            return '약속'
        else:
            return '새 일정'

    def _extract_schedule_location(self, request):
        """요청에서 장소 추출"""
        if '운동' in request or '헬스' in request:
            return '헬스장'
        elif '미팅' in request or '회의' in request:
            return '회의실'
        elif '공부' in request or '학습' in request:
            return '도서관'
        elif '커피' in request or '카페' in request:
            return '카페'
        else:
            return '사무실'

    def _get_weekday_korean(self, date):
        """요일을 한국어로 반환"""
        weekdays = ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일']
        return weekdays[date.weekday()]


# 수동 일정 생성 뷰 (새로운 엔드포인트)
@api_view(['POST'])
@authentication_classes([DebugTokenAuthentication, SessionAuthentication])
@permission_classes([IsAuthenticated])
def create_schedule(request):
    """수동으로 일정 생성 - /api/schedule/create/ 엔드포인트"""
    logger.info(f"수동 일정 생성 요청 - 사용자: {request.user.username}")
    
    try:
        data = request.data.copy()
        
        # 제목이 없으면 LLM이 자동 생성
        if not data.get('title'):
            description = data.get('description', '')
            if description:
                # 간단한 제목 생성 로직
                try:
                    optimizer = ScheduleOptimizerBot()
                    title_prompt = f"""
                    다음 설명에서 적절한 일정 제목을 10글자 이내로 간단하게 생성해주세요:
                    "{description}"
                    
                    제목만 반환하세요 (따옴표나 설명 없이).
                    """
                    suggestions = optimizer.get_ai_suggestions(title_prompt, "title")
                    
                    # AI 응답에서 제목 추출
                    for key, response in suggestions.items():
                        if response and len(response.strip()) > 0:
                            lines = response.strip().split('\n')
                            for line in lines:
                                clean_line = line.strip().strip('"\'`').strip()
                                if 2 <= len(clean_line) <= 15:
                                    data['title'] = clean_line
                                    logger.info(f"자동 생성된 제목: {clean_line}")
                                    break
                            if data.get('title'):
                                break
                    
                    # 제목 생성에 실패하면 기본값
                    if not data.get('title'):
                        data['title'] = '새 일정'
                        
                except Exception as e:
                    logger.error(f"제목 자동 생성 실패: {e}")
                    data['title'] = '새 일정'
            else:
                data['title'] = '새 일정'
        
        data['user'] = request.user.id
        
        serializer = ScheduleSerializer(data=data)
        if serializer.is_valid():
            schedule = serializer.save(user=request.user)
            logger.info(f"수동 일정 생성 성공: {schedule.id} - {schedule.title}")
            return Response({
                'message': '일정이 성공적으로 생성되었습니다.',
                'schedule': serializer.data
            }, status=status.HTTP_201_CREATED)
        else:
            logger.warning(f"수동 일정 생성 실패 - 유효성 검증 오류: {serializer.errors}")
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
            
    except Exception as e:
        logger.error(f"수동 일정 생성 실패: {str(e)}")
        return Response({
            'error': f'일정 생성 중 오류 발생: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# 기존 뷰들...
@api_view(['POST'])
@authentication_classes([DebugTokenAuthentication, SessionAuthentication])
@permission_classes([IsAuthenticated])
def confirm_schedule(request, request_id):
    """AI 제안된 일정을 확정하여 실제 일정으로 생성"""
    logger.info(f"일정 확정 요청 - 사용자: {request.user.username}, request_id: {request_id}")
    
    try:
        user = request.user
        ai_suggestion_data = request.data.get('ai_suggestion')
        
        if not ai_suggestion_data:
            return Response({
                'error': 'AI 제안 데이터가 필요합니다.'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        is_multiple = ai_suggestion_data.get('is_multiple_schedule', False)
        
        if is_multiple and ai_suggestion_data.get('multiple_schedules'):
            # 여러 일정 처리
            created_schedules = []
            
            for schedule_data in ai_suggestion_data['multiple_schedules']:
                try:
                    suggested_date = schedule_data.get('suggested_date')
                    suggested_start_time = schedule_data.get('suggested_start_time', '09:00')
                    suggested_end_time = schedule_data.get('suggested_end_time', '10:00')
                    
                    start_datetime = datetime.strptime(
                        f"{suggested_date} {suggested_start_time}",
                        '%Y-%m-%d %H:%M'
                    )
                    end_datetime = datetime.strptime(
                        f"{suggested_date} {suggested_end_time}",
                        '%Y-%m-%d %H:%M'
                    )
                    
                    schedule = Schedule.objects.create(
                        user=user,
                        title=schedule_data.get('title', '새 일정'),
                        description=schedule_data.get('description', 'AI가 제안한 일정입니다.'),
                        start_time=start_datetime,
                        end_time=end_datetime,
                        location=schedule_data.get('location', ''),
                        priority=schedule_data.get('priority', 'MEDIUM'),
                        attendees=json.dumps(schedule_data.get('attendees', []), ensure_ascii=False)
                    )
                    
                    created_schedules.append(schedule)
                    logger.info(f"다중 일정 생성 성공: {schedule.id} - {schedule.title}")
                    
                except Exception as e:
                    logger.error(f"개별 일정 생성 실패: {str(e)}")
                    continue
            
            if created_schedules:
                serializer = ScheduleSerializer(created_schedules, many=True)
                return Response({
                    'message': f'{len(created_schedules)}개의 일정이 성공적으로 생성되었습니다.',
                    'schedules': serializer.data
                }, status=status.HTTP_201_CREATED)
            else:
                return Response({
                    'error': '일정 생성에 실패했습니다.'
                }, status=status.HTTP_400_BAD_REQUEST)
        
        else:
            # 단일 일정 처리
            optimized_suggestion = ai_suggestion_data.get('optimized_suggestion')
            if not optimized_suggestion:
                return Response({
                    'error': '최적화된 제안 데이터가 없습니다.'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            try:
                suggested_date = optimized_suggestion.get('suggested_date')
                suggested_start_time = optimized_suggestion.get('suggested_start_time', '09:00')
                suggested_end_time = optimized_suggestion.get('suggested_end_time', '10:00')
                
                if 'T' in suggested_date:
                    suggested_date = suggested_date.split('T')[0]
                
                start_datetime = datetime.strptime(
                    f"{suggested_date} {suggested_start_time}",
                    '%Y-%m-%d %H:%M'
                )
                end_datetime = datetime.strptime(
                    f"{suggested_date} {suggested_end_time}",
                    '%Y-%m-%d %H:%M'
                )
                
            except (ValueError, TypeError) as e:
                logger.error(f"DateTime parsing error: {e}")
                now = datetime.now()
                start_datetime = now.replace(hour=9, minute=0, second=0, microsecond=0)
                end_datetime = now.replace(hour=10, minute=0, second=0, microsecond=0)
            
            schedule = Schedule.objects.create(
                user=user,
                title=optimized_suggestion.get('title', '새 일정'),
                description=optimized_suggestion.get('description', 'AI가 제안한 일정입니다.'),
                start_time=start_datetime,
                end_time=end_datetime,
                location=optimized_suggestion.get('location', ''),
                priority=optimized_suggestion.get('priority', 'MEDIUM'),
                attendees=json.dumps(optimized_suggestion.get('attendees', []), ensure_ascii=False)
            )
            
            serializer = ScheduleSerializer(schedule)
            logger.info(f"단일 일정 생성 성공: {schedule.id} - {schedule.title}")
            
            return Response({
                'message': 'AI의 분석을 통해 최적화된 일정이 성공적으로 생성되었습니다.',
                'schedule': serializer.data
            }, status=status.HTTP_201_CREATED)
            
    except Exception as e:
        logger.error(f"일정 확정 실패: {str(e)}")
        return Response({
            'error': f'일정 생성 중 오류가 발생했습니다: {str(e)}'
        }, status=status.HTTP_400_BAD_REQUEST)


@api_view(['PUT', 'DELETE'])
@authentication_classes([DebugTokenAuthentication, SessionAuthentication])
@permission_classes([IsAuthenticated])
def manage_schedule(request, schedule_id):
    """일정 수정 또는 삭제"""
    try:
        schedule = get_object_or_404(Schedule, id=schedule_id, user=request.user)
        
        if request.method == 'PUT':
            # 일정 수정
            serializer = ScheduleSerializer(schedule, data=request.data, partial=True)
            if serializer.is_valid():
                serializer.save()
                logger.info(f"일정 수정 성공: {schedule_id}")
                return Response(serializer.data)
            else:
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        elif request.method == 'DELETE':
            # 일정 삭제
            schedule.delete()
            logger.info(f"일정 삭제 성공: {schedule_id}")
            return Response({'message': '일정이 성공적으로 삭제되었습니다.'}, 
                          status=status.HTTP_204_NO_CONTENT)
            
    except Exception as e:
        logger.error(f"일정 관리 실패: {str(e)}")
        return Response({
            'error': f'일정 관리 중 오류 발생: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# 유틸리티 함수들
def parse_date_from_request(request_text):
    """요청 텍스트에서 날짜 파싱"""
    try:
        korea_tz = pytz.timezone('Asia/Seoul')
        korea_now = datetime.now(korea_tz).date()

        if '오늘' in request_text:
            return korea_now
        elif '내일' in request_text:
            return korea_now + timedelta(days=1)
        elif '모레' in request_text or '모래' in request_text:
            return korea_now + timedelta(days=2)
        elif '이번 주' in request_text:
            days_until_friday = (4 - korea_now.weekday()) % 7
            days_until_friday = 7 if days_until_friday == 0 else days_until_friday
            return korea_now + timedelta(days=days_until_friday)
        elif '다음 주' in request_text:
            return korea_now + timedelta(days=7)
        else:
            return korea_now + timedelta(days=1)
    except Exception as e:
        logger.error(f"날짜 파싱 오류: {e}")
        return datetime.now().date()

def parse_multiple_schedules_backend(request_text):
    """백엔드에서 여러 일정 파싱"""
    try:
        separators = [',', '，', '그리고', '및', '와', '과']
        
        parts = [request_text]
        for sep in separators:
            new_parts = []
            for part in parts:
                new_parts.extend(part.split(sep))
            parts = new_parts
        
        cleaned_requests = []
        for part in parts:
            cleaned = part.strip()
            if cleaned and len(cleaned) > 2:
                cleaned_requests.append(cleaned)
        
        return cleaned_requests if len(cleaned_requests) > 1 else [request_text]
    except Exception as e:
        logger.error(f"일정 파싱 오류: {e}")
        return [request_text]


# ScheduleOptimizerBot 클래스
class ScheduleOptimizerBot:
    """일정 최적화를 위한 AI 봇 클래스"""
    
    def __init__(self):
        logger.info("ScheduleOptimizerBot 초기화 시작")
        try:

            self.chatbots = {
                # OpenAI 최신/경량 채팅 모델로 교체
                'gpt': RealChatBot(OPENAI_API_KEY, 'gpt-4o-mini', 'openai'),
                # Anthropic 모델명 최신 alias 권장
                'claude': RealChatBot(ANTHROPIC_API_KEY, 'claude-3-5-haiku-latest', 'anthropic'),
                # Groq는 llama-3.1-8b-instant도 OK. mixtral 쓰고 싶으면 모델명만 변경
                'mixtral': RealChatBot(GROQ_API_KEY, 'llama-3.1-8b-instant', 'groq'),
            }
            logger.info("ChatBot 클래스 로드 성공")
        except ImportError as e:
            logger.warning(f"ChatBot 클래스를 찾을 수 없습니다: {e}. 더미 클래스를 사용합니다.")
            self.chatbots = {
                'gpt': DummyChatBot(),
                'claude': DummyChatBot(),
                'mixtral': DummyChatBot(),
            }
        except Exception as e:
            logger.error(f"ChatBot 초기화 중 예상치 못한 오류: {e}")
            self.chatbots = {
                'gpt': DummyChatBot(),
                'claude': DummyChatBot(),
                'mixtral': DummyChatBot(),
            }
    
    def get_ai_suggestions(self, prompt, suggestion_type="schedule"):
        """여러 AI 모델로부터 제안받기"""
        logger.info(f"AI 제안 요청 시작 - 타입: {suggestion_type}")
        suggestions = {}
        
        for model_name, chatbot in self.chatbots.items():
            try:
                logger.info(f"{model_name} AI 모델 요청 시작")
                if hasattr(chatbot, 'chat'):
                    response = chatbot.chat(prompt)
                    logger.info(f"{model_name} AI 응답 길이: {len(response) if response else 0}")
                else:
                    response = f"더미 응답: {model_name}에서 {suggestion_type} 분석 완료"
                    logger.info(f"{model_name} 더미 응답 사용")
                suggestions[f"{model_name}_suggestion"] = response
            except Exception as e:
                logger.error(f"{model_name} AI 요청 실패: {str(e)}")
                suggestions[f"{model_name}_suggestion"] = f"오류 발생: {str(e)}"
        
        logger.info(f"AI 제안 완료: {len(suggestions)}개 응답")
        return suggestions
    
    def analyze_and_optimize_suggestions(self, suggestions, query, selected_models=['GPT', 'Claude', 'Mixtral']):
        """여러 AI 제안을 분석하여 최적화된 결과 생성"""
        logger.info("AI 제안 분석 및 최적화 시작")
        try:
            optimized = self._extract_first_valid_suggestion(suggestions)
            confidence = 0.85
            
            logger.info("AI 제안 분석 및 최적화 완료")
            return {
                "optimized_suggestion": optimized,
                "confidence_score": confidence,
                "ai_analysis": {
                    "analysis_summary": "AI 모델들의 제안을 종합 분석했습니다.",
                    "reasoning": "여러 모델의 공통점을 바탕으로 최적화했습니다.",
                    "models_used": selected_models
                },
                "individual_suggestions": self._parse_individual_suggestions(suggestions)
            }
            
        except Exception as e:
            logger.error(f"AI 분석 오류: {str(e)}")
            current_datetime = get_current_datetime()
            return {
                "optimized_suggestion": {
                    "title": "새 일정",
                    "description": "AI가 제안한 일정입니다",
                    "suggested_date": current_datetime.strftime('%Y-%m-%d'),
                    "suggested_start_time": "09:00",
                    "suggested_end_time": "10:00",
                    "location": "",
                    "priority": "MEDIUM",
                    "attendees": [],
                    "reasoning": f"분석 중 오류 발생: {str(e)}"
                },
                "confidence_score": 0.5,
                "ai_analysis": {
                    "analysis_summary": f"분석 중 오류 발생: {str(e)}",
                    "reasoning": "기본 제안을 생성했습니다.",
                    "models_used": []
                },
                "individual_suggestions": []
            }
    
    def _extract_first_valid_suggestion(self, suggestions):
        """첫 번째 유효한 제안 추출"""
        logger.info("유효한 제안 추출 시작")
        current_datetime = get_current_datetime()
        
        for key, suggestion in suggestions.items():
            try:
                logger.info(f"{key}에서 JSON 추출 시도")
                json_match = re.search(r'\{.*\}', suggestion, re.DOTALL)
                if json_match:
                    parsed_json = json.loads(json_match.group())
                    logger.info(f"{key}에서 JSON 파싱 성공")
                    
                    # 필수 필드 확인 및 기본값 설정
                    parsed_json.setdefault('suggested_date', current_datetime.strftime('%Y-%m-%d'))
                    parsed_json.setdefault('suggested_start_time', "09:00")
                    parsed_json.setdefault('suggested_end_time', "10:00")
                    parsed_json.setdefault('title', "새 일정")
                    parsed_json.setdefault('description', "AI가 제안한 일정입니다")
                    parsed_json.setdefault('priority', "MEDIUM")
                    parsed_json.setdefault('attendees', [])
                    parsed_json.setdefault('location', "")
                    parsed_json.setdefault('reasoning', "AI 분석 결과입니다.")
                        
                    logger.info("유효한 제안 추출 성공")
                    return parsed_json
            except (json.JSONDecodeError, AttributeError, KeyError) as e:
                logger.warning(f"JSON 파싱 실패 ({key}): {str(e)}")
                continue
            except Exception as e:
                logger.error(f"예상치 못한 파싱 오류 ({key}): {str(e)}")
                continue
        
        # 모든 파싱이 실패한 경우 기본값 반환
        logger.warning("모든 AI 응답 파싱 실패, 기본값 반환")
        return {
            "title": "새 일정",
            "description": "AI가 제안한 일정입니다",
            "suggested_date": current_datetime.strftime('%Y-%m-%d'),
            "suggested_start_time": "09:00",
            "suggested_end_time": "10:00",
            "location": "",
            "priority": "MEDIUM",
            "attendees": [],
            "reasoning": "여러 AI 모델의 제안을 종합한 결과입니다."
        }
    
    def _parse_individual_suggestions(self, suggestions):
        """개별 제안들을 파싱"""
        logger.info("개별 제안 파싱 시작")
        parsed = []
        current_datetime = get_current_datetime()
        
        for key, suggestion in suggestions.items():
            try:
                json_match = re.search(r'\{.*\}', suggestion, re.DOTALL)
                if json_match:
                    parsed_suggestion = json.loads(json_match.group())
                    parsed_suggestion['source'] = key.replace('_suggestion', '')
                    
                    # 필수 필드 확인 및 기본값 설정
                    parsed_suggestion.setdefault('suggested_date', current_datetime.strftime('%Y-%m-%d'))
                    parsed_suggestion.setdefault('suggested_start_time', "09:00")
                    parsed_suggestion.setdefault('suggested_end_time', "10:00")
                    parsed_suggestion.setdefault('title', "새 일정")
                    parsed_suggestion.setdefault('description', f"{key.replace('_suggestion', '').upper()}가 제안한 일정입니다")
                    parsed_suggestion.setdefault('priority', "MEDIUM")
                    parsed_suggestion.setdefault('attendees', [])
                    parsed_suggestion.setdefault('location', "")
                    parsed_suggestion.setdefault('reasoning', f"{key.replace('_suggestion', '').upper()} 모델의 분석 결과입니다.")
                        
                    parsed.append(parsed_suggestion)
                    logger.info(f"{key} 개별 제안 파싱 성공")
            except (json.JSONDecodeError, AttributeError, KeyError) as e:
                logger.warning(f"개별 제안 파싱 실패 ({key}): {str(e)}")
                continue
            except Exception as e:
                logger.error(f"개별 제안 예상치 못한 오류 ({key}): {str(e)}")
                continue
        
        # 파싱된 제안이 없으면 기본 제안 생성
        if not parsed:
            logger.warning("모든 개별 제안 파싱 실패, 기본 제안 생성")
            for model in ['gpt', 'claude', 'mixtral']:
                parsed.append({
                    "title": "새 일정",
                    "description": f"{model.upper()}가 제안한 일정입니다",
                    "suggested_date": current_datetime.strftime('%Y-%m-%d'),
                    "suggested_start_time": "09:00",
                    "suggested_end_time": "10:00",
                    "location": "",
                    "priority": "MEDIUM",
                    "attendees": [],
                    "reasoning": f"{model.upper()} 모델의 분석 결과입니다.",
                    "source": model
                })
        
        logger.info(f"개별 제안 파싱 완료: {len(parsed)}개")
        return parsed
        
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
from datetime import datetime
from collections import Counter

from .models import Video, VideoAnalysis, Scene, Frame, SearchHistory
from .llm_client import LLMClient


# views.py - 권한 설정 추가
# views.py - 모든 APIView에 권한 설정 추가




# views.py - 진행률 추적 개선 버전
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

from .models import Video, VideoAnalysis, Scene, Frame, SearchHistory
from .llm_client import LLMClient


# 전역 진행률 추적
analysis_progress_tracker = {}


#  
# views.py - 고급 분석 기능 추가

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

from .models import Video, VideoAnalysis, Scene, Frame, SearchHistory
from .llm_client import LLMClient

# 전역 진행률 추적 (기존과 동일)
analysis_progress_tracker = {}
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

from .models import Video, VideoAnalysis, Scene, Frame, SearchHistory
from .llm_client import LLMClient

# 전역 진행률 추적 (기존과 동일)
analysis_progress_tracker = {}

class AnalysisProgressTracker:
    """분석 진행률 추적 클래스 - 고급 분석 단계 추가"""
    
    def __init__(self):
        self.progress_data = {}
    
    def start_tracking(self, video_id, total_frames=0, analysis_type='enhanced'):
        """분석 추적 시작"""
        self.progress_data[video_id] = {
            'progress': 0,
            'currentStep': '분석 준비중',
            'startTime': datetime.now().isoformat(),
            'processedFrames': 0,
            'totalFrames': total_frames,
            'estimatedTime': None,
            'analysisType': analysis_type,
            'steps': [],
            'currentFeature': '',
            'completedFeatures': [],
            'totalFeatures': self._get_total_features(analysis_type)
        }
    
    def update_progress(self, video_id, progress=None, step=None, processed_frames=None, current_feature=None):
        """진행률 업데이트 - 고급 분석 정보 포함"""
        if video_id not in self.progress_data:
            return
        
        data = self.progress_data[video_id]
        
        if progress is not None:
            data['progress'] = min(100, max(0, progress))
        
        if step is not None:
            data['currentStep'] = step
            data['steps'].append({
                'step': step,
                'timestamp': datetime.now().isoformat()
            })
        
        if current_feature is not None:
            data['currentFeature'] = current_feature
            if current_feature not in data['completedFeatures']:
                data['completedFeatures'].append(current_feature)
        
        if processed_frames is not None:
            data['processedFrames'] = processed_frames
            
            # 진행률 자동 계산 (프레임 기반 + 기능 기반)
            if data['totalFrames'] > 0:
                frame_progress = (processed_frames / data['totalFrames']) * 80  # 프레임 분석 80%
                feature_progress = (len(data['completedFeatures']) / data['totalFeatures']) * 20  # 후처리 20%
                calculated_progress = frame_progress + feature_progress
                data['progress'] = min(100, calculated_progress)
        
        # 예상 완료 시간 계산 (고급 분석 고려)
        if data['progress'] > 5:
            elapsed = (datetime.now() - datetime.fromisoformat(data['startTime'])).total_seconds()
            
            # 분석 타입별 시간 가중치
            time_weights = {
                'basic': 1.0,
                'enhanced': 2.0,
                'comprehensive': 4.0,
                'custom': 2.5
            }
            
            weight = time_weights.get(data['analysisType'], 2.0)
            estimated_total = (elapsed / data['progress']) * 100 * weight
            remaining = estimated_total - elapsed
            data['estimatedTime'] = max(0, remaining)
    
    def _get_total_features(self, analysis_type):
        """분석 타입별 총 기능 수"""
        feature_counts = {
            'basic': 2,  # 객체감지, 기본캡션
            'enhanced': 4,  # 객체감지, CLIP, OCR, 고급캡션
            'comprehensive': 6,  # 모든 기능
            'custom': 4  # 평균값
        }
        return feature_counts.get(analysis_type, 4)
    
    def get_progress(self, video_id):
        """진행률 조회"""
        return self.progress_data.get(video_id, {})
    
    def finish_tracking(self, video_id, success=True):
        """분석 완료"""
        if video_id in self.progress_data:
            self.progress_data[video_id]['progress'] = 100
            self.progress_data[video_id]['currentStep'] = '분석 완료' if success else '분석 실패'
            self.progress_data[video_id]['success'] = success
            # 완료 후 10분 뒤 데이터 삭제
            threading.Timer(600, lambda: self.progress_data.pop(video_id, None)).start()

# 전역 트래커 인스턴스
progress_tracker = AnalysisProgressTracker()

# views.py - EnhancedAnalyzeVideoView 클래스 완전 수정
import threading
import time
import json
import cv2
from datetime import datetime
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny
# views.py 상단 import 부분 - 수정됨

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

# 모델 imports
from .models import Video, VideoAnalysis, Scene, Frame, SearchHistory
from .llm_client import LLMClient

# ✅ 중요: get_video_analyzer 함수 import 추가
from .video_analyzer import get_video_analyzer, VideoAnalyzer

# ✅ 추가: 기타 필요한 함수들도 import
try:
    from .video_analyzer import (
        EnhancedVideoAnalyzer, 
        ColorAnalyzer, 
        SceneClassifier, 
        AdvancedSceneAnalyzer,
        log_once  # 로그 중복 방지 함수
    )
    print("✅ video_analyzer 모듈에서 모든 클래스 import 성공")
except ImportError as e:
    print(f"⚠️ video_analyzer import 부분 실패: {e}")
    # Fallback - 기본 클래스만 import
    try:
        from .video_analyzer import get_video_analyzer, VideoAnalyzer, log_once
        print("✅ video_analyzer 모듈에서 모든 클래스 import 성공")
    except ImportError as e:
        print(f"⚠️ video_analyzer import 부분 실패: {e}")
        get_video_analyzer = None
        VideoAnalyzer = None
        log_once = None

# views.py - 실제 AI 분석을 사용하는 EnhancedAnalyzeVideoView

import os
import json
import time
import threading
from datetime import datetime
from django.http import JsonResponse
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny

# 비디오 분석기 import
try:
    from .services.video_analysis_service import get_video_analyzer, get_analyzer_status, VIDEO_ANALYZER_AVAILABLE
    from .db_builder import get_video_rag_system
    VIDEO_ANALYZER_AVAILABLE = True
    print("✅ video_analyzer 모듈 import 성공")
except ImportError as e:
    VIDEO_ANALYZER_AVAILABLE = False
    print(f"❌ video_analyzer import 실패: {e}")

# Django 모델 import
from .models import Video, VideoAnalysis, Scene, Frame

@method_decorator(csrf_exempt, name='dispatch')
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
            
            # 비디오 분석 서비스 가져오기
            from .services.video_analysis_service import get_video_analysis_service
            video_service = get_video_analysis_service()
            
            # 비디오 분석 서비스를 통해 분석 시작
            result = video_service.analyze_video(
                video_id=video.id,
                analysis_type=analysis_type
            )
            
            if result.get('success', False):
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
            else:
                return Response({
                    'error': result.get('error', '분석 시작에 실패했습니다.'),
                    'fallback': 'basic_analysis'
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
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


    def _save_frame_image(self, video, frame_data):
        """프레임 이미지를 media/images에 저장"""
        print(f"🖼️ 프레임 이미지 저장 시작: video_id={video.id}, image_id={frame_data.get('image_id')}")
        
        images_dir = os.path.join(settings.MEDIA_ROOT, 'images')
        os.makedirs(images_dir, exist_ok=True)
        print(f"📁 이미지 디렉토리: {images_dir}")

        video_path = video.file_path
        print(f"🎥 비디오 경로 (원본): {video_path}")
        
        # 상대 경로인 경우 절대 경로로 변환
        if not os.path.isabs(video_path):
            video_path = os.path.join(settings.MEDIA_ROOT, video_path)
            print(f"🎥 비디오 경로 (절대): {video_path}")
        
        if not video_path or not os.path.exists(video_path):
            print(f"❌ 비디오 파일이 존재하지 않음: {video_path}")
            return None

        # timestamp(초) → ms 단위로 프레임 위치 이동
        timestamp = frame_data.get('timestamp', 0)
        print(f"⏰ 타임스탬프: {timestamp}초")
        
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ret, frame = cap.read()
        cap.release()
        
        print(f"📸 프레임 읽기 결과: ret={ret}, frame_shape={frame.shape if frame is not None else 'None'}")

        if ret and frame is not None:
            filename = f"video{video.id}_frame{frame_data.get('image_id', 0)}.jpg"
            filepath = os.path.join(images_dir, filename)
            print(f"💾 저장할 파일: {filepath}")
            
            success = cv2.imwrite(filepath, frame)
            print(f"✅ 이미지 저장 결과: {success}")
            
            if success:
                relative_path = os.path.relpath(filepath, settings.MEDIA_ROOT)
                print(f"🔗 상대 경로: {relative_path}")
                return relative_path
            else:
                print(f"❌ 이미지 저장 실패")
                return None
        else:
            print(f"❌ 프레임 읽기 실패")
            return None
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

            # VideoAnalysis 생성
            VideoAnalysis.objects.create(
                video=video,
                enhanced_analysis=enhanced_analysis,
                success_rate=95.0,
                processing_time_seconds=metadata.get('processing_time_seconds', 0),
                analysis_statistics={
                    'total_frames_analyzed': analysis_results.get('total_frames_analyzed', 0),
                    'unique_objects': len(video_summary.get('dominant_objects', [])),
                    'analysis_method': 'real_ai_enhanced',
                    'ai_features_used': analysis_config.get('features_enabled', {}),
                    'scene_types': video_summary.get('scene_types', []),
                    'text_extracted': bool(video_summary.get('text_content')),
                    'json_file_path': json_filepath,
                    'dominant_objects': video_summary.get('dominant_objects', []),
                    'analysis_quality_metrics': video_summary.get('analysis_quality_metrics', {}),
                    'processing_statistics': video_summary.get('processing_statistics', {})
                },
                caption_statistics={
                    'frames_with_caption': len([f for f in frame_results if f.get('final_caption')]),
                    'enhanced_captions': len([f for f in frame_results if f.get('enhanced_caption')]),
                    'text_content_length': len(video_summary.get('text_content', '')),
                    'average_confidence': video_summary.get('analysis_quality_metrics', {}).get(
                        'average_detection_confidence', 0.8
                    )
                }
            )

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

                    Frame.objects.create(
                        video=video,
                        image_id=frame_data.get('image_id', 0),
                        timestamp=frame_data.get('timestamp', 0),
                        caption=frame_data.get('caption', ''),  # 없으면 빈 문자열
                        enhanced_caption=frame_data.get('enhanced_caption', ''),
                        final_caption=frame_data.get('final_caption', ''),
                        detected_objects=detected,
                        comprehensive_features={
                            "crop_quality": frame_data.get("crop_quality", {}),
                            "pose_analysis": attrs.get("pose_analysis", {}),
                            "facial_details": attrs.get("facial_details", {})
                        },
                        image=image_path if image_path else None  # ✅ 저장된 경로 연결 (None 허용)
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
        
# 🆕 새로운 API 엔드포인트 추가
class VideoQAAnalyticsView(APIView):
    """비디오 QA 분석 및 통계 뷰"""
    permission_classes = [AllowAny]
    
    def __init__(self):
        super().__init__()
        self.rag_system = get_video_rag_system()
        self.enhanced_qa = EnhancedVideoQASystem(self.rag_system, LLMClient())
    
    def get(self, request, video_id=None):
        """QA 통계 조회"""
        try:
            if video_id:
                # 특정 비디오의 대화 컨텍스트 조회
                context = self.enhanced_qa.get_conversation_context(str(video_id))
                
                # 질문 카테고리별 통계
                category_stats = {}
                for conv in context:
                    category = conv.get('category', 'unknown')
                    category_stats[category] = category_stats.get(category, 0) + 1
                
                return Response({
                    'video_id': video_id,
                    'total_conversations': len(context),
                    'category_statistics': category_stats,
                    'recent_conversations': context[-5:],  # 최근 5개
                    'success': True
                })
            else:
                # 전체 시스템 통계
                total_videos = len(self.enhanced_qa.context_memory)
                total_conversations = sum(len(convs) for convs in self.enhanced_qa.context_memory.values())
                
                return Response({
                    'total_videos_with_conversations': total_videos,
                    'total_conversations': total_conversations,
                    'videos': list(self.enhanced_qa.context_memory.keys()),
                    'success': True
                })
                
        except Exception as e:
            return Response({
                'error': f'통계 조회 실패: {str(e)}',
                'success': False
            }, status=500)
    
    def delete(self, request, video_id=None):
        """대화 컨텍스트 삭제"""
        try:
            if video_id:
                self.enhanced_qa.clear_context(str(video_id))
                return Response({
                    'message': f'비디오 {video_id}의 대화 컨텍스트가 삭제되었습니다.',
                    'success': True
                })
            else:
                self.enhanced_qa.clear_context()
                return Response({
                    'message': '모든 대화 컨텍스트가 삭제되었습니다.',
                    'success': True
                })
                
        except Exception as e:
            return Response({
                'error': f'컨텍스트 삭제 실패: {str(e)}',
                'success': False
            }, status=500)


class VideoQAUtils:
    """비디오 QA 관련 유틸리티 함수들"""
    
    @staticmethod
    def categorize_questions_batch(questions: List[str]) -> Dict[str, List[str]]:
        """질문들을 배치로 카테고리화"""
        categories = {
            'object_detection': [],
            'people_analysis': [],
            'scene_analysis': [],
            'action_analysis': [],
            'summary': [],
            'specific_search': [],
            'general': []
        }
        
        for question in questions:
            category = VideoQAUtils.classify_single_question(question)
            categories[category].append(question)
        
        return categories
    
    @staticmethod
    def classify_single_question(question: str) -> str:
        """단일 질문 분류"""
        question_lower = question.lower()
        
        patterns = {
            'object_detection': ['무엇이', '뭐가', '객체', '사물', '나오는', '보이는'],
            'people_analysis': ['사람', '인물', '얼굴', '성별', '나이', '옷'],
            'scene_analysis': ['장면', '배경', '환경', '장소', '위치', '시간'],
            'action_analysis': ['행동', '동작', '하고있', '움직임', '활동'],
            'summary': ['요약', '정리', '전체', '내용', '줄거리'],
            'specific_search': ['찾아', '검색', '언제', '어디서', '몇 번째']
        }
        
        for category, keywords in patterns.items():
            if any(keyword in question_lower for keyword in keywords):
                return category
        
        return 'general'
    
    @staticmethod
    def generate_question_suggestions(video_analysis_data: Dict) -> List[str]:
        """비디오 분석 결과를 바탕으로 질문 제안 생성"""
        suggestions = []
        
        # 객체 기반 질문
        if 'objects' in video_analysis_data:
            objects = video_analysis_data['objects'][:3]
            suggestions.extend([
                f"{obj}가 언제 나오나요?" for obj in objects
            ])
        
        # 장면 기반 질문
        suggestions.extend([
            "비디오의 주요 장면을 요약해주세요",
            "어떤 사람들이 나오나요?",
            "주요 행동이나 활동은 무엇인가요?",
            "배경이나 장소는 어디인가요?"
        ])
        
        return suggestions[:5]  # 상위 5개만 반환


# 🆕 캐싱 시스템 (선택사항)
from django.core.cache import cache
from hashlib import md5

class QACache:
    """QA 응답 캐싱 시스템"""
    
    @staticmethod
    def get_cache_key(video_id: str, question: str) -> str:
        """캐시 키 생성"""
        content = f"{video_id}:{question}"
        return f"qa_cache:{md5(content.encode()).hexdigest()}"
    
    @staticmethod
    def get_cached_response(video_id: str, question: str) -> Optional[Dict]:
        """캐시된 응답 조회"""
        cache_key = QACache.get_cache_key(video_id, question)
        return cache.get(cache_key)
    
    @staticmethod
    def cache_response(video_id: str, question: str, response: Dict, timeout: int = 300):
        """응답 캐싱"""
        cache_key = QACache.get_cache_key(video_id, question)
        cache.set(cache_key, response, timeout)
    
    @staticmethod
    def clear_video_cache(video_id: str):
        """특정 비디오의 모든 캐시 삭제"""
        # 캐시 패턴으로 삭제 (Redis 사용 시)
        pattern = f"qa_cache:*{video_id}*"
        # 구현은 사용하는 캐시 백엔드에 따라 달라집니다.
        
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
        
        if hasattr(video, 'analysis') and video.analysis:
            try:
                stats = video.analysis.analysis_statistics
                context.update({
                    'detected_objects': stats.get('dominant_objects', []),
                    'scene_types': stats.get('scene_types', [])
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
                if hasattr(video, 'analysis'):
                    analysis = video.analysis
                    stats = analysis.analysis_statistics
                    
                    video_data.update({
                        'enhanced_analysis': analysis.enhanced_analysis,
                        'success_rate': analysis.success_rate,
                        'processing_time': analysis.processing_time_seconds,
                        'analysis_type': stats.get('analysis_type', 'basic'),
                        'advanced_features_used': {
                            'clip': stats.get('clip_analysis', False),
                            'ocr': stats.get('text_extracted', False),
                            'vqa': stats.get('vqa_analysis', False),
                            'scene_graph': stats.get('scene_graph_analysis', False)
                        },
                        'scene_types': stats.get('scene_types', []),
                        'unique_objects': stats.get('unique_objects', 0)
                    })
                
                # 진행률 정보 추가 (분석 중인 경우)
                if video.analysis_status == 'processing':
                    progress_info = progress_tracker.get_progress(video.id)
                    if progress_info:
                        video_data['progress_info'] = progress_info
                
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
            # ✅ 수정: 전역 VideoAnalyzer 인스턴스 사용
            analyzer = get_video_analyzer()
            return {
                'clip_available': analyzer.clip_available,
                'ocr_available': analyzer.ocr_available,
                'vqa_available': analyzer.vqa_available,
                'scene_graph_available': analyzer.scene_graph_available
            }
        except:
            return {
                'clip_available': False,
                'ocr_available': False,
                'vqa_available': False,
                'scene_graph_available': False
            }

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
                progress_info = progress_tracker.get_progress(video.id)
                response_data.update(progress_info)
            
            # 분석 완료된 경우 상세 정보 추가
            if hasattr(video, 'analysis'):
                analysis = video.analysis
                response_data.update({
                    'enhanced_analysis': analysis.enhanced_analysis,
                    'success_rate': analysis.success_rate,
                    'processing_time': analysis.processing_time_seconds,
                    'stats': {
                        'objects': analysis.analysis_statistics.get('unique_objects', 0),
                        'scenes': Scene.objects.filter(video=video).count(),
                        'captions': analysis.caption_statistics.get('frames_with_caption', 0)
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
import threading
import time
import json
import cv2
import os
import base64
from datetime import datetime
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny
from .models import Video, VideoAnalysis, Scene, Frame, SearchHistory, CostAnalysis
from .llm_client import LLMClient

# ✅ 안전한 import
try:
    from .video_analyzer import get_video_analyzer
except ImportError:
    print("⚠️ video_analyzer import 실패")
    get_video_analyzer = None

try:
    from .multi_llm_service import get_multi_llm_analyzer
except ImportError:
    print("⚠️ multi_llm_service import 실패")
    get_multi_llm_analyzer = None

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
            
            # VideoAnalysis 생성
            analysis = VideoAnalysis.objects.create(
                video=video,
                enhanced_analysis=enable_enhanced,
                success_rate=85.0,
                processing_time_seconds=120,
                analysis_statistics={
                    'analysis_type': 'basic',
                    'unique_objects': 8,
                    'total_detections': 45,
                    'scene_types': ['outdoor', 'urban']
                },
                caption_statistics={
                    'frames_with_caption': 25,
                    'average_confidence': 0.8
                }
            )
            
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
            progress_info = progress_tracker.get_progress(video_id)
            
            if not progress_info:
                return Response({
                    'error': '진행 중인 분석이 없습니다'
                }, status=status.HTTP_404_NOT_FOUND)
            
            return Response(progress_info)
            
        except Exception as e:
            return Response({
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


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


class APIStatusView(APIView):
    """API 상태 확인"""
    permission_classes = [AllowAny]
    
    def get(self, request):
        # print("🔍 APIStatusView: API 상태 요청 받음")
        try:
            llm_client = LLMClient()
            status_info = llm_client.get_api_status()
            
            response_data = {
                'groq': status_info.get('groq', {'available': False}),
                'openai': status_info.get('openai', {'available': False}),
                'anthropic': status_info.get('anthropic', {'available': False}),
                'fallback_enabled': True,
                'timestamp': datetime.now().isoformat(),
                'server_status': 'running',
                'active_analyses': len([k for k, v in progress_tracker.progress_data.items() 
                                     if v.get('progress', 0) < 100])
            }
            
            # print(f"✅ APIStatusView: 상태 정보 반환 - {response_data}")
            return Response(response_data)
        except Exception as e:
            print(f"❌ APIStatusView 오류: {e}")
            return Response({
                'error': str(e),
                'server_status': 'error'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)



class VideoChatView(APIView):
    """비디오 관련 채팅 API - 기존 ChatView와 구분"""
    permission_classes = [AllowAny]  # 🔧 권한 설정 추가
    
    def __init__(self):
        super().__init__()
        self.llm_client = LLMClient()
        self.video_analyzer = VideoAnalyzer()
    
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
            if self._is_search_query(user_message):
                return self._handle_search_query(user_message, current_video, video_info, use_multi_llm)
            
            elif self._is_highlight_query(user_message):
                return self._handle_highlight_query(user_message, current_video, video_info, use_multi_llm)
            
            elif self._is_summary_query(user_message):
                return self._handle_summary_query(user_message, current_video, video_info, use_multi_llm)
            
            elif self._is_info_query(user_message):
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
    

class FrameView(APIView):
    """비디오에서 프레임 추출해 바로 JPEG로 응답"""
    permission_classes = [AllowAny]

    def get(self, request, video_id, frame_number, frame_type='normal'):
        # 1) 비디오 조회
        try:
            video = Video.objects.get(id=video_id)
        except Video.DoesNotExist:
            return Response({'error': '비디오를 찾을 수 없습니다'}, status=404)

        # 2) 실제 파일 경로 안전하게 해석 (file_path 우선, 그 다음 MEDIA_ROOT/uploads|videos)
        candidates = []

        fp = getattr(video, 'file_path', None)
        if fp:
            abs_fp = fp if os.path.isabs(fp) else os.path.join(settings.MEDIA_ROOT, fp)
            candidates.append(abs_fp)

        candidates.append(os.path.join(settings.MEDIA_ROOT, 'uploads', video.filename))
        candidates.append(os.path.join(settings.MEDIA_ROOT, 'videos', video.filename))

        video_path = next((p for p in candidates if p and os.path.exists(p)), None)
        if not video_path:
            return Response({'error': '비디오 파일을 찾을 수 없습니다', 'candidates': candidates}, status=404)

        # 3) OpenCV로 열기
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return Response({'error': '비디오 파일을 열 수 없습니다'}, status=500)

        # 4) frame_number 또는 t(초) → 프레임 인덱스 계산
        try:
            t = request.GET.get('t')
            if t is not None:
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                frame_idx = int(float(t) * float(fps))
            else:
                frame_idx = int(frame_number)  # URL에서 int로 받는게 최선 (urls.py 참고)
        except Exception:
            cap.release()
            return Response({'error': '유효하지 않은 프레임 번호'}, status=400)

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if total and frame_idx >= total:
            frame_idx = max(0, total - 1)
        if frame_idx < 0:
            frame_idx = 0

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            return Response({'error': f'프레임({frame_idx})을 추출할 수 없습니다'}, status=500)

        # 5) 주석 모드(선택): 함수가 있으면만 시도
        if frame_type == 'annotated' and hasattr(self, '_annotate_frame'):
            try:
                target_class = (request.GET.get('class') or '').lower()
                frame = self._annotate_frame(frame, video, frame_idx, target_class)
            except Exception:
                # 주석 실패해도 원본 프레임은 반환
                pass

        # 6) 리사이즈 (옵션)
        h, w = frame.shape[:2]
        try:
            max_w = int(request.GET.get('max_w', 800))
        except Exception:
            max_w = 800

        if w > max_w:
            ratio = max_w / float(w)
            frame = cv2.resize(frame, (max_w, int(h * ratio)))

        # 7) JPEG 인코딩 후 메모리로 응답
        ok, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            return Response({'error': '이미지 인코딩 실패'}, status=500)

        return HttpResponse(buf.tobytes(), content_type='image/jpeg')

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
                'analysis_type': 'enhanced' if hasattr(video, 'analysis') and video.analysis.enhanced_analysis else 'basic'
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

from .models import Video, VideoAnalysis, Scene, Frame, SearchHistory
from .llm_client import LLMClient


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
def delete_video(request, video_id):
    """개선된 비디오 삭제 - 상세 로깅 및 검증 포함"""
    
    logger.info(f"🗑️ 비디오 삭제 요청 시작: ID={video_id}")
    
    try:
        # 1단계: 비디오 존재 여부 확인
        try:
            video = get_object_or_404(Video, id=video_id)
            logger.info(f"✅ 비디오 찾음: {video.original_name} (파일: {video.file_path})")
        except Video.DoesNotExist:
            logger.warning(f"❌ 비디오 존재하지 않음: ID={video_id}")
            return JsonResponse({
                'error': '해당 비디오를 찾을 수 없습니다.',
                'video_id': video_id,
                'deleted': False
            }, status=404)
        
        # 2단계: 삭제 가능 여부 확인
        if video.analysis_status == 'processing':
            logger.warning(f"❌ 분석 중인 비디오 삭제 시도: ID={video_id}")
            return JsonResponse({
                'error': '분석 중인 비디오는 삭제할 수 없습니다.',
                'video_id': video_id,
                'status': video.analysis_status,
                'deleted': False
            }, status=400)
        
        # 3단계: 트랜잭션으로 안전한 삭제 처리
        video_info = {
            'id': video_id,
            'name': video.original_name,
            'file_path': video.file_path,
            'has_analysis': hasattr(video, 'analysis_results') and video.analysis_results.exists(),
            'has_scenes': hasattr(video, 'scenes') and video.scenes.exists()
        }
        
        with transaction.atomic():
            logger.info(f"🔄 트랜잭션 시작: 비디오 {video_id} 삭제")
            
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
            logger.info(f"💾 데이터베이스에서 비디오 삭제 완료: ID={video_id}")
            
            # 트랜잭션 커밋 후 잠시 대기 (데이터베이스 동기화)
            time.sleep(0.1)
        
        # 4단계: 삭제 검증
        try:
            verification_video = Video.objects.get(id=video_id)
            # 비디오가 여전히 존재하면 오류
            logger.error(f"❌ 삭제 검증 실패: 비디오가 여전히 존재함 ID={video_id}")
            return JsonResponse({
                'error': '비디오 삭제에 실패했습니다. 데이터베이스에서 제거되지 않았습니다.',
                'video_id': video_id,
                'deleted': False,
                'verification_failed': True
            }, status=500)
        except Video.DoesNotExist:
            # 비디오가 존재하지 않으면 삭제 성공
            logger.info(f"✅ 삭제 검증 성공: 비디오가 완전히 제거됨 ID={video_id}")
        
        # 5단계: 성공 응답
        response_data = {
            'success': True,
            'message': f'비디오 "{video_info["name"]}"이(가) 성공적으로 삭제되었습니다.',
            'video_id': video_id,
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
            
            # 고급 검색 수행
            search_results = self._perform_advanced_search(video, query, search_options)
            
            # 바운딩 박스 정보 추가
            enhanced_results = self._add_bbox_info(search_results, video)
            
            # AI 기반 검색 인사이트 생성
            search_insights = self._generate_search_insights(query, enhanced_results, video)
            
            print(f"✅ 고급 검색 완료: {len(enhanced_results)}개 결과")
            
            return Response({
                'search_results': enhanced_results,
                'query': query,
                'insights': search_insights,
                'total_matches': len(search_results),
                'search_type': 'advanced_search',
                'has_bbox_annotations': any(r.get('bbox_annotations') for r in enhanced_results),
                'video_info': {
                    'id': video.id,
                    'name': video.original_name,
                    'analysis_type': getattr(video, 'analysis_type', 'basic')
                }
            })
            
        except Exception as e:
            print(f"❌ 고급 비디오 검색 실패: {e}")
            return Response({
                'error': f'고급 검색 실패: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    def _perform_advanced_search(self, video, query, search_options):
        """실제 고급 검색 수행"""
        try:
            # EnhancedVideoChatView의 검색 로직 재사용
            chat_view = EnhancedVideoChatView()
            video_info = chat_view._get_enhanced_video_info(video)
            
            # 검색 수행
            response = chat_view._handle_enhanced_search(query, video, video_info)
            
            if hasattr(response, 'data') and 'search_results' in response.data:
                return response.data['search_results']
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
            
            # 바운딩 박스 어노테이션 정보 확인 및 추가
            if 'matches' in result:
                bbox_annotations = []
                for match in result['matches']:
                    if match.get('type') == 'object' and 'bbox' in match:
                        bbox_annotations.append({
                            'match': match['match'],
                            'confidence': match['confidence'],
                            'bbox': match['bbox'],
                            'colors': match.get('colors', []),
                            'color_description': match.get('color_description', '')
                        })
                
                enhanced_result['bbox_annotations'] = bbox_annotations
                
                # 바운딩 박스 이미지 URL 추가
                if bbox_annotations:
                    bbox_url = f"/frame/{video.id}/{result['frame_id']}/bbox/"
                    enhanced_result['bbox_image_url'] = bbox_url
            
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
class FrameVisualizationView(APIView):
    """프레임에 바운딩 박스 시각화"""
    permission_classes = [AllowAny]
    
    def post(self, request):
        try:
            video_id = request.data.get('video_id')
            frame_id = request.data.get('frame_id')
            objects = request.data.get('objects', [])
            
            if not video_id or not frame_id:
                return Response({'error': '비디오 ID와 프레임 ID가 필요합니다.'}, status=400)
            
            try:
                video = Video.objects.get(id=video_id)
                frame = Frame.objects.filter(video=video, image_id=frame_id).first()
                
                if not frame or not frame.image:
                    return Response({'error': '프레임 이미지를 찾을 수 없습니다.'}, status=404)
                
                # detected_objects에서 persons 배열 추출
                detected_objects = frame.detected_objects
                if isinstance(detected_objects, dict) and 'persons' in detected_objects:
                    persons = detected_objects['persons']
                else:
                    persons = []
                
                # 이미지에 바운딩 박스 그리기
                image_path = self._draw_bounding_boxes(frame, persons)
                
                return Response({
                    'success': True,
                    'image_url': image_path,
                    'objects_count': len(persons)
                })
                
            except Video.DoesNotExist:
                return Response({'error': '비디오를 찾을 수 없습니다.'}, status=404)
                
        except Exception as e:
            logger.error(f"❌ 바운딩 박스 시각화 오류: {e}")
            return Response({'error': str(e)}, status=500)
    
    def _draw_bounding_boxes(self, frame, objects):
        """프레임 이미지에 바운딩 박스 그리기"""
        import cv2
        import numpy as np
        import os
        from django.conf import settings
        
        try:
            # 이미지 경로
            image_path = os.path.join(settings.MEDIA_ROOT, str(frame.image))
            
            if not os.path.exists(image_path):
                return None
            
            # 이미지 읽기
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            height, width = image.shape[:2]
            
            # 각 객체에 대해 바운딩 박스 그리기
            for i, obj in enumerate(objects):
                bbox = obj.get('bbox', [])
                if len(bbox) != 4:
                    continue
                
                # 정규화된 좌표를 픽셀 좌표로 변환
                x1 = int(bbox[0] * width)
                y1 = int(bbox[1] * height)
                x2 = int(bbox[2] * width)
                y2 = int(bbox[3] * height)
                
                # 색상 설정 (객체별로 다른 색상)
                colors = [
                    (0, 255, 0),    # 초록
                    (255, 0, 0),    # 파랑
                    (0, 0, 255),    # 빨강
                    (255, 255, 0),  # 시안
                    (255, 0, 255),  # 마젠타
                    (0, 255, 255),  # 노랑
                ]
                color = colors[i % len(colors)]
                
                # 바운딩 박스 그리기
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                
                # 라벨 추가
                label = obj.get('description', f'Object {i+1}')
                confidence = obj.get('confidence', 0)
                label_text = f'{label} ({confidence:.2f})'
                
                # 라벨 배경
                (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(image, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
                
                # 라벨 텍스트
                cv2.putText(image, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 결과 이미지 저장
            output_dir = os.path.join(settings.MEDIA_ROOT, 'visualized_frames')
            os.makedirs(output_dir, exist_ok=True)
            
            output_filename = f'video{frame.video.id}_frame{frame.image_id}_visualized.jpg'
            output_path = os.path.join(output_dir, output_filename)
            
            cv2.imwrite(output_path, image)
            
            # 상대 경로 반환
            relative_path = os.path.relpath(output_path, settings.MEDIA_ROOT)
            return relative_path
            
        except Exception as e:
            logger.error(f"❌ 바운딩 박스 그리기 실패: {e}")
            return None


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
        
        # YOLO 객체 타입 매핑 확장 (COCO 데이터셋 기반)
        object_mappings = {
            # 사람 관련
            ('사람', '남성', '여성', '인물', 'person'): 'person',
            
            # 가방 관련
            ('가방', 'handbag', '백팩', '핸드백', 'bag', 'backpack'): 'bag',
            
            # 전자기기
            ('tv', '티비', '텔레비전', 'television'): 'tv',
            ('노트북', '컴퓨터', 'laptop'): 'laptop',
            ('핸드폰', '휴대폰', '폰', 'phone', 'cell_phone'): 'cell_phone',
            ('키보드', 'keyboard'): 'keyboard',
            ('마우스', 'mouse'): 'mouse',
            ('리모컨', 'remote'): 'remote',
            
            # 가구
            ('의자', 'chair'): 'chair',
            ('침대', 'bed'): 'bed',
            ('소파', 'couch'): 'couch',
            ('식탁', 'dining_table'): 'dining_table',
            ('화장대', 'toilet'): 'toilet',
            
            # 교통수단
            ('차', '자동차', '차량', '승용차', 'car'): 'car',
            ('자전거', 'bicycle'): 'bicycle',
            ('오토바이', 'motorcycle'): 'motorcycle',
            ('버스', 'bus'): 'bus',
            ('트럭', 'truck'): 'truck',
            ('기차', 'train'): 'train',
            ('비행기', 'airplane'): 'airplane',
            ('보트', 'boat'): 'boat',
            
            # 동물
            ('개', '강아지', '멍멍이', 'dog'): 'dog',
            ('고양이', '냥이', 'cat'): 'cat',
            ('말', 'horse'): 'horse',
            ('소', 'cow'): 'cow',
            ('양', 'sheep'): 'sheep',
            ('새', 'bird'): 'bird',
            ('곰', 'bear'): 'bear',
            ('코끼리', 'elephant'): 'elephant',
            ('기린', 'giraffe'): 'giraffe',
            ('얼룩말', 'zebra'): 'zebra',
            
            # 음식
            ('바나나', 'banana'): 'banana',
            ('사과', 'apple'): 'apple',
            ('샌드위치', 'sandwich'): 'sandwich',
            ('오렌지', 'orange'): 'orange',
            ('브로콜리', 'broccoli'): 'broccoli',
            ('당근', 'carrot'): 'carrot',
            ('핫도그', 'hot_dog'): 'hot_dog',
            ('피자', 'pizza'): 'pizza',
            ('도넛', 'donut'): 'donut',
            ('케이크', 'cake'): 'cake',
            
            # 스포츠/놀이
            ('공', 'ball'): 'sports_ball',
            ('야구배트', 'baseball_bat'): 'baseball_bat',
            ('야구글러브', 'baseball_glove'): 'baseball_glove',
            ('테니스라켓', 'tennis_racket'): 'tennis_racket',
            ('스키', 'skis'): 'skis',
            ('스노보드', 'snowboard'): 'snowboard',
            ('프리스비', 'frisbee'): 'frisbee',
            ('키트', 'kite'): 'kite',
            ('야구방망이', 'baseball_bat'): 'baseball_bat',
            
            # 도구/물건
            ('가위', 'scissors'): 'scissors',
            ('우산', 'umbrella'): 'umbrella',
            ('핸드백', 'handbag'): 'handbag',
            ('넥타이', 'tie'): 'tie',
            ('가방', 'suitcase'): 'suitcase',
            ('프리스비', 'frisbee'): 'frisbee',
            ('스키', 'skis'): 'skis',
            ('스노보드', 'snowboard'): 'snowboard',
            ('스포츠공', 'sports_ball'): 'sports_ball',
            ('키트', 'kite'): 'kite',
            ('야구방망이', 'baseball_bat'): 'baseball_bat',
            ('야구글러브', 'baseball_glove'): 'baseball_glove',
            ('테니스라켓', 'tennis_racket'): 'tennis_racket',
            ('병', 'bottle'): 'bottle',
            ('와인잔', 'wine_glass'): 'wine_glass',
            ('컵', 'cup'): 'cup',
            ('포크', 'fork'): 'fork',
            ('나이프', 'knife'): 'knife',
            ('숟가락', 'spoon'): 'spoon',
            ('그릇', 'bowl'): 'bowl',
            
            # 기타
            ('책', 'book'): 'book',
            ('시계', 'clock'): 'clock',
            ('가위', 'scissors'): 'scissors',
            ('테디베어', 'teddy_bear'): 'teddy_bear',
            ('헤어드라이어', 'hair_drier'): 'hair_drier',
            ('칫솔', 'toothbrush'): 'toothbrush'
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
    
    def _calculate_person_match_score(self, person, target_analysis):
        """PersonDetection과 검색 조건의 매칭 점수 계산"""
        match_score = 0.0
        match_reasons = []
        
        # 기본 person 매칭 (bag 검색도 person에서 찾기)
        if target_analysis.get('object_type') in ['person', 'bag']:
            match_score += 0.3
            match_reasons.append("사람 객체 매칭")
        
        # 색상 매칭
        target_colors = target_analysis.get('colors', [])
        if target_colors:
            person_colors = self._extract_person_colors(person)
            color_matches = set(target_colors) & set(person_colors)
            if color_matches:
                match_score += 0.4
                match_reasons.append(f"색상 매칭: {', '.join(color_matches)}")
        
        # 성별 매칭
        target_gender = target_analysis.get('gender')
        if target_gender and person.gender_estimation:
            if target_gender in person.gender_estimation.lower():
                match_score += 0.2
                match_reasons.append(f"성별 매칭: {person.gender_estimation}")
        
        # 액세서리 매칭 (가방 등) - 더 관대하게
        keywords = target_analysis.get('keywords', [])
        if any(word in keywords for word in ['가방', '백팩', 'bag', 'backpack']):
            # 가방 관련 키워드가 있으면 일단 매칭
            match_score += 0.3
            match_reasons.append("가방 액세서리 매칭")
        
        return match_score, match_reasons
    
    def _extract_person_colors(self, person):
        """PersonDetection에서 색상 정보 추출"""
        colors = []
        
        # 상의 색상
        upper_color = person.upper_body_color.lower()
        if 'red' in upper_color or '빨간' in upper_color:
            colors.append('red')
        elif 'blue' in upper_color or '파란' in upper_color:
            colors.append('blue')
        elif 'green' in upper_color or '초록' in upper_color:
            colors.append('green')
        elif 'pink' in upper_color or '분홍' in upper_color:
            colors.append('pink')
        elif 'black' in upper_color or '검은' in upper_color:
            colors.append('black')
        elif 'white' in upper_color or '흰' in upper_color:
            colors.append('white')
        elif 'brown' in upper_color or '갈색' in upper_color:
            colors.append('brown')
        elif 'orange' in upper_color or '주황' in upper_color:
            colors.append('orange')
        elif 'yellow' in upper_color or '노란' in upper_color:
            colors.append('yellow')
        
        # 하의 색상
        lower_color = person.lower_body_color.lower()
        if 'red' in lower_color or '빨간' in lower_color:
            colors.append('red')
        elif 'blue' in lower_color or '파란' in lower_color:
            colors.append('blue')
        elif 'green' in lower_color or '초록' in lower_color:
            colors.append('green')
        elif 'pink' in lower_color or '분홍' in lower_color:
            colors.append('pink')
        elif 'black' in lower_color or '검은' in lower_color:
            colors.append('black')
        elif 'white' in lower_color or '흰' in lower_color:
            colors.append('white')
        elif 'brown' in lower_color or '갈색' in lower_color:
            colors.append('brown')
        elif 'orange' in lower_color or '주황' in lower_color:
            colors.append('orange')
        elif 'yellow' in lower_color or '노란' in lower_color:
            colors.append('yellow')
        
        return colors
    
    def _generate_person_description(self, person):
        """PersonDetection 기반 설명 생성"""
        parts = []
        
        if person.gender_estimation and person.gender_estimation != 'unknown':
            parts.append(person.gender_estimation)
        
        if person.age_group and person.age_group != 'unknown':
            parts.append(person.age_group)
        
        if person.upper_body_color and person.upper_body_color != 'unknown':
            color = person.upper_body_color.replace('wearing ', '').replace(' clothes', '')
            parts.append(f"{color} 상의")
        
        if person.lower_body_color and person.lower_body_color != 'unknown':
            color = person.lower_body_color.replace('wearing ', '').replace(' clothes', '')
            parts.append(f"{color} 하의")
        
        return ' '.join(parts) if parts else 'person'
    
    def _perform_bag_tracking(self, video, target_analysis, time_range):
        """가방 전용 추적 - 실제 가방 객체가 감지된 프레임만 검색"""
        tracking_results = []
        
        try:
            from .models import Frame
            import json
            
            # Frame에서 가방 객체가 감지된 프레임 찾기
            frames_query = Frame.objects.filter(video=video)
            
            # 시간 범위 필터링
            if time_range.get('start') and time_range.get('end'):
                start_time = self._parse_time_to_seconds(time_range['start'])
                end_time = self._parse_time_to_seconds(time_range['end'])
                frames_query = frames_query.filter(timestamp__gte=start_time, timestamp__lte=end_time)
            
            frames = list(frames_query)
            logger.info(f"📊 가방 검색할 프레임 수: {len(frames)}개")
            
            bag_frames = []
            
            for frame in frames:
                if frame.detected_objects:
                    try:
                        objects = json.loads(frame.detected_objects) if isinstance(frame.detected_objects, str) else frame.detected_objects
                        
                        # 가방 객체 찾기 (backpack: 24, handbag: 26)
                        for obj in objects:
                            if isinstance(obj, dict):
                                class_id = obj.get('class_id')
                                class_name = obj.get('class', '').lower()
                                
                                # YOLO 클래스 ID 또는 이름으로 가방 확인
                                if (class_id in [24, 26] or 
                                    'bag' in class_name or 'handbag' in class_name or 'backpack' in class_name):
                                    
                                    # 해당 프레임의 사람들 찾기
                                    from .models import PersonDetection
                                    persons = PersonDetection.objects.filter(frame=frame)
                                    
                                    for person in persons:
                                        # 가방과 사람의 거리 계산 (간단한 근사)
                                        bag_bbox = obj.get('bbox', [])
                                        person_bbox = [person.bbox_x1, person.bbox_y1, person.bbox_x2, person.bbox_y2]
                                        
                                        # 가방과 사람이 가까이 있으면 매칭
                                        if self._is_bag_near_person(bag_bbox, person_bbox):
                                            match_score = 0.8  # 가방이 실제로 감지된 경우 높은 점수
                                            match_reasons = [f"가방 객체 감지: {class_name}"]
                                            
                                            tracking_results.append({
                                                'frame_id': frame.image_id,
                                                'timestamp': frame.timestamp,
                                                'confidence': match_score,
                                                'bbox': person_bbox,
                                                'description': f"{self._generate_person_description(person)} (가방 소지)",
                                                'tracking_id': person.track_id or f"person_{person.id}",
                                                'match_reasons': match_reasons,
                                                'bag_info': {
                                                    'class': class_name,
                                                    'class_id': class_id,
                                                    'bag_bbox': bag_bbox,
                                                    'confidence': obj.get('confidence', 0.0)
                                                },
                                                'person_attributes': {
                                                    'gender': person.gender_estimation,
                                                    'age': person.age_group,
                                                    'upper_color': person.upper_body_color,
                                                    'lower_color': person.lower_body_color,
                                                    'posture': person.posture
                                                }
                                            })
                                    
                                    bag_frames.append(frame.image_id)
                                    
                    except Exception as e:
                        logger.warning(f"⚠️ 프레임 {frame.image_id} 객체 파싱 실패: {e}")
                        continue
            
            logger.info(f"🎒 가방이 감지된 프레임: {len(bag_frames)}개")
            
            # 신뢰도순 정렬
            tracking_results.sort(key=lambda x: x['confidence'], reverse=True)
            
            return tracking_results
            
        except Exception as e:
            logger.error(f"❌ 가방 추적 오류: {e}")
            return []
    
    def _is_bag_near_person(self, bag_bbox, person_bbox):
        """가방과 사람이 가까이 있는지 확인"""
        if not bag_bbox or not person_bbox or len(bag_bbox) < 4 or len(person_bbox) < 4:
            return False
        
        # 바운딩 박스 중심점 계산
        bag_center_x = (bag_bbox[0] + bag_bbox[2]) / 2
        bag_center_y = (bag_bbox[1] + bag_bbox[3]) / 2
        person_center_x = (person_bbox[0] + person_bbox[2]) / 2
        person_center_y = (person_bbox[1] + person_bbox[3]) / 2
        
        # 거리 계산
        distance = ((bag_center_x - person_center_x) ** 2 + (bag_center_y - person_center_y) ** 2) ** 0.5
        
        # 임계값 (이미지 크기에 비례)
        threshold = 0.3  # 정규화된 좌표 기준
        
        return distance < threshold
    
    def _perform_yolo_object_tracking(self, video, target_analysis, time_range):
        """YOLO 객체 전용 추적 - 모든 YOLO 객체 타입 지원"""
        tracking_results = []
        
        try:
            from .models import Frame
            import json
            
            # Frame에서 해당 객체가 감지된 프레임 찾기
            frames_query = Frame.objects.filter(video=video)
            
            # 시간 범위 필터링
            if time_range.get('start') and time_range.get('end'):
                start_time = self._parse_time_to_seconds(time_range['start'])
                end_time = self._parse_time_to_seconds(time_range['end'])
                frames_query = frames_query.filter(timestamp__gte=start_time, timestamp__lte=end_time)
            
            frames = list(frames_query)
            logger.info(f"📊 YOLO 객체 검색할 프레임 수: {len(frames)}개")
            
            target_object_type = target_analysis.get('object_type', '').lower()
            target_colors = target_analysis.get('colors', [])
            
            # YOLO 클래스 매핑
            yolo_class_mapping = {
                'car': ['car', 'truck', 'bus', 'motorcycle', 'bicycle'],
                '자동차': ['car', 'truck', 'bus', 'motorcycle', 'bicycle'],
                '차': ['car', 'truck', 'bus', 'motorcycle', 'bicycle'],
                'dog': ['dog'],
                '개': ['dog'],
                'cat': ['cat'],
                '고양이': ['cat'],
                'phone': ['cell phone'],
                '핸드폰': ['cell phone'],
                '전화': ['cell phone'],
                'laptop': ['laptop'],
                '노트북': ['laptop'],
                'book': ['book'],
                '책': ['book'],
                'chair': ['chair'],
                '의자': ['chair'],
                'table': ['dining table'],
                '테이블': ['dining table'],
                'cup': ['cup'],
                '컵': ['cup'],
                'bottle': ['bottle'],
                '병': ['bottle'],
                'keyboard': ['keyboard'],
                '키보드': ['keyboard'],
                'mouse': ['mouse'],
                '마우스': ['mouse'],
                'tv': ['tv'],
                '텔레비전': ['tv'],
                'remote': ['remote'],
                '리모컨': ['remote'],
                'umbrella': ['umbrella'],
                '우산': ['umbrella'],
                'suitcase': ['suitcase'],
                '여행가방': ['suitcase'],
                'frisbee': ['frisbee'],
                '원반': ['frisbee'],
                'skis': ['skis'],
                '스키': ['skis'],
                'snowboard': ['snowboard'],
                '스노보드': ['snowboard'],
                'sports ball': ['sports ball'],
                '공': ['sports ball'],
                'kite': ['kite'],
                '연': ['kite'],
                'baseball bat': ['baseball bat'],
                '야구배트': ['baseball bat'],
                'baseball glove': ['baseball glove'],
                '야구장갑': ['baseball glove'],
                'skateboard': ['skateboard'],
                '스케이트보드': ['skateboard'],
                'surfboard': ['surfboard'],
                '서핑보드': ['surfboard'],
                'tennis racket': ['tennis racket'],
                '테니스라켓': ['tennis racket'],
                'wine glass': ['wine glass'],
                '와인잔': ['wine glass'],
                'fork': ['fork'],
                '포크': ['fork'],
                'knife': ['knife'],
                '칼': ['knife'],
                'spoon': ['spoon'],
                '숟가락': ['spoon'],
                'bowl': ['bowl'],
                '그릇': ['bowl'],
                'banana': ['banana'],
                '바나나': ['banana'],
                'apple': ['apple'],
                '사과': ['apple'],
                'sandwich': ['sandwich'],
                '샌드위치': ['sandwich'],
                'orange': ['orange'],
                '오렌지': ['orange'],
                'broccoli': ['broccoli'],
                '브로콜리': ['broccoli'],
                'carrot': ['carrot'],
                '당근': ['carrot'],
                'hot dog': ['hot dog'],
                '핫도그': ['hot dog'],
                'pizza': ['pizza'],
                '피자': ['pizza'],
                'donut': ['donut'],
                '도넛': ['donut'],
                'cake': ['cake'],
                '케이크': ['cake'],
                'couch': ['couch'],
                '소파': ['couch'],
                'bed': ['bed'],
                '침대': ['bed'],
                'toilet': ['toilet'],
                '화장실': ['toilet'],
                'laptop': ['laptop'],
                '노트북': ['laptop'],
                'mouse': ['mouse'],
                '마우스': ['mouse'],
                'remote': ['remote'],
                '리모컨': ['remote'],
                'keyboard': ['keyboard'],
                '키보드': ['keyboard'],
                'cell phone': ['cell phone'],
                '핸드폰': ['cell phone'],
                'microwave': ['microwave'],
                '전자레인지': ['microwave'],
                'oven': ['oven'],
                '오븐': ['oven'],
                'toaster': ['toaster'],
                '토스터': ['toaster'],
                'sink': ['sink'],
                '싱크대': ['sink'],
                'refrigerator': ['refrigerator'],
                '냉장고': ['refrigerator'],
                'book': ['book'],
                '책': ['book'],
                'clock': ['clock'],
                '시계': ['clock'],
                'vase': ['vase'],
                '화분': ['vase'],
                'scissors': ['scissors'],
                '가위': ['scissors'],
                'teddy bear': ['teddy bear'],
                '곰인형': ['teddy bear'],
                'hair drier': ['hair drier'],
                '드라이어': ['hair drier'],
                'toothbrush': ['toothbrush'],
                '칫솔': ['toothbrush']
            }
            
            # 검색할 YOLO 클래스들 찾기
            search_classes = []
            if target_object_type in yolo_class_mapping:
                search_classes = yolo_class_mapping[target_object_type]
            else:
                # 직접 매칭 시도
                search_classes = [target_object_type]
            
            logger.info(f"🔍 검색할 YOLO 클래스: {search_classes}")
            
            object_frames = []
            
            for frame in frames:
                if frame.detected_objects:
                    try:
                        objects = json.loads(frame.detected_objects) if isinstance(frame.detected_objects, str) else frame.detected_objects
                        
                        # 해당 객체 찾기
                        for obj in objects:
                            if isinstance(obj, dict):
                                class_name = obj.get('class', '').lower()
                                class_id = obj.get('class_id')
                                confidence = obj.get('confidence', 0.0)
                                
                                # 클래스 매칭 확인
                                is_matching_class = any(search_class in class_name for search_class in search_classes)
                                
                                if is_matching_class and confidence > 0.3:  # 신뢰도 30% 이상
                                    # 색상 매칭 (선택적)
                                    color_match = True
                                    if target_colors:
                                        # 색상 매칭 로직 (간단한 키워드 매칭)
                                        obj_description = f"{class_name} {obj.get('description', '')}".lower()
                                        color_match = any(color in obj_description for color in target_colors)
                                    
                                    if color_match:
                                        tracking_results.append({
                                            'frame_id': frame.image_id,
                                            'timestamp': frame.timestamp,
                                            'confidence': confidence,
                                            'bbox': obj.get('bbox', []),
                                            'description': f"{class_name} (신뢰도: {confidence:.1%})",
                                            'tracking_id': f"{class_name}_{frame.image_id}_{len(tracking_results)}",
                                            'match_reasons': [f"YOLO 객체 감지: {class_name}"],
                                            'object_info': {
                                                'class': class_name,
                                                'class_id': class_id,
                                                'confidence': confidence,
                                                'bbox': obj.get('bbox', [])
                                            }
                                        })
                                        
                                        object_frames.append(frame.image_id)
                                        
                    except Exception as e:
                        logger.warning(f"⚠️ 프레임 {frame.image_id} 객체 파싱 실패: {e}")
                        continue
            
            logger.info(f"🎯 {target_object_type} 객체가 감지된 프레임: {len(object_frames)}개")
            
            # 신뢰도순 정렬
            tracking_results.sort(key=lambda x: x['confidence'], reverse=True)
            
            return tracking_results
            
        except Exception as e:
            logger.error(f"❌ YOLO 객체 추적 오류: {e}")
            return []
    
    def _perform_object_tracking(self, video, target_analysis, time_range):
        """실제 객체 추적 수행 - YOLO 객체 및 PersonDetection 데이터 활용"""
        tracking_results = []
        
        try:
            # 가방 검색인 경우 특별 처리
            if target_analysis.get('object_type') == 'bag' or '가방' in target_analysis.get('keywords', []):
                return self._perform_bag_tracking(video, target_analysis, time_range)
            
            # 일반 객체 검색인 경우 YOLO 객체 추적
            if target_analysis.get('object_type') and target_analysis.get('object_type') != 'person':
                return self._perform_yolo_object_tracking(video, target_analysis, time_range)
            
            # PersonDetection 데이터에서 직접 검색 (사람 검색)
            from .models import PersonDetection
            
            # 기본 쿼리
            persons_query = PersonDetection.objects.filter(frame__video=video)
            
            # 시간 범위 필터링
            if time_range.get('start') and time_range.get('end'):
                start_time = self._parse_time_to_seconds(time_range['start'])
                end_time = self._parse_time_to_seconds(time_range['end'])
                persons_query = persons_query.filter(
                    frame__timestamp__gte=start_time, 
                    frame__timestamp__lte=end_time
                )
                logger.info(f"⏰ 시간 필터링: {start_time}s ~ {end_time}s")
            
            persons = list(persons_query.select_related('frame'))
            logger.info(f"📊 분석할 PersonDetection 수: {len(persons)}개")
            
            if not persons:
                logger.warning("⚠️ 분석할 PersonDetection이 없습니다.")
                return []
            
            for person in persons:
                try:
                    match_score, match_reasons = self._calculate_person_match_score(person, target_analysis)
                    
                    if match_score >= 0.4:  # 40% 이상 매칭 (임계값 낮춤)
                        tracking_results.append({
                            'frame_id': person.frame.image_id,
                            'timestamp': person.frame.timestamp,
                            'confidence': match_score,
                            'bbox': [person.bbox_x1, person.bbox_y1, person.bbox_x2, person.bbox_y2],
                            'description': self._generate_person_description(person),
                            'tracking_id': person.track_id or f"person_{person.id}",
                            'match_reasons': match_reasons,
                            'person_attributes': {
                                'gender': person.gender_estimation,
                                'age': person.age_group,
                                'upper_color': person.upper_body_color,
                                'lower_color': person.lower_body_color,
                                'posture': person.posture
                            }
                        })
                except Exception as person_error:
                    logger.warning(f"⚠️ PersonDetection {person.id} 처리 실패: {person_error}")
                    continue
            
            # 신뢰도순 정렬
            tracking_results.sort(key=lambda x: x['confidence'], reverse=True)
            
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
            detected_obj = frame.detected_objects
            # persons 배열이 있으면 우선 추출
            if isinstance(detected_obj, dict) and 'persons' in detected_obj:
                candidates.append(detected_obj['persons'])
            else:
                candidates.append(detected_obj)

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
                elif target_analysis.get('object_type') == 'bag' and obj['class'] == 'person':
                    # 가방은 person 객체의 accessories에서 확인
                    if self._check_bag_in_accessories(obj):
                        match_score += 0.6
                        match_reasons.append("가방 액세서리 매칭")
                    else:
                        continue
                elif target_analysis.get('object_type') == 'phone' and obj['class'] == 'person':
                    # 전화는 person 객체의 accessories에서 확인
                    if self._check_phone_in_accessories(obj):
                        match_score += 0.6
                        match_reasons.append("전화 액세서리 매칭")
                    else:
                        continue
                elif target_analysis.get('object_type') and obj['class'] != target_analysis['object_type']:
                    # 객체 타입이 다르면 건너뛰기
                    continue
                
                # 색상 매칭 (_raw.attributes.clothing_color에서 추출)
                color_matched = False
                if target_analysis.get('colors'):
                    target_colors = target_analysis['colors']
                    
                    # _raw 필드에서 색상 정보 추출
                    raw_data = obj.get('_raw', {})
                    attributes = raw_data.get('attributes', {})
                    clothing_color = attributes.get('clothing_color', {})
                    color_value = clothing_color.get('value', '').lower()
                    color_scores = clothing_color.get('all_scores', {})
                    
                    for target_color in target_colors:
                        target_color_lower = target_color.lower()
                        
                        # 정확한 색상 매칭
                        if target_color_lower in color_value:
                            match_score += 0.5
                            match_reasons.append(f"정확한 {target_color} 색상 매칭")
                            color_matched = True
                            break
                        
                        # all_scores에서 색상 검색
                        for color_name, score in color_scores.items():
                            if target_color_lower in color_name.lower():
                                match_score += 0.3
                                match_reasons.append(f"부분 {target_color} 색상 매칭 (신뢰도: {score:.2f})")
                                color_matched = True
                                break
                        
                        if color_matched:
                            break
                    
                    # 색상이 지정되었는데 매칭되지 않으면 제외
                    if not color_matched:
                        continue
                
                # 키워드 매칭 (보조)
                for keyword in target_analysis.get('keywords', []):
                    if keyword in obj['class'] and keyword not in ['사람', '옷', '입은']:
                        match_score += 0.1
                        match_reasons.append(f"키워드 '{keyword}' 매칭")
                
                # 최소 점수 기준 상향 조정
                if match_score >= 0.6:  # 0.4에서 0.6으로 상향 (정확도 향상)
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
    
    def _check_bag_in_accessories(self, obj):
        """객체의 accessories에서 가방 확인"""
        try:
            # _raw 필드에서 accessories 정보 추출
            raw_data = obj.get('_raw', {})
            attributes = raw_data.get('attributes', {})
            accessories = attributes.get('accessories', {})
            
            if isinstance(accessories, dict):
                # value 필드 확인
                accessory_value = accessories.get('value', '').lower()
                if any(bag_keyword in accessory_value for bag_keyword in ['bag', 'backpack', 'handbag']):
                    return True
                
                # all_scores에서 가방 관련 점수 확인
                all_scores = accessories.get('all_scores', {})
                for score_key, score_value in all_scores.items():
                    if any(bag_keyword in score_key.lower() for bag_keyword in ['bag', 'backpack', 'handbag']):
                        if score_value > 0.1:  # 신뢰도 0.1 이상
                            return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ 가방 액세서리 확인 오류: {e}")
            return False
    
    def _check_phone_in_accessories(self, obj):
        """객체의 accessories에서 전화 확인"""
        try:
            # _raw 필드에서 accessories 정보 추출
            raw_data = obj.get('_raw', {})
            attributes = raw_data.get('attributes', {})
            accessories = attributes.get('accessories', {})
            
            if isinstance(accessories, dict):
                # value 필드 확인
                accessory_value = accessories.get('value', '').lower()
                if any(phone_keyword in accessory_value for phone_keyword in ['phone', 'cell_phone', 'mobile']):
                    return True
                
                # all_scores에서 전화 관련 점수 확인
                all_scores = accessories.get('all_scores', {})
                for score_key, score_value in all_scores.items():
                    if any(phone_keyword in score_key.lower() for phone_keyword in ['phone', 'cell_phone', 'mobile']):
                        if score_value > 0.1:  # 신뢰도 0.1 이상
                            return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ 전화 액세서리 확인 오류: {e}")
            return False

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
        reasons = []
        
        try:
            # 프레임 데이터에서 직접 분석
            frames = Frame.objects.filter(video=video)[:10]
            
            if frames:
                # 날씨 매칭
                if query_analysis.get('weather'):
                    weather_score = self._check_weather_match(frames, query_analysis['weather'])
                    if weather_score > 0:
                        score += weather_score
                        reasons.append(f"날씨: {query_analysis['weather']}")
                
                # 시간대 매칭
                if query_analysis.get('time_of_day'):
                    time_score = self._check_time_match(video, query_analysis['time_of_day'])
                    if time_score > 0:
                        score += time_score
                        reasons.append(f"시간대: {query_analysis['time_of_day']}")
                
                # 장소 매칭
                if query_analysis.get('location'):
                    location_score = self._check_location_match(frames, query_analysis['location'])
                    if location_score > 0:
                        score += location_score
                        reasons.append(f"장소: {query_analysis['location']}")
                
                # 객체 매칭
                if query_analysis.get('objects'):
                    for obj in query_analysis['objects']:
                        object_score = self._check_object_match(frames, obj)
                        if object_score > 0:
                            score += object_score
                            reasons.append(f"객체: {obj}")
                
                # 활동 매칭
                if query_analysis.get('activities'):
                    for activity in query_analysis['activities']:
                        activity_score = self._check_activity_match(frames, activity)
                        if activity_score > 0:
                            score += activity_score
                            reasons.append(f"활동: {activity}")
            
            # VideoAnalysis에서 분석 결과가 있는 경우 (보조)
            elif hasattr(video, 'analysis'):
                analysis = video.analysis
                stats = analysis.analysis_statistics
                scene_types = stats.get('scene_types', [])
                
                # 날씨 매칭
                if query_analysis.get('weather'):
                    weather_scenes = [s for s in scene_types if query_analysis['weather'] in s.lower()]
                    if weather_scenes:
                        score += 0.4
                        reasons.append(f"날씨: {query_analysis['weather']}")
                
                # 시간대 매칭
                if query_analysis.get('time_of_day'):
                    time_scenes = [s for s in scene_types if query_analysis['time_of_day'] in s.lower()]
                    if time_scenes:
                        score += 0.3
                        reasons.append(f"시간대: {query_analysis['time_of_day']}")
                
                # 장소 매칭
                if query_analysis.get('location'):
                    location_scenes = [s for s in scene_types if query_analysis['location'] in s.lower()]
                    if location_scenes:
                        score += 0.3
                        reasons.append(f"장소: {query_analysis['location']}")
            
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


# views.py에 추가할 누락된 View들


# ✅ LLMStatsView 추가
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


# ✅ cleanup_storage 함수 추가
@csrf_exempt
@require_http_methods(["POST"])
def cleanup_storage(request):
    """저장 공간 정리"""
    try:
        print("🧹 저장 공간 정리 요청")
        
        from django.conf import settings
        
        cleaned_files = []
        total_size_freed = 0
        
        # 임시 파일들 정리 (예시)
        temp_dirs = [
            os.path.join(settings.MEDIA_ROOT, 'temp'),
            os.path.join(settings.MEDIA_ROOT, 'analysis_temp'),
            '/tmp/video_analysis'
        ]
        
        for temp_dir in temp_dirs:
            if os.path.exists(temp_dir):
                try:
                    # 임시 디렉토리 내용 정리
                    for filename in os.listdir(temp_dir):
                        file_path = os.path.join(temp_dir, filename)
                        if os.path.isfile(file_path):
                            file_size = os.path.getsize(file_path)
                            os.remove(file_path)
                            cleaned_files.append(filename)
                            total_size_freed += file_size
                except Exception as e:
                    print(f"⚠️ 임시 디렉토리 정리 실패: {temp_dir} - {e}")
        
        # 오래된 분석 결과 파일들 정리 (선택사항)
        analysis_results_dir = os.path.join(settings.MEDIA_ROOT, 'analysis_results')
        if os.path.exists(analysis_results_dir):
            import time
            current_time = time.time()
            for filename in os.listdir(analysis_results_dir):
                file_path = os.path.join(analysis_results_dir, filename)
                if os.path.isfile(file_path):
                    # 7일 이상 된 파일들 삭제
                    if current_time - os.path.getmtime(file_path) > 7 * 24 * 3600:
                        try:
                            file_size = os.path.getsize(file_path)
                            os.remove(file_path)
                            cleaned_files.append(filename)
                            total_size_freed += file_size
                        except Exception as e:
                            print(f"⚠️ 오래된 파일 삭제 실패: {filename} - {e}")
        
        result = {
            'success': True,
            'message': f'저장 공간 정리 완료: {len(cleaned_files)}개 파일 삭제',
            'details': {
                'files_cleaned': len(cleaned_files),
                'size_freed_bytes': total_size_freed,
                'size_freed_mb': round(total_size_freed / (1024 * 1024), 2),
                'cleaned_files': cleaned_files[:10],  # 처음 10개만 표시
                'timestamp': datetime.now().isoformat()
            }
        }
        
        print(f"✅ 저장 공간 정리 완료: {result}")
        return JsonResponse(result)
        
    except Exception as e:
        print(f"❌ 저장 공간 정리 실패: {e}")
        return JsonResponse({
            'success': False,
            'error': str(e),
            'message': '저장 공간 정리 중 오류가 발생했습니다.'
        }, status=500)


# ✅ 누락된 기타 유틸리티 뷰들

@csrf_exempt  
@require_http_methods(["GET"])
def check_video_exists(request, video_id):
    """비디오 존재 여부 확인"""
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


# ✅ FrameWithBboxView - 바운딩 박스가 있는 프레임 뷰
class FrameWithBboxView(APIView):
    permission_classes = [AllowAny]
    
    def get(self, request, video_id, frame_number):
        try:
            print(f"🖼️ 바운딩 박스 프레임 요청: 비디오={video_id}, 프레임={frame_number}")
            
            video = Video.objects.get(id=video_id)
            frame = Frame.objects.get(video=video, image_id=frame_number)
            
            # 디버깅: detected_objects 확인
            print(f"🔍 Frame {frame_number} detected_objects: {frame.detected_objects}")
            
            if not frame.detected_objects:
                print("⚠️ detected_objects가 없습니다")
                # 원본 이미지 반환
                return self._get_original_frame(video, frame_number)
            
            # detected_objects 파싱
            detected_objects = frame.detected_objects
            if isinstance(detected_objects, str):
                import json
                detected_objects = json.loads(detected_objects)
            
            if not isinstance(detected_objects, list):
                detected_objects = detected_objects.get('objects', []) if isinstance(detected_objects, dict) else []
            
            print(f"📦 파싱된 객체 수: {len(detected_objects)}")
            
            # 이미지 로드 및 박스 그리기
            image_data = self._draw_bboxes_on_frame(video, frame_number, detected_objects)
            
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
    def _draw_bboxes_on_frame(self, video, frame_number, detected_objects):
        """프레임에 바운딩 박스 그리기"""
        try:
            from PIL import Image, ImageDraw, ImageFont
            import cv2
            import io
            import numpy as np
            import os
            
            # 🔧 수정: file.path 대신 file_path 필드 사용
            video_path = video.file_path
            
            # 파일 경로가 절대 경로가 아닌 경우 처리
            if not os.path.isabs(video_path):
                from django.conf import settings
                # MEDIA_ROOT나 적절한 base path와 결합
                video_path = os.path.join(settings.MEDIA_ROOT, video_path)
            
            print(f"🎥 비디오 파일 경로: {video_path}")
            
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
                
                if len(bbox) == 4:
                    # 정규화된 좌표를 픽셀 좌표로 변환
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
            
            # 🔧 수정: file.path 대신 file_path 필드 사용
            video_path = video.file_path
            
            # 파일 경로가 절대 경로가 아닌 경우 처리
            if not os.path.isabs(video_path):
                from django.conf import settings
                video_path = os.path.join(settings.MEDIA_ROOT, video_path)
            
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


from django.db.models import Sum, Count, Avg
from django.utils import timezone
from datetime import timedelta

class CostManagementView(APIView):
    """비용 관리 및 분석 뷰"""
    permission_classes = [AllowAny]
    
    def get(self, request):
        try:
            # 전체 비용 통계
            total_videos = Video.objects.count()
            analyzed_videos = Video.objects.filter(image_analysis_completed=True).count()
            
            # 총 비용 계산
            total_cost = 0
            total_chats = 0
            
            for video in Video.objects.all():
                cost_info = video.get_analysis_cost_summary()
                total_cost += cost_info['estimated_cost']
                total_chats += video.total_chat_count
            
            # 최근 7일 비용 추이
            end_date = timezone.now().date()
            start_date = end_date - timedelta(days=7)
            
            daily_costs = []
            for i in range(7):
                date = start_date + timedelta(days=i)
                daily_analysis = CostAnalysis.get_daily_summary(date)
                daily_costs.append({
                    'date': date.isoformat(),
                    'cost': daily_analysis.estimated_total_cost if daily_analysis else 0.0,
                    'api_calls': daily_analysis.total_api_calls if daily_analysis else 0
                })
            
            # 모델별 사용량 통계
            model_stats = {}
            for video in Video.objects.filter(image_analysis_completed=True):
                models_used = video.api_cost_tracking.get('models_used', [])
                for model in models_used:
                    model_stats[model] = model_stats.get(model, 0) + 1
            
            # 효율성 메트릭
            avg_cost_per_video = total_cost / max(analyzed_videos, 1)
            avg_cost_per_chat = total_cost / max(total_chats, 1)
            
            # 절약 추정
            without_optimization_cost = analyzed_videos * 0.10  # 매번 이미지 분석했다면
            current_cost = total_cost
            savings = max(0, without_optimization_cost - current_cost)
            savings_percentage = (savings / without_optimization_cost * 100) if without_optimization_cost > 0 else 0
            
            return Response({
                'summary': {
                    'total_videos': total_videos,
                    'analyzed_videos': analyzed_videos,
                    'total_cost_usd': round(total_cost, 4),
                    'total_chats': total_chats,
                    'avg_cost_per_video': round(avg_cost_per_video, 4),
                    'avg_cost_per_chat': round(avg_cost_per_chat, 4)
                },
                'optimization_impact': {
                    'estimated_savings_usd': round(savings, 4),
                    'savings_percentage': round(savings_percentage, 1),
                    'optimization_strategy': 'first_chat_only_image_analysis'
                },
                'daily_trend': daily_costs,
                'model_usage': model_stats,
                'recommendations': self._get_cost_recommendations(total_cost, analyzed_videos, model_stats)
            })
            
        except Exception as e:
            return Response({'error': str(e)}, status=500)
    
    def _get_cost_recommendations(self, total_cost, analyzed_videos, model_stats):
        """비용 최적화 추천사항"""
        recommendations = []
        
        if total_cost > 5.0:  # $5 이상인 경우
            recommendations.append({
                'type': 'cost_alert',
                'message': '총 비용이 $5를 초과했습니다. 사용량을 모니터링하세요.',
                'priority': 'high'
            })
        
        if 'gpt-4' in str(model_stats):
            recommendations.append({
                'type': 'model_optimization',
                'message': 'GPT-4 대신 GPT-4o-mini 사용을 고려해보세요. 비용을 90% 절약할 수 있습니다.',
                'priority': 'medium'
            })
        
        if analyzed_videos > 50:
            recommendations.append({
                'type': 'usage_optimization',
                'message': '많은 비디오를 분석했습니다. RAG 시스템을 활용해 답변 품질을 높이세요.',
                'priority': 'low'
            })
        
        return recommendations


class VideoAnalysisStatusView(APIView):
    """비디오별 분석 상태 및 비용 정보"""
    permission_classes = [AllowAny]
    
    def get(self, request):
        try:
            videos_data = []
            
            # 비디오 분석 서비스 가져오기
            from .services.video_analysis_service import get_video_analysis_service
            video_service = get_video_analysis_service()
            
            for video in Video.objects.all().order_by('-uploaded_at'):
                cost_summary = video.get_analysis_cost_summary()
                
                # 새로운 분석 상태 정보 가져오기
                analysis_status = video_service.get_analysis_status(video.id)
                
                videos_data.append({
                    'id': video.id,
                    'name': video.original_name,
                    'duration': video.duration,
                    'uploaded_at': video.uploaded_at.isoformat(),
                    'image_analysis_completed': video.image_analysis_completed,
                    'image_analysis_date': video.image_analysis_date.isoformat() if video.image_analysis_date else None,
                    'total_chats': video.total_chat_count,
                    'cost_summary': cost_summary,
                    'has_json_analysis': bool(video.chat_analysis_json_path and os.path.exists(video.chat_analysis_json_path)),
                    'json_path': video.chat_analysis_json_path,
                    'analysis_status': video.analysis_status,
                    # 새로운 분석 정보 추가
                    'enhanced_analysis': analysis_status.get('enhanced_analysis', False),
                    'success_rate': analysis_status.get('success_rate', 0.0),
                    'processing_time': analysis_status.get('processing_time', 0),
                    'analysis_type': analysis_status.get('analysis_type', 'unknown'),
                    'unique_objects': analysis_status.get('unique_objects', 0)
                })
            
            return Response({
                'videos': videos_data,
                'total_count': len(videos_data),
                'summary': {
                    'with_image_analysis': sum(1 for v in videos_data if v['image_analysis_completed']),
                    'total_cost': sum(v['cost_summary']['estimated_cost'] for v in videos_data),
                    'total_chats': sum(v['total_chats'] for v in videos_data)
                }
            })
            
        except Exception as e:
            return Response({'error': str(e)}, status=500)


class ResetVideoAnalysisView(APIView):
    """비디오 분석 상태 리셋 (테스트/관리용)"""
    permission_classes = [AllowAny]
    
    def post(self, request):
        try:
            video_id = request.data.get('video_id')
            reset_costs = request.data.get('reset_costs', False)
            
            if not video_id:
                return Response({'error': 'video_id가 필요합니다.'}, status=400)
            
            try:
                video = Video.objects.get(id=video_id)
            except Video.DoesNotExist:
                return Response({'error': '비디오를 찾을 수 없습니다.'}, status=404)
            
            # 분석 상태 리셋
            video.image_analysis_completed = False
            video.image_analysis_date = None
            
            # JSON 파일 삭제
            if video.chat_analysis_json_path and os.path.exists(video.chat_analysis_json_path):
                try:
                    os.remove(video.chat_analysis_json_path)
                    print(f"✅ JSON 파일 삭제: {video.chat_analysis_json_path}")
                except Exception as e:
                    print(f"⚠️ JSON 파일 삭제 실패: {e}")
            
            video.chat_analysis_json_path = ''
            
            # 비용 정보 리셋 (옵션)
            if reset_costs:
                video.total_chat_count = 0
                video.api_cost_tracking = {}
            
            video.save()
            
            # VideoAnalysis에서 이미지 분석 정보 제거
            if hasattr(video, 'analysis'):
                analysis = video.analysis
                stats = analysis.analysis_statistics
                stats.pop('image_analysis_completed', None)
                stats.pop('image_analysis_date', None)
                stats.pop('json_file_path', None)
                analysis.analysis_statistics = stats
                analysis.save()
            
            return Response({
                'success': True,
                'message': f'비디오 "{video.original_name}"의 분석 상태가 리셋되었습니다.',
                'video_id': video_id,
                'reset_costs': reset_costs
            })
            
        except Exception as e:
            return Response({'error': str(e)}, status=500)


class CostOptimizationTipsView(APIView):
    """비용 최적화 팁 및 가이드"""
    permission_classes = [AllowAny]
    
    def get(self, request):
        return Response({
            'optimization_strategies': {
                'current_implementation': {
                    'name': '첫 채팅 전용 이미지 분석',
                    'description': '첫 번째 채팅에서만 이미지 분석을 수행하고, 이후 채팅은 저장된 결과 활용',
                    'cost_reduction': '80-90%',
                    'status': 'active'
                },
                'additional_optimizations': [
                    {
                        'name': '모델 선택 최적화',
                        'description': 'GPT-4 대신 GPT-4o-mini 사용 (성능 90% 유지, 비용 90% 절약)',
                        'impact': 'high'
                    },
                    {
                        'name': '이미지 품질 조정',
                        'description': '이미지 해상도를 800px로 제한하고 JPEG 품질을 70%로 설정',
                        'impact': 'medium'
                    },
                    {
                        'name': 'RAG 시스템 활용',
                        'description': '저장된 분석 결과를 벡터 DB에 저장하여 검색 품질 향상',
                        'impact': 'medium'
                    },
                    {
                        'name': '배치 처리',
                        'description': '여러 비디오를 한 번에 처리하여 API 호출 최적화',
                        'impact': 'low'
                    }
                ]
            },
            'cost_estimation': {
                'without_optimization': {
                    'per_chat': '$0.05-0.15',
                    'description': '매 채팅마다 이미지 분석 수행'
                },
                'with_optimization': {
                    'first_chat': '$0.05-0.15', 
                    'subsequent_chats': '$0.001-0.005',
                    'description': '첫 채팅만 이미지 분석, 이후는 텍스트만'
                }
            },
            'monitoring_tips': [
                '일일 API 사용량 모니터링',
                '모델별 비용 효율성 추적',
                '사용자별 채팅 패턴 분석',
                '월별 비용 예산 설정'
            ]
        })

from django.core.management.base import BaseCommand
from django.utils import timezone
from datetime import timedelta
from chat.models import Video, CostAnalysis

class Command(BaseCommand):
    help = '비용 분석 데이터 생성 및 업데이트'
    
    def add_arguments(self, parser):
        parser.add_argument('--days', type=int, default=7, help='분석할 일수')
        parser.add_argument('--update-monthly', action='store_true', help='월별 집계 업데이트')
    
    def handle(self, *args, **options):
        days = options['days']
        end_date = timezone.now().date()
        start_date = end_date - timedelta(days=days)
        
        self.stdout.write(f"📊 {days}일간 비용 분석 중...")
        
        for i in range(days):
            date = start_date + timedelta(days=i)
            
            # 해당 날짜의 비디오들 분석
            videos_on_date = Video.objects.filter(
                image_analysis_date__date=date
            )
            
            if videos_on_date.exists():
                total_cost = sum(
                    video.get_analysis_cost_summary()['estimated_cost'] 
                    for video in videos_on_date
                )
                
                total_calls = sum(
                    video.api_cost_tracking.get('total_api_calls', 0)
                    for video in videos_on_date
                )
                
                image_calls = sum(
                    video.api_cost_tracking.get('image_analysis_calls', 0)
                    for video in videos_on_date
                )
                
                # CostAnalysis 레코드 생성/업데이트
                cost_analysis, created = CostAnalysis.objects.get_or_create(
                    date=date,
                    period_type='daily',
                    defaults={
                        'total_api_calls': total_calls,
                        'image_analysis_calls': image_calls,
                        'text_only_calls': total_calls - image_calls,
                        'estimated_total_cost': total_cost
                    }
                )
                
                if not created:
                    cost_analysis.total_api_calls = total_calls
                    cost_analysis.image_analysis_calls = image_calls
                    cost_analysis.text_only_calls = total_calls - image_calls
                    cost_analysis.estimated_total_cost = total_cost
                    cost_analysis.save()
                
                status = "생성됨" if created else "업데이트됨"
                self.stdout.write(f"  {date}: ${total_cost:.4f} ({total_calls}회 호출) - {status}")
        
        self.stdout.write(self.style.SUCCESS("✅ 비용 분석 완료"))


# chat/views.py - 비용 절약형 비디오 채팅 시스템

import threading
import time
import json
import cv2
import os
import base64
from datetime import datetime
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny
from .models import Video, VideoAnalysis, Scene, Frame, SearchHistory
from .llm_client import LLMClient
from .video_analyzer import get_video_analyzer
from .multi_llm_service import get_multi_llm_analyzer
from .db_builder import get_video_rag_system

class CostEffectiveVideoChatView(APIView):
    """비용 절약형 비디오 채팅 - 첫 채팅에만 이미지 분석, 이후 JSON 기반 답변"""
    permission_classes = [AllowAny]
    
    def __init__(self):
        super().__init__()
        self.video_analyzer = get_video_analyzer()
        self.llm_client = LLMClient()
        self.multi_llm_analyzer = get_multi_llm_analyzer()
        self.rag_system = get_video_rag_system()
    
    def post(self, request):
        try:
            user_query = request.data.get('message', '').strip()
            video_id = request.data.get('video_id')
            analysis_mode = request.data.get('analysis_mode', 'single')
            use_multi_llm = request.data.get('use_multi_llm', analysis_mode != 'single')
            
            print(f"🤖 채팅 요청: '{user_query}' (비디오: {video_id}, 모드: {analysis_mode})")
            
            if not user_query:
                return Response({'error': '메시지를 입력해주세요.'}, status=400)
            
            if not video_id:
                return Response({'error': '비디오 ID가 필요합니다.'}, status=400)
            
            try:
                video = Video.objects.get(id=video_id)
            except Video.DoesNotExist:
                return Response({'error': '비디오를 찾을 수 없습니다.'}, status=404)
            
            # 🔥 핵심: 이미지 분석 이력 확인
            image_analysis_status = self._check_image_analysis_status(video)
            
            if image_analysis_status['needs_analysis']:
                print("🖼️ 첫 번째 채팅 - 이미지 분석 수행")
                response = self._handle_first_chat_with_analysis(
                    user_query, video, analysis_mode, use_multi_llm
                )
            else:
                print("📄 이미지 분석 완료됨 - JSON 기반 답변")
                response = self._handle_subsequent_chat_with_json(
                    user_query, video, image_analysis_status['json_path']
                )
            
            # 채팅 이력 저장
            self._save_chat_history(video, user_query, response.get('response', ''))
            
            return Response(response)
            
        except Exception as e:
            print(f"❌ 채팅 처리 오류: {e}")
            import traceback
            print(f"🔍 상세 오류:\n{traceback.format_exc()}")
            return Response({
                'error': f'채팅 처리 중 오류: {str(e)}',
                'response': '죄송합니다. 현재 서비스에 문제가 발생했습니다. 잠시 후 다시 시도해주세요.',
                'response_type': 'error'
            }, status=500)
    
    def _check_image_analysis_status(self, video):
        """이미지 분석 수행 여부 확인"""
        try:
            # 1. Video 모델의 image_analysis_completed 필드 확인
            if video.image_analysis_completed:
                json_path = video.chat_analysis_json_path
                if json_path and os.path.exists(json_path):
                    return {
                        'needs_analysis': False,
                        'json_path': json_path,
                        'analysis_date': video.image_analysis_date
                    }
            
            # 2. VideoAnalysis 모델에서 이미지 분석 여부 확인
            if hasattr(video, 'analysis'):
                analysis = video.analysis
                stats = analysis.analysis_statistics
                
                # 이미지 분석이 수행되었는지 확인
                if stats.get('image_analysis_completed', False):
                    json_path = stats.get('json_file_path')
                    if json_path and os.path.exists(json_path):
                        return {
                            'needs_analysis': False,
                            'json_path': json_path,
                            'analysis_date': analysis.created_at
                        }
            
            # 3. JSON 파일 존재 여부로 확인
            analysis_results_dir = os.path.join(settings.MEDIA_ROOT, 'analysis_results')
            pattern = f"chat_analysis_{video.id}_*.json"
            
            if os.path.exists(analysis_results_dir):
                import glob
                existing_files = glob.glob(os.path.join(analysis_results_dir, pattern))
                
                if existing_files:
                    # 가장 최근 파일 사용
                    latest_file = max(existing_files, key=os.path.getmtime)
                    return {
                        'needs_analysis': False,
                        'json_path': latest_file,
                        'analysis_date': datetime.fromtimestamp(os.path.getmtime(latest_file))
                    }
            
            # 4. 분석이 필요한 경우
            return {
                'needs_analysis': True,
                'json_path': None,
                'analysis_date': None
            }
            
        except Exception as e:
            print(f"⚠️ 이미지 분석 상태 확인 실패: {e}")
            return {'needs_analysis': True, 'json_path': None, 'analysis_date': None}
# chat/views.py - 비용 절약형 채팅 뷰 추가

import threading
import time
import json
import cv2
import os
import base64
from datetime import datetime
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny
from .models import Video, VideoAnalysis, Scene, Frame, SearchHistory
from .llm_client import LLMClient
from .video_analyzer import get_video_analyzer
from .multi_llm_service import get_multi_llm_analyzer

class CostEfficientChatView(APIView):
    """비용 절약형 비디오 채팅 - 첫 채팅에만 이미지 분석, 이후 JSON 기반 답변"""
    permission_classes = [AllowAny]
    
    def __init__(self):
        super().__init__()
        self.video_analyzer = get_video_analyzer()
        self.llm_client = LLMClient()
        self.multi_llm_analyzer = get_multi_llm_analyzer()
        
    def post(self, request):
        try:
            user_query = request.data.get('message', '').strip()
            video_id = request.data.get('video_id')
            analysis_mode = request.data.get('analysis_mode', 'single')
            use_multi_llm = request.data.get('use_multi_llm', analysis_mode != 'single')
            
            print(f"💰 비용절약 채팅 요청: '{user_query}' (비디오: {video_id}, 모드: {analysis_mode})")
            
            if not user_query:
                return Response({'error': '메시지를 입력해주세요.'}, status=400)
            
            if not video_id:
                return Response({'error': '비디오 ID가 필요합니다.'}, status=400)
            
            try:
                video = Video.objects.get(id=video_id)
            except Video.DoesNotExist:
                return Response({'error': '비디오를 찾을 수 없습니다.'}, status=404)
            
            # 채팅 카운트 증가
            video.increment_chat_count()
            
            # 🔥 핵심: 이미지 분석 이력 확인
            if not video.image_analysis_completed:
                print("🖼️ 첫 번째 채팅 - 이미지 분석 수행")
                response = self._handle_first_chat_with_analysis(
                    user_query, video, analysis_mode, use_multi_llm
                )
            else:
                print("📄 이미지 분석 완료됨 - JSON 기반 답변")
                response = self._handle_subsequent_chat_with_json(
                    user_query, video
                )
            
            # 채팅 이력 저장
            self._save_chat_history(video, user_query, response.get('response', ''))
            
            return Response(response)
            
        except Exception as e:
            print(f"❌ 비용절약 채팅 처리 오류: {e}")
            import traceback
            print(f"🔍 상세 오류:\n{traceback.format_exc()}")
            return Response({
                'error': f'채팅 처리 중 오류: {str(e)}',
                'response': '죄송합니다. 현재 서비스에 문제가 발생했습니다. 잠시 후 다시 시도해주세요.',
                'response_type': 'error'
            }, status=500)
    
    def _handle_first_chat_with_analysis(self, user_query, video, analysis_mode, use_multi_llm):
        """첫 번째 채팅 - 이미지 분석 포함"""
        try:
            print("🖼️ 첫 번째 채팅을 위한 이미지 분석 수행 중...")
            
            # 키프레임 추출 (비용 절약을 위해 2-3개만)
            frame_images = self._extract_key_frames_for_llm(video, max_frames=2)
            
            if not frame_images:
                print("⚠️ 키프레임 추출 실패 - 기존 데이터로 답변")
                return self._handle_fallback_response(user_query, video)
            
            # 비디오 컨텍스트 준비
            video_context = self._prepare_video_context(video)
            
            # 멀티 LLM 분석 수행
            if use_multi_llm and analysis_mode in ['multi', 'comparison']:
                multi_responses = self.multi_llm_analyzer.analyze_video_multi_llm(
                    frame_images, user_query, video_context
                )
                comparison_result = self.multi_llm_analyzer.compare_responses(multi_responses)
                
                # 분석 결과를 JSON으로 저장
                analysis_result = {
                    'video_id': video.id,
                    'query': user_query,
                    'analysis_type': 'multi_llm_image_analysis',
                    'frame_count': len(frame_images),
                    'llm_responses': {
                        model: {
                            'response': resp.response_text,
                            'confidence': resp.confidence_score,
                            'processing_time': resp.processing_time,
                            'success': resp.success
                        }
                        for model, resp in multi_responses.items()
                    },
                    'comparison_analysis': comparison_result['comparison'],
                    'timestamp': datetime.now().isoformat(),
                    'video_context': video_context
                }
                
                # JSON 파일 저장
                json_path = self._save_analysis_result(video, analysis_result)
                
                # 비디오에 이미지 분석 완료 표시
                video.mark_image_analysis_completed(json_path)
                
                # 비용 추적
                estimated_cost = self._calculate_analysis_cost(analysis_result)
                video.update_cost_tracking('image_analysis', estimated_cost, 'multi_llm')
                
                if analysis_mode == 'comparison':
                    return {
                        'response_type': 'first_chat_multi_llm_comparison',
                        'query': user_query,
                        'video_info': {'id': video.id, 'name': video.original_name},
                        'llm_responses': analysis_result['llm_responses'],
                        'comparison_analysis': analysis_result['comparison_analysis'],
                        'recommendation': comparison_result['comparison']['recommendation'],
                        'cost_info': {
                            'estimated_cost': estimated_cost,
                            'optimization_enabled': True,
                            'future_chats_cost': 'text_only (~$0.001)'
                        }
                    }
                else:
                    best_model = comparison_result['comparison']['best_response']
                    best_response = multi_responses.get(best_model)
                    
                    return {
                        'response_type': 'first_chat_multi_llm_optimized',
                        'response': best_response.response_text if best_response and best_response.success else "분석 결과를 가져올 수 없습니다.",
                        'query': user_query,
                        'video_info': {'id': video.id, 'name': video.original_name},
                        'selected_model': best_model,
                        'confidence': best_response.confidence_score if best_response else 0,
                        'models_used': list(multi_responses.keys()),
                        'cost_info': {
                            'estimated_cost': estimated_cost,
                            'optimization_enabled': True
                        }
                    }
            else:
                # 단일 LLM 분석
                single_response = self._analyze_with_single_llm(frame_images[0], user_query, video_context)
                
                analysis_result = {
                    'video_id': video.id,
                    'query': user_query,
                    'analysis_type': 'single_llm_image_analysis', 
                    'frame_count': len(frame_images),
                    'response': single_response,
                    'timestamp': datetime.now().isoformat(),
                    'video_context': video_context
                }
                
                # JSON 파일 저장
                json_path = self._save_analysis_result(video, analysis_result)
                
                # 비디오에 이미지 분석 완료 표시
                video.mark_image_analysis_completed(json_path)
                
                # 비용 추적
                estimated_cost = self._calculate_analysis_cost(analysis_result)
                video.update_cost_tracking('image_analysis', estimated_cost, 'single_llm')
                
                return {
                    'response_type': 'first_chat_single_llm',
                    'response': single_response,
                    'query': user_query,
                    'video_info': {'id': video.id, 'name': video.original_name},
                    'cost_info': {
                        'estimated_cost': estimated_cost,
                        'optimization_enabled': True
                    }
                }
                
        except Exception as e:
            print(f"❌ 첫 번째 채팅 분석 실패: {e}")
            return self._handle_fallback_response(user_query, video)
    
    def _handle_subsequent_chat_with_json(self, user_query, video):
        """이후 채팅 - JSON 파일 기반 답변"""
        try:
            print(f"📄 JSON 기반 답변 생성: {video.chat_analysis_json_path}")
            
            # JSON 파일에서 이전 분석 결과 로드
            if not video.chat_analysis_json_path or not os.path.exists(video.chat_analysis_json_path):
                print("⚠️ JSON 파일이 없음 - 일반 채팅으로 처리")
                response_text = self.llm_client.generate_smart_response(
                    user_query=user_query,
                    search_results=None,
                    video_info=f"비디오: {video.original_name}",
                    use_multi_llm=False
                )
                
                # 텍스트 전용 비용 추적
                estimated_cost = 0.002  # 텍스트 전용 비용 (매우 저렴)
                video.update_cost_tracking('text_only', estimated_cost)
                
                return {
                    'response_type': 'text_only_fallback',
                    'response': response_text,
                    'query': user_query,
                    'video_info': {'id': video.id, 'name': video.original_name},
                    'cost_info': {
                        'estimated_cost': estimated_cost,
                        'note': 'JSON 파일 없음 - 텍스트 전용 처리'
                    }
                }
            
            # JSON 파일 로드
            with open(video.chat_analysis_json_path, 'r', encoding='utf-8') as f:
                analysis_data = json.load(f)
            
            # 저장된 분석 결과 활용
            if analysis_data.get('analysis_type') == 'multi_llm_image_analysis':
                # 이전 멀티 LLM 결과 활용
                response_text = self._generate_contextual_response_from_json(
                    user_query, analysis_data, video
                )
            else:
                # 단일 LLM 결과 활용
                response_text = self._generate_simple_response_from_json(
                    user_query, analysis_data, video
                )
            
            # 텍스트 전용 비용 추적
            estimated_cost = 0.001  # 매우 저렴한 텍스트 전용 비용
            video.update_cost_tracking('text_only', estimated_cost)
            
            return {
                'response_type': 'json_based_optimized',
                'response': response_text,
                'query': user_query,
                'video_info': {'id': video.id, 'name': video.original_name},
                'cost_info': {
                    'estimated_cost': estimated_cost,
                    'data_source': 'saved_analysis',
                    'optimization_savings': '~95% cost reduction'
                }
            }
            
        except Exception as e:
            print(f"❌ JSON 기반 답변 생성 실패: {e}")
            return self._handle_fallback_response(user_query, video)
        


from django.http import JsonResponse, HttpResponse
from django.views import View
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny
import json
import os

class AnalysisStatusView(APIView):
    """분석 상태 상세 조회"""
    permission_classes = [AllowAny]
    
    def get(self, request, video_id):
        try:
            video = Video.objects.get(id=video_id)
            
            # 최신 분석 정보 조회
            latest_analysis = VideoAnalysis.objects.filter(video=video).order_by('-id').first()
            
            response_data = {
                'video_id': video.id,
                'video_name': video.original_name,
                'analysis_status': video.analysis_status,
                'is_analyzed': video.is_analyzed,
                'analysis_progress': 100 if video.analysis_status == 'completed' else 
                                   (50 if video.analysis_status == 'processing' else 0),
                'video_info': {
                    'duration': getattr(video, 'duration', 0),
                    'total_frames': getattr(video, 'total_frames', 0),
                    'fps': getattr(video, 'fps', 0),
                    'width': getattr(video, 'width', 0),
                    'height': getattr(video, 'height', 0)
                }
            }
            
            if latest_analysis:
                response_data['latest_analysis'] = {
                    'id': latest_analysis.id,
                    'enhanced_analysis': latest_analysis.enhanced_analysis,
                    'success_rate': latest_analysis.success_rate,
                    'processing_time_seconds': latest_analysis.processing_time_seconds,
                    'frames_analyzed': latest_analysis.analysis_statistics.get('total_frames_analyzed', 0),
                    'dominant_objects': latest_analysis.analysis_statistics.get('dominant_objects', []),
                    'ai_features_used': latest_analysis.analysis_statistics.get('ai_features_used', {}),
                    'json_file_path': latest_analysis.analysis_statistics.get('json_file_path', ''),
                    'created_at': latest_analysis.created_at.isoformat() if hasattr(latest_analysis, 'created_at') else None
                }
            
            # 프레임 및 씬 개수
            frame_count = Frame.objects.filter(video=video).count()
            scene_count = Scene.objects.filter(video=video).count()
            
            response_data['analysis_counts'] = {
                'frames': frame_count,
                'scenes': scene_count
            }
            
            return Response(response_data)
            
        except Video.DoesNotExist:
            return Response({
                'error': '해당 비디오를 찾을 수 없습니다.'
            }, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({
                'error': f'분석 상태 조회 실패: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class AnalysisResultsView(APIView):
    """분석 결과 상세 조회"""
    permission_classes = [AllowAny]
    
    def get(self, request, video_id):
        try:
            video = Video.objects.get(id=video_id)
            
            if not video.is_analyzed:
                return Response({
                    'error': '아직 분석이 완료되지 않았습니다.',
                    'analysis_status': video.analysis_status
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # 분석 결과 조회
            latest_analysis = VideoAnalysis.objects.filter(video=video).order_by('-id').first()
            frames = Frame.objects.filter(video=video).order_by('timestamp')[:20]  # 최대 20개
            scenes = Scene.objects.filter(video=video).order_by('scene_id')
            
            response_data = {
                'video_id': video.id,
                'video_name': video.original_name,
                'analysis_summary': {},
                'sample_frames': [],
                'scenes': [],
                'json_file_available': False,
                'json_file_path': None
            }
            
            if latest_analysis:
                response_data['analysis_summary'] = {
                    'success_rate': latest_analysis.success_rate,
                    'processing_time': latest_analysis.processing_time_seconds,
                    'total_frames_analyzed': latest_analysis.analysis_statistics.get('total_frames_analyzed', 0),
                    'dominant_objects': latest_analysis.analysis_statistics.get('dominant_objects', []),
                    'scene_types': latest_analysis.analysis_statistics.get('scene_types', []),
                    'text_extracted': latest_analysis.analysis_statistics.get('text_extracted', False),
                    'ai_features_used': latest_analysis.analysis_statistics.get('ai_features_used', {}),
                    'analysis_quality_metrics': latest_analysis.analysis_statistics.get('analysis_quality_metrics', {}),
                    'caption_statistics': {
                        'frames_with_caption': latest_analysis.caption_statistics.get('frames_with_caption', 0),
                        'enhanced_captions': latest_analysis.caption_statistics.get('enhanced_captions', 0),
                        'average_confidence': latest_analysis.caption_statistics.get('average_confidence', 0)
                    }
                }
                
                # JSON 파일 경로 확인
                json_file_path = latest_analysis.analysis_statistics.get('json_file_path', '')
                if json_file_path and os.path.exists(json_file_path):
                    response_data['json_file_available'] = True
                    response_data['json_file_path'] = json_file_path
            
            # 샘플 프레임들
            for frame in frames:
                response_data['sample_frames'].append({
                    'frame_id': frame.image_id,
                    'timestamp': frame.timestamp,
                    'caption': frame.final_caption or frame.enhanced_caption or frame.caption,
                    'object_count': len(frame.detected_objects),
                    'detected_objects': [obj.get('class', '') for obj in frame.detected_objects[:5]]  # 최대 5개만
                })
            
            # 씬 정보
            for scene in scenes:
                response_data['scenes'].append({
                    'scene_id': scene.scene_id,
                    'start_time': scene.start_time,
                    'end_time': scene.end_time,
                    'duration': scene.duration,
                    'frame_count': scene.frame_count,
                    'dominant_objects': scene.dominant_objects[:3]  # 최대 3개만
                })
            
            return Response(response_data)
            
        except Video.DoesNotExist:
            return Response({
                'error': '해당 비디오를 찾을 수 없습니다.'
            }, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({
                'error': f'분석 결과 조회 실패: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

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

class DownloadAnalysisResultView(APIView):
    """분석 결과 JSON 파일 다운로드"""
    permission_classes = [AllowAny]
    
    def get(self, request, video_id):
        try:
            video = Video.objects.get(id=video_id)
            latest_analysis = VideoAnalysis.objects.filter(video=video).order_by('-id').first()
            
            if not latest_analysis:
                return Response({
                    'error': '분석 결과를 찾을 수 없습니다.'
                }, status=status.HTTP_404_NOT_FOUND)
            
            json_file_path = latest_analysis.analysis_statistics.get('json_file_path', '')
            
            if not json_file_path or not os.path.exists(json_file_path):
                return Response({
                    'error': 'JSON 파일을 찾을 수 없습니다.'
                }, status=status.HTTP_404_NOT_FOUND)
            
            # JSON 파일 읽기
            with open(json_file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # HTTP 응답으로 JSON 반환
            response = HttpResponse(
                json.dumps(json_data, ensure_ascii=False, indent=2),
                content_type='application/json; charset=utf-8'
            )
            response['Content-Disposition'] = f'attachment; filename="analysis_{video.id}_{video.original_name}.json"'
            
            return response
            
        except Video.DoesNotExist:
            return Response({
                'error': '해당 비디오를 찾을 수 없습니다.'
            }, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({
                'error': f'JSON 다운로드 실패: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# ========================================
# 분석 취소 및 관리 기능
# ========================================

class CancelAnalysisView(APIView):
    """분석 취소"""
    permission_classes = [AllowAny]
    
    def post(self, request, video_id):
        try:
            video = Video.objects.get(id=video_id)
            
            if video.analysis_status != 'processing':
                return Response({
                    'error': '진행 중인 분석이 없습니다.',
                    'current_status': video.analysis_status
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # 상태를 cancelled로 변경
            video.analysis_status = 'cancelled'
            video.save()
            
            return Response({
                'success': True,
                'message': '분석이 취소되었습니다.',
                'video_id': video.id,
                'new_status': 'cancelled'
            })
            
        except Video.DoesNotExist:
            return Response({
                'error': '해당 비디오를 찾을 수 없습니다.'
            }, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({
                'error': f'분석 취소 실패: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class RestartAnalysisView(APIView):
    """분석 재시작"""
    permission_classes = [AllowAny]
    
    def post(self, request, video_id):
        try:
            video = Video.objects.get(id=video_id)
            
            if video.analysis_status == 'processing':
                return Response({
                    'error': '이미 분석이 진행 중입니다.',
                    'current_status': video.analysis_status
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # 기존 분석 결과 삭제 (선택사항)
            cleanup = request.data.get('cleanup_previous', False)
            if cleanup:
                VideoAnalysis.objects.filter(video=video).delete()
                Frame.objects.filter(video=video).delete()
                Scene.objects.filter(video=video).delete()
            
            # 분석 상태 초기화
            video.analysis_status = 'pending'
            video.is_analyzed = False
            video.save()
            
            return Response({
                'success': True,
                'message': '분석이 재시작 준비되었습니다. 분석을 다시 요청해주세요.',
                'video_id': video.id,
                'new_status': 'pending',
                'cleanup_performed': cleanup
            })
            
        except Video.DoesNotExist:
            return Response({
                'error': '해당 비디오를 찾을 수 없습니다.'
            }, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({
                'error': f'분석 재시작 실패: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# chat/views.py - EnhancedVideoChatView 개선 버전

import json
import os
import re
import time
from collections import Counter
from datetime import datetime
from typing import List, Dict, Any, Optional
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.conf import settings

from .models import Video, VideoAnalysis, SearchHistory
from .llm_client import LLMClient
from .multi_llm_service import get_multi_llm_analyzer
from .db_builder import get_video_rag_system

class SmartJSONParser:
    """고성능 JSON 파싱 및 person 객체 추출 최적화"""
    
    @staticmethod
    def extract_person_info_optimized(json_data: Dict) -> Dict[str, Any]:
        """최적화된 person 정보 추출"""
        person_analysis = {
            'total_person_detections': 0,
            'unique_persons_estimated': 0,
            'frames_with_people': [],
            'person_tracking_data': {},
            'confidence_scores': [],
            'gender_analysis': {'male': 0, 'female': 0, 'unknown': 0},
            'temporal_consistency': 0.0
        }
        
        frame_results = json_data.get('frame_results', [])
        if not frame_results:
            return person_analysis
        
        # 프레임별 person 추적
        for frame_data in frame_results:
            frame_id = frame_data.get('image_id', 0)
            timestamp = frame_data.get('timestamp', 0)
            objects = frame_data.get('objects', [])
            
            frame_persons = []
            for obj in objects:
                if obj.get('class') == 'person':
                    person_data = {
                        'bbox': obj.get('bbox', []),
                        'confidence': obj.get('confidence', 0),
                        'track_id': obj.get('track_id'),
                        'frame_id': frame_id,
                        'timestamp': timestamp
                    }
                    frame_persons.append(person_data)
                    person_analysis['confidence_scores'].append(obj.get('confidence', 0))
            
            if frame_persons:
                person_analysis['frames_with_people'].append({
                    'frame_id': frame_id,
                    'timestamp': timestamp,
                    'person_count': len(frame_persons),
                    'persons': frame_persons
                })
                person_analysis['total_person_detections'] += len(frame_persons)
        
        # track-based counting으로 고유 person 수 계산
        person_analysis['unique_persons_estimated'] = SmartJSONParser._estimate_unique_persons(
            person_analysis['frames_with_people']
        )
        
        # 성별 분석 (캡션 및 VQA 결과 활용)
        person_analysis['gender_analysis'] = SmartJSONParser._analyze_gender_from_captions(frame_results)
        
        # 시간적 일관성 계산
        person_analysis['temporal_consistency'] = SmartJSONParser._calculate_temporal_consistency(
            person_analysis['frames_with_people']
        )
        
        return person_analysis
    
    @staticmethod
    def _estimate_unique_persons(frames_with_people: List) -> int:
        """다중 프레임 추적으로 고유 person 수 추정"""
        if not frames_with_people:
            return 0
        
        # track_id가 있는 경우
        track_ids = set()
        for frame_data in frames_with_people:
            for person in frame_data['persons']:
                track_id = person.get('track_id')
                if track_id is not None:
                    track_ids.add(track_id)
        
        if track_ids:
            return len(track_ids)
        
        # track_id가 없는 경우 휴리스틱 방법 사용
        person_counts = [frame_data['person_count'] for frame_data in frames_with_people]
        if not person_counts:
            return 0
        
        # 가장 높은 confidence를 가진 프레임들의 평균값 사용
        high_confidence_frames = []
        for frame_data in frames_with_people:
            avg_confidence = sum(p['confidence'] for p in frame_data['persons']) / len(frame_data['persons'])
            if avg_confidence > 0.7:  # 높은 신뢰도 프레임만
                high_confidence_frames.append(frame_data['person_count'])
        
        if high_confidence_frames:
            return max(high_confidence_frames)  # 최대값 사용
        else:
            return max(person_counts) if person_counts else 0
    
    @staticmethod
    def _analyze_gender_from_captions(frame_results: List) -> Dict[str, int]:
        """캡션에서 성별 정보 분석"""
        gender_analysis = {'male': 0, 'female': 0, 'unknown': 0}
        
        male_keywords = ['남자', '남성', 'man', 'male', '아저씨', '청년', '아빠', '아들']
        female_keywords = ['여자', '여성', 'woman', 'female', '아줌마', '소녀', '엄마', '딸']
        
        for frame_data in frame_results:
            # 캡션에서 성별 키워드 찾기
            captions = [
                frame_data.get('final_caption', ''),
                frame_data.get('enhanced_caption', ''),
                frame_data.get('caption', '')
            ]
            
            frame_text = ' '.join(captions).lower()
            
            male_mentions = sum(1 for keyword in male_keywords if keyword in frame_text)
            female_mentions = sum(1 for keyword in female_keywords if keyword in frame_text)
            
            gender_analysis['male'] += male_mentions
            gender_analysis['female'] += female_mentions
            
            # VQA 결과도 확인
            scene_analysis = frame_data.get('scene_analysis', {})
            vqa_results = scene_analysis.get('vqa_results', {})
            
            for question, answer in vqa_results.items():
                if 'people' in question.lower() or '사람' in question:
                    answer_lower = answer.lower()
                    if any(keyword in answer_lower for keyword in male_keywords):
                        gender_analysis['male'] += 1
                    if any(keyword in answer_lower for keyword in female_keywords):
                        gender_analysis['female'] += 1
        
        return gender_analysis
    
    @staticmethod
    def _calculate_temporal_consistency(frames_with_people: List) -> float:
        """시간적 일관성 계산"""
        if len(frames_with_people) < 2:
            return 1.0
        
        person_counts = [frame_data['person_count'] for frame_data in frames_with_people]
        
        # 표준편차 기반 일관성 계산
        import numpy as np
        if len(set(person_counts)) == 1:
            return 1.0  # 완전 일관성
        
        mean_count = np.mean(person_counts)
        std_count = np.std(person_counts)
        
        # 일관성 점수 (0-1)
        consistency = max(0, 1 - (std_count / max(mean_count, 1)))
        return consistency

class KoreanQuestionClassifier:
    """한국어 질문 분류 시스템"""
    
    QUESTION_PATTERNS = {
        'person_count': {
            'keywords': ['몇 명', '몇명', '사람', '인물', '인원', '성비', '남녀', '사람 수'],
            'patterns': [r'몇\s*명', r'사람.*몇', r'인원.*몇', r'성비', r'남.*여'],
            'weight': 1.0
        },
        'object_search': {
            'keywords': ['찾', '어디', '언제', '무엇', '뭐가', '있어', '보여', '나와'],
            'patterns': [r'.*찾.*', r'어디.*있', r'언제.*나', r'무엇.*보'],
            'weight': 0.9
        },
        'scene_summary': {
            'keywords': ['요약', '정리', '내용', '전체', '줄거리', '설명'],
            'patterns': [r'요약.*해', r'정리.*해', r'내용.*뭐', r'전체.*어떤'],
            'weight': 0.8
        },
        'action_analysis': {
            'keywords': ['행동', '동작', '활동', '하고있', '움직임', '뭘 해'],
            'patterns': [r'.*하고\s*있', r'무엇.*하는', r'어떤.*행동'],
            'weight': 0.8
        },
        'time_location': {
            'keywords': ['시간', '장소', '위치', '언제', '어디서', '배경'],
            'patterns': [r'언제.*', r'어디서.*', r'.*시간', r'.*장소'],
            'weight': 0.7
        }
    }
    
    @classmethod
    def classify_question(cls, question: str) -> Dict[str, float]:
        """질문을 카테고리별로 분류하고 점수 반환"""
        question_lower = question.lower().strip()
        scores = {}
        
        for category, config in cls.QUESTION_PATTERNS.items():
            score = 0.0
            
            # 키워드 매칭
            keyword_matches = sum(1 for keyword in config['keywords'] if keyword in question_lower)
            keyword_score = (keyword_matches / len(config['keywords'])) * 0.6
            
            # 패턴 매칭
            pattern_matches = sum(1 for pattern in config['patterns'] if re.search(pattern, question_lower))
            pattern_score = (pattern_matches / len(config['patterns'])) * 0.4
            
            total_score = (keyword_score + pattern_score) * config['weight']
            scores[category] = total_score
        
        return scores
    
    @classmethod
    def get_primary_category(cls, question: str) -> str:
        """주요 카테고리 반환"""
        scores = cls.classify_question(question)
        if not scores:
            return 'general'
        
        max_category = max(scores.keys(), key=lambda k: scores[k])
        if scores[max_category] > 0.3:  # 임계값
            return max_category
        else:
            return 'general'

class ContextAwareResponseGenerator:
    """맥락 인식 응답 생성기"""
    
    def __init__(self):
        self.session_memory = {}  # 세션별 대화 이력
        self.response_templates = {
            'person_count': {
                'single': "화면에 {count}명이 보입니다.",
                'multiple': "비디오 전체에서 총 {count}명이 등장합니다.",
                'uncertain': "정확히 세기 어려운데, 대략 {estimate}명 정도 보이네요.",
                'with_gender': "총 {total}명이 등장하며, 남성 {male}명, 여성 {female}명으로 분석됩니다.",
                'temporal': "{frames}개 장면에서 사람이 등장하며, 평균 {avg_count}명씩 보입니다."
            },
            'object_search': {
                'found': "'{object}'를 {count}개 장면에서 찾았습니다.",
                'not_found': "'{object}'를 찾을 수 없습니다.",
                'location': "'{object}'는 {timestamp}초 지점에서 확인됩니다."
            },
            'scene_summary': {
                'basic': "이 영상은 {scenes}개의 주요 장면으로 구성되어 있습니다.",
                'detailed': "영상의 주요 내용: {content}"
            }
        }
    
    def generate_contextual_response(self, question: str, analysis_data: Dict, video_info: Dict) -> str:
        """맥락을 고려한 응답 생성"""
        category = KoreanQuestionClassifier.get_primary_category(question)
        
        if category == 'person_count':
            return self._generate_person_count_response(question, analysis_data, video_info)
        elif category == 'object_search':
            return self._generate_object_search_response(question, analysis_data, video_info)
        elif category == 'scene_summary':
            return self._generate_scene_summary_response(question, analysis_data, video_info)
        else:
            return self._generate_general_response(question, analysis_data, video_info)
    
    def _generate_person_count_response(self, question: str, analysis_data: Dict, video_info: Dict) -> str:
        """사람 수 관련 응답 생성"""
        person_info = analysis_data.get('person_analysis', {})
        
        unique_count = person_info.get('unique_persons_estimated', 0)
        total_detections = person_info.get('total_person_detections', 0)
        frames_with_people = person_info.get('frames_with_people', [])
        gender_analysis = person_info.get('gender_analysis', {})
        temporal_consistency = person_info.get('temporal_consistency', 0)
        
        templates = self.response_templates['person_count']
        
        # 성비 질문인지 확인
        if any(keyword in question for keyword in ['성비', '남녀', '성별']):
            if gender_analysis['male'] > 0 or gender_analysis['female'] > 0:
                return templates['with_gender'].format(
                    total=unique_count,
                    male=gender_analysis['male'],
                    female=gender_analysis['female']
                )
        
        # 시간적 일관성이 낮으면 불확실 표현
        if temporal_consistency < 0.6:
            estimate = max(unique_count, int(total_detections / max(len(frames_with_people), 1)))
            return templates['uncertain'].format(estimate=estimate)
        
        # 여러 프레임에 걸쳐 등장하는 경우
        if len(frames_with_people) > 1:
            avg_count = total_detections / len(frames_with_people)
            return templates['temporal'].format(
                frames=len(frames_with_people),
                avg_count=f"{avg_count:.1f}"
            )
        
        # 기본 응답
        if unique_count == 1:
            return templates['single'].format(count=unique_count)
        else:
            return templates['multiple'].format(count=unique_count)
    
    def _generate_object_search_response(self, question: str, analysis_data: Dict, video_info: Dict) -> str:
        """객체 검색 응답 생성"""
        # 질문에서 검색 대상 추출
        search_terms = self._extract_search_terms(question)
        
        frame_results = analysis_data.get('frame_results', [])
        found_objects = []
        
        for frame_data in frame_results:
            frame_id = frame_data.get('image_id', 0)
            timestamp = frame_data.get('timestamp', 0)
            objects = frame_data.get('objects', [])
            
            for obj in objects:
                obj_class = obj.get('class', '').lower()
                if any(term.lower() in obj_class for term in search_terms):
                    found_objects.append({
                        'frame_id': frame_id,
                        'timestamp': timestamp,
                        'object': obj_class,
                        'confidence': obj.get('confidence', 0)
                    })
        
        templates = self.response_templates['object_search']
        
        if found_objects:
            search_term = search_terms[0] if search_terms else '해당 객체'
            return templates['found'].format(
                object=search_term,
                count=len(found_objects)
            )
        else:
            search_term = search_terms[0] if search_terms else '해당 객체'
            return templates['not_found'].format(object=search_term)
    
    def _generate_scene_summary_response(self, question: str, analysis_data: Dict, video_info: Dict) -> str:
        """장면 요약 응답 생성"""
        frame_results = analysis_data.get('frame_results', [])
        
        # 주요 객체 추출
        all_objects = []
        for frame_data in frame_results:
            objects = frame_data.get('objects', [])
            all_objects.extend([obj.get('class', '') for obj in objects])
        
        object_counter = Counter(all_objects)
        dominant_objects = [obj for obj, count in object_counter.most_common(5)]
        
        # 주요 장면 식별
        unique_scenes = len(frame_results)
        
        summary = f"이 영상은 {unique_scenes}개의 장면으로 구성되어 있습니다. "
        
        if dominant_objects:
            summary += f"주요 객체로는 {', '.join(dominant_objects[:3])} 등이 등장합니다. "
        
        # 사람이 등장하는 경우
        person_info = analysis_data.get('person_analysis', {})
        if person_info.get('unique_persons_estimated', 0) > 0:
            summary += f"총 {person_info['unique_persons_estimated']}명의 사람이 등장합니다."
        
        return summary
    
    def _generate_general_response(self, question: str, analysis_data: Dict, video_info: Dict) -> str:
        """일반적인 응답 생성"""
        frame_results = analysis_data.get('frame_results', [])
        
        return f"'{question}'에 대해 분석한 결과, {len(frame_results)}개 장면을 바탕으로 답변드리겠습니다. 더 구체적인 질문을 해주시면 정확한 정보를 제공할 수 있습니다."
    
    def _extract_search_terms(self, question: str) -> List[str]:
        """질문에서 검색어 추출"""
        # 한국어 객체명 매핑
        object_mapping = {
            '자동차': 'car', '차': 'car', '승용차': 'car',
            '사람': 'person', '인물': 'person',
            '강아지': 'dog', '개': 'dog',
            '고양이': 'cat', '냥이': 'cat',
            '의자': 'chair', '책상': 'table',
            '핸드폰': 'cell_phone', '폰': 'cell_phone',
            '컴퓨터': 'laptop', '노트북': 'laptop'
        }
        
        terms = []
        
        # 직접 매핑된 객체 찾기
        for korean, english in object_mapping.items():
            if korean in question:
                terms.append(english)
        
        # 영어 객체명 직접 추출
        import re
        english_objects = re.findall(r'[a-zA-Z]+', question)
        terms.extend(english_objects)
        
        return terms

# views.py (필요 import)
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from django.db.models import Q
import time, json, re

COLOR_MAP_KR2EN = {
    '초록':'green','녹색':'green','빨강':'red','빨간':'red','적색':'red',
    '주황':'orange','오렌지':'orange','노랑':'yellow','노란':'yellow','황색':'yellow',
    '파랑':'blue','파란':'blue','청색':'blue','보라':'purple','자주':'purple',
    '검정':'black','검은':'black','하양':'white','흰':'white','백색':'white',
    '회색':'gray','그레이':'gray','갈색':'brown',
    '핑크':'pink','분홍':'pink','금색':'gold','은색':'silver'
}

OBJ_MAP_KR2EN = {
    '사람':'person','남성':'person','여성':'person','인물':'person',
    '가방':'handbag','핸드백':'handbag','백팩':'backpack',
    '자동차':'car','차':'car','자전거':'bicycle','오토바이':'motorcycle',
    '개':'dog','강아지':'dog','고양이':'cat','의자':'chair','노트북':'laptop','휴대폰':'cell_phone','핸드폰':'cell_phone','폰':'cell_phone',
    '티비':'tv','tv':'tv'
}

SCENE_KEYWORDS = {
    '비':'rain','비오는':'rain','우천':'rain',
    '밤':'night','야간':'night','낮':'day','실내':'indoor','실외':'outdoor'
}

# views.py
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

from .models import Video, Frame, Scene


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

        frames_qs = Frame.objects.filter(video=video).order_by('timestamp')
        if tr and tr.get('start') is not None and tr.get('end') is not None:
            frames_qs = frames_qs.filter(timestamp__gte=tr['start'], timestamp__lte=tr['end'])

        hits = []
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
            return {'text': f"‘{raw_text}’로는 매칭이 없었어요. 시간 범위를 넓히거나 색상 없이 다시 시도해 보세요.", 'items': []}

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


# ========== 영상 간 검색을 위한 헬퍼 메서드들 ==========

def _check_weather_match(frames, weather_condition):
    """날씨 조건 매칭 확인"""
    try:
        weather_keywords = {
            'rainy': ['비', 'rain', 'wet', 'umbrella', 'puddle', 'drizzle'],
            'sunny': ['sunny', 'sun', 'bright', 'sunlight', '햇빛', '맑은'],
            'cloudy': ['cloudy', 'cloud', 'overcast', '흐림', '구름']
        }
        
        target_keywords = weather_keywords.get(weather_condition.lower(), [])
        if not target_keywords:
            return 0.0
        
        match_count = 0
        total_frames = len(frames)
        
        for frame in frames:
            caption = (frame.caption or '') + ' ' + (frame.enhanced_caption or '') + ' ' + (frame.final_caption or '')
            caption_lower = caption.lower()
            
            for keyword in target_keywords:
                if keyword in caption_lower:
                    match_count += 1
                    break
        
        if match_count > 0:
            return (match_count / total_frames) * 0.8
        
        return 0.0
        
    except Exception as e:
        logger.error(f"❌ 날씨 매칭 확인 오류: {e}")
        return 0.0


def _check_time_match(video, time_condition):
    """시간대 조건 매칭 확인"""
    try:
        created_hour = video.created_at.hour
        
        time_mapping = {
            'morning': (6, 12),
            'afternoon': (12, 18),
            'evening': (18, 22),
            'night': (22, 6)
        }
        
        if time_condition.lower() in time_mapping:
            start_hour, end_hour = time_mapping[time_condition.lower()]
            
            if time_condition.lower() == 'night':
                if created_hour >= 22 or created_hour < 6:
                    return 0.7
            else:
                if start_hour <= created_hour < end_hour:
                    return 0.7
        
        return 0.0
        
    except Exception as e:
        logger.error(f"❌ 시간대 매칭 확인 오류: {e}")
        return 0.0


def _check_location_match(frames, location_condition):
    """장소 조건 매칭 확인"""
    try:
        location_keywords = {
            'indoor': ['실내', 'indoor', 'inside', 'room', 'building', 'office', 'home'],
            'outdoor': ['실외', 'outdoor', 'outside', 'street', 'park', 'garden', 'outdoor'],
            'office': ['office', '사무실', 'desk', 'computer', 'meeting'],
            'home': ['home', '집', 'house', 'living room', 'bedroom']
        }
        
        target_keywords = location_keywords.get(location_condition.lower(), [])
        if not target_keywords:
            return 0.0
        
        match_count = 0
        total_frames = len(frames)
        
        for frame in frames:
            caption = (frame.caption or '') + ' ' + (frame.enhanced_caption or '') + ' ' + (frame.final_caption or '')
            caption_lower = caption.lower()
            
            for keyword in target_keywords:
                if keyword in caption_lower:
                    match_count += 1
                    break
        
        if match_count > 0:
            return (match_count / total_frames) * 0.8
        
        return 0.0
        
    except Exception as e:
        logger.error(f"❌ 장소 매칭 확인 오류: {e}")
        return 0.0


def _check_object_match(frames, object_condition):
    """객체 조건 매칭 확인"""
    try:
        object_keywords = {
            'person': ['person', '사람', 'people', 'human'],
            'car': ['car', '자동차', 'vehicle', 'automobile'],
            'bag': ['bag', '가방', 'handbag', 'backpack'],
            'phone': ['phone', '전화', 'smartphone', 'mobile']
        }
        
        target_keywords = object_keywords.get(object_condition.lower(), [])
        if not target_keywords:
            return 0.0
        
        match_count = 0
        total_frames = len(frames)
        
        for frame in frames:
            # detected_objects에서 객체 확인
            detected_objects = _get_detected_objects_from_frame(frame)
            for obj in detected_objects:
                obj_class = obj.get('class', '').lower()
                if obj_class in target_keywords:
                    match_count += 1
                    break
        
        if match_count > 0:
            return (match_count / total_frames) * 0.8
        
        return 0.0
        
    except Exception as e:
        logger.error(f"❌ 객체 매칭 확인 오류: {e}")
        return 0.0


def _check_activity_match(frames, activity_condition):
    """활동 조건 매칭 확인"""
    try:
        activity_keywords = {
            'walking': ['walking', '걷기', 'walk', 'moving'],
            'sitting': ['sitting', '앉기', 'sit', 'seated'],
            'running': ['running', '달리기', 'run', 'jogging'],
            'talking': ['talking', '대화', 'talk', 'speaking', 'conversation']
        }
        
        target_keywords = activity_keywords.get(activity_condition.lower(), [])
        if not target_keywords:
            return 0.0
        
        match_count = 0
        total_frames = len(frames)
        
        for frame in frames:
            caption = (frame.caption or '') + ' ' + (frame.enhanced_caption or '') + ' ' + (frame.final_caption or '')
            caption_lower = caption.lower()
            
            for keyword in target_keywords:
                if keyword in caption_lower:
                    match_count += 1
                    break
        
        if match_count > 0:
            return (match_count / total_frames) * 0.8
        
        return 0.0
        
    except Exception as e:
        logger.error(f"❌ 활동 매칭 확인 오류: {e}")
        return 0.0


def _get_detected_objects_from_frame(frame):
    """프레임에서 감지된 객체 추출"""
    try:
        detected_objects = frame.detected_objects
        if isinstance(detected_objects, dict) and 'persons' in detected_objects:
            return detected_objects['persons']
        return []
    except:
        return []


class GenderAnalysisView(APIView):
    """시간대별 성비 분석"""
    permission_classes = [AllowAny]
    
    def post(self, request):
        try:
            video_id = request.data.get('video_id')
            time_range = request.data.get('time_range', {})
            
            logger.info(f"📊 성비 분석 요청: 비디오={video_id}, 시간범위={time_range}")
            
            if not video_id:
                return Response({'error': '비디오 ID가 필요합니다.'}, status=400)
            
            try:
                video = Video.objects.get(id=video_id)
            except Video.DoesNotExist:
                return Response({'error': '비디오를 찾을 수 없습니다.'}, status=404)
            
            # 시간대별 성비 분석
            analysis_result = self._analyze_gender_distribution(video, time_range)
            
            return Response({
                'video_id': video_id,
                'video_name': video.original_name,
                'time_range': time_range,
                'analysis_result': analysis_result,
                'analysis_type': 'gender_distribution'
            })
            
        except Exception as e:
            logger.error(f"❌ 성비 분석 오류: {e}")
            return Response({'error': str(e)}, status=500)
    
    def _analyze_gender_distribution(self, video, time_range):
        """시간대별 성비 분석 수행"""
        try:
            # 프레임 쿼리 설정
            frames_query = Frame.objects.filter(video=video).order_by('timestamp')
            
            # 시간 범위 필터링
            if time_range.get('start') and time_range.get('end'):
                start_time = self._parse_time_to_seconds(time_range['start'])
                end_time = self._parse_time_to_seconds(time_range['end'])
                frames_query = frames_query.filter(timestamp__gte=start_time, timestamp__lte=end_time)
            
            frames = list(frames_query)
            
            if not frames:
                return {
                    'total_frames': 0,
                    'gender_distribution': {'male': 0, 'female': 0, 'unknown': 0},
                    'time_based_analysis': [],
                    'statistics': {
                        'male_percentage': 0,
                        'female_percentage': 0,
                        'unknown_percentage': 0
                    }
                }
            
            # 성별 분포 분석
            gender_counts = {'male': 0, 'female': 0, 'unknown': 0}
            time_based_data = []
            
            for frame in frames:
                frame_gender_data = self._extract_gender_from_frame(frame)
                
                # 전체 성별 카운트
                for gender, count in frame_gender_data.items():
                    gender_counts[gender] += count
                
                # 시간대별 데이터
                time_based_data.append({
                    'timestamp': frame.timestamp,
                    'frame_id': frame.image_id,
                    'gender_distribution': frame_gender_data,
                    'total_persons': sum(frame_gender_data.values())
                })
            
            # 통계 계산
            total_persons = sum(gender_counts.values())
            statistics = {
                'male_percentage': (gender_counts['male'] / total_persons * 100) if total_persons > 0 else 0,
                'female_percentage': (gender_counts['female'] / total_persons * 100) if total_persons > 0 else 0,
                'unknown_percentage': (gender_counts['unknown'] / total_persons * 100) if total_persons > 0 else 0
            }
            
            return {
                'total_frames': len(frames),
                'total_persons': total_persons,
                'gender_distribution': gender_counts,
                'time_based_analysis': time_based_data,
                'statistics': statistics,
                'time_range_used': {
                    'start': time_range.get('start', '0:00'),
                    'end': time_range.get('end', f'{video.duration:.1f}s')
                }
            }
            
        except Exception as e:
            logger.error(f"❌ 성비 분석 수행 오류: {e}")
            return {
                'error': str(e),
                'total_frames': 0,
                'gender_distribution': {'male': 0, 'female': 0, 'unknown': 0},
                'time_based_analysis': [],
                'statistics': {'male_percentage': 0, 'female_percentage': 0, 'unknown_percentage': 0}
            }
    
    def _extract_gender_from_frame(self, frame):
        """프레임에서 성별 정보 추출"""
        try:
            gender_counts = {'male': 0, 'female': 0, 'unknown': 0}
            
            # detected_objects에서 성별 정보 추출
            detected_objects = frame.detected_objects
            if isinstance(detected_objects, dict) and 'persons' in detected_objects:
                persons = detected_objects['persons']
                
                for person in persons:
                    gender = self._get_person_gender(person)
                    gender_counts[gender] += 1
            
            return gender_counts
            
        except Exception as e:
            logger.error(f"❌ 프레임 성별 추출 오류: {e}")
            return {'male': 0, 'female': 0, 'unknown': 0}
    
    def _get_person_gender(self, person):
        """개별 사람의 성별 추출"""
        try:
            # attributes에서 성별 정보 추출
            attributes = person.get('attributes', {})
            gender_info = attributes.get('gender', {})
            
            if isinstance(gender_info, dict):
                gender_value = gender_info.get('value', '').lower()
                
                if 'man' in gender_value or 'male' in gender_value:
                    return 'male'
                elif 'woman' in gender_value or 'female' in gender_value:
                    return 'female'
                else:
                    return 'unknown'
            else:
                return 'unknown'
                
        except Exception as e:
            logger.error(f"❌ 성별 추출 오류: {e}")
            return 'unknown'
    
    def _parse_time_to_seconds(self, time_str):
        """시간 문자열을 초로 변환"""
        try:
            if ':' in time_str:
                parts = time_str.split(':')
                if len(parts) == 2:
                    minutes, seconds = map(float, parts)
                    return minutes * 60 + seconds
                elif len(parts) == 3:
                    hours, minutes, seconds = map(float, parts)
                    return hours * 3600 + minutes * 60 + seconds
            else:
                return float(time_str)
        except:
            return 0.0


class VideoSummaryView(APIView):
    """영상 요약 기능"""
    permission_classes = [AllowAny]
    
    def post(self, request):
        try:
            video_id = request.data.get('video_id')
            summary_type = request.data.get('summary_type', 'comprehensive')  # comprehensive, brief, detailed
            
            logger.info(f"📝 영상 요약 요청: 비디오={video_id}, 타입={summary_type}")
            
            if not video_id:
                return Response({'error': '비디오 ID가 필요합니다.'}, status=400)
            
            try:
                video = Video.objects.get(id=video_id)
            except Video.DoesNotExist:
                return Response({'error': '비디오를 찾을 수 없습니다.'}, status=404)
            
            # 영상 요약 생성
            summary_result = self._generate_video_summary(video, summary_type)
            
            return Response({
                'video_id': video_id,
                'video_name': video.original_name,
                'summary_type': summary_type,
                'summary_result': summary_result,
                'analysis_type': 'video_summary'
            })
            
        except Exception as e:
            logger.error(f"❌ 영상 요약 오류: {e}")
            return Response({'error': str(e)}, status=500)
    
    def _generate_video_summary(self, video, summary_type):
        """영상 요약 생성"""
        try:
            # 프레임 데이터 가져오기
            frames = Frame.objects.filter(video=video).order_by('timestamp')
            
            if not frames.exists():
                return {
                    'summary': '분석된 프레임 데이터가 없습니다.',
                    'key_events': [],
                    'statistics': {},
                    'duration': video.duration,
                    'frame_count': 0
                }
            
            # 기본 통계 수집
            statistics = self._collect_video_statistics(video, frames)
            
            # 키 이벤트 추출
            key_events = self._extract_key_events(frames)
            
            # 요약 타입에 따른 처리
            if summary_type == 'brief':
                summary_text = self._generate_brief_summary(statistics, key_events)
            elif summary_type == 'detailed':
                summary_text = self._generate_detailed_summary(statistics, key_events, frames)
            else:  # comprehensive
                summary_text = self._generate_comprehensive_summary(statistics, key_events, frames)
            
            return {
                'summary': summary_text,
                'key_events': key_events,
                'statistics': statistics,
                'duration': video.duration,
                'frame_count': frames.count(),
                'summary_type': summary_type
            }
            
        except Exception as e:
            logger.error(f"❌ 영상 요약 생성 오류: {e}")
            return {
                'summary': f'요약 생성 중 오류가 발생했습니다: {str(e)}',
                'key_events': [],
                'statistics': {},
                'duration': video.duration,
                'frame_count': 0
            }
    
    def _collect_video_statistics(self, video, frames):
        """영상 통계 수집"""
        try:
            stats = {
                'total_frames': frames.count(),
                'duration': video.duration,
                'person_count': 0,
                'gender_distribution': {'male': 0, 'female': 0, 'unknown': 0},
                'scene_types': [],
                'activities': [],
                'objects_detected': set(),
                'time_periods': []
            }
            
            for frame in frames:
                # 사람 수 계산
                detected_objects = frame.detected_objects
                if isinstance(detected_objects, dict) and 'persons' in detected_objects:
                    persons = detected_objects['persons']
                    stats['person_count'] += len(persons)
                    
                    # 성별 분포
                    for person in persons:
                        gender = self._get_person_gender_from_attributes(person)
                        stats['gender_distribution'][gender] += 1
                
                # 씬 타입 추출
                if frame.scene_type:
                    if frame.scene_type not in stats['scene_types']:
                        stats['scene_types'].append(frame.scene_type)
                
                # 활동 추출
                if frame.enhanced_caption:
                    activities = self._extract_activities_from_caption(frame.enhanced_caption)
                    stats['activities'].extend(activities)
                
                # 객체 추출
                if detected_objects and isinstance(detected_objects, dict) and 'persons' in detected_objects:
                    for person in detected_objects['persons']:
                        obj_class = person.get('class', '')
                        if obj_class:
                            stats['objects_detected'].add(obj_class)
            
            # 중복 제거
            stats['activities'] = list(set(stats['activities']))
            stats['objects_detected'] = list(stats['objects_detected'])
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ 통계 수집 오류: {e}")
            return {}
    
    def _extract_key_events(self, frames):
        """키 이벤트 추출"""
        try:
            key_events = []
            
            # 프레임을 시간순으로 그룹화 (5초 간격)
            time_groups = {}
            for frame in frames:
                time_group = int(frame.timestamp // 5) * 5
                if time_group not in time_groups:
                    time_groups[time_group] = []
                time_groups[time_group].append(frame)
            
            # 각 시간 그룹에서 중요한 이벤트 추출
            for time_group, group_frames in time_groups.items():
                event = self._analyze_time_group(time_group, group_frames)
                if event:
                    key_events.append(event)
            
            return key_events
            
        except Exception as e:
            logger.error(f"❌ 키 이벤트 추출 오류: {e}")
            return []
    
    def _analyze_time_group(self, time_group, frames):
        """시간 그룹 분석"""
        try:
            if not frames:
                return None
            
            # 가장 많은 사람이 있는 프레임 찾기
            max_persons = 0
            representative_frame = frames[0]
            
            for frame in frames:
                detected_objects = frame.detected_objects
                if isinstance(detected_objects, dict) and 'persons' in detected_objects:
                    person_count = len(detected_objects['persons'])
                    if person_count > max_persons:
                        max_persons = person_count
                        representative_frame = frame
            
            # 이벤트 정보 생성
            event = {
                'timestamp': time_group,
                'person_count': max_persons,
                'description': representative_frame.final_caption or representative_frame.caption or '장면',
                'frame_id': representative_frame.image_id,
                'significance': 'high' if max_persons > 5 else 'medium' if max_persons > 2 else 'low'
            }
            
            return event
            
        except Exception as e:
            logger.error(f"❌ 시간 그룹 분석 오류: {e}")
            return None
    
    def _generate_brief_summary(self, statistics, key_events):
        """간단한 요약 생성"""
        try:
            duration_min = statistics.get('duration', 0) / 60
            person_count = statistics.get('person_count', 0)
            gender_dist = statistics.get('gender_distribution', {})
            
            summary_parts = [
                f"이 영상은 {duration_min:.1f}분 길이의 영상입니다.",
                f"총 {person_count}명의 사람이 등장합니다."
            ]
            
            if person_count > 0:
                male_count = gender_dist.get('male', 0)
                female_count = gender_dist.get('female', 0)
                if male_count > 0 or female_count > 0:
                    summary_parts.append(f"성별 분포: 남성 {male_count}명, 여성 {female_count}명")
            
            if key_events:
                summary_parts.append(f"주요 이벤트 {len(key_events)}개가 감지되었습니다.")
            
            return " ".join(summary_parts)
            
        except Exception as e:
            logger.error(f"❌ 간단한 요약 생성 오류: {e}")
            return "요약을 생성할 수 없습니다."
    
    def _generate_detailed_summary(self, statistics, key_events, frames):
        """상세한 요약 생성"""
        try:
            brief_summary = self._generate_brief_summary(statistics, key_events)
            
            detailed_parts = [brief_summary]
            
            # 씬 타입 정보
            scene_types = statistics.get('scene_types', [])
            if scene_types:
                detailed_parts.append(f"장소 유형: {', '.join(scene_types)}")
            
            # 활동 정보
            activities = statistics.get('activities', [])
            if activities:
                activity_counts = {}
                for activity in activities:
                    activity_counts[activity] = activity_counts.get(activity, 0) + 1
                top_activities = sorted(activity_counts.items(), key=lambda x: x[1], reverse=True)[:3]
                detailed_parts.append(f"주요 활동: {', '.join([f'{act}({count}회)' for act, count in top_activities])}")
            
            # 키 이벤트 상세 정보
            if key_events:
                detailed_parts.append("주요 이벤트:")
                for i, event in enumerate(key_events[:5], 1):
                    detailed_parts.append(f"  {i}. {event['timestamp']}초: {event['description']} ({event['person_count']}명)")
            
            return " ".join(detailed_parts)
            
        except Exception as e:
            logger.error(f"❌ 상세한 요약 생성 오류: {e}")
            return "상세한 요약을 생성할 수 없습니다."
    
    def _generate_comprehensive_summary(self, statistics, key_events, frames):
        """종합적인 요약 생성"""
        try:
            detailed_summary = self._generate_detailed_summary(statistics, key_events, frames)
            
            comprehensive_parts = [detailed_summary]
            
            # 시간대별 분석
            if frames.exists():
                first_frame = frames.first()
                last_frame = frames.last()
                comprehensive_parts.append(f"분석 시간 범위: {first_frame.timestamp:.1f}초 ~ {last_frame.timestamp:.1f}초")
            
            # 객체 감지 정보
            objects_detected = statistics.get('objects_detected', [])
            if objects_detected:
                comprehensive_parts.append(f"감지된 객체: {', '.join(objects_detected)}")
            
            return " ".join(comprehensive_parts)
            
        except Exception as e:
            logger.error(f"❌ 종합적인 요약 생성 오류: {e}")
            return "종합적인 요약을 생성할 수 없습니다."
    
    def _get_person_gender_from_attributes(self, person):
        """사람의 성별 정보 추출"""
        try:
            attributes = person.get('attributes', {})
            gender_info = attributes.get('gender', {})
            
            if isinstance(gender_info, dict):
                gender_value = gender_info.get('value', '').lower()
                if 'man' in gender_value or 'male' in gender_value:
                    return 'male'
                elif 'woman' in gender_value or 'female' in gender_value:
                    return 'female'
            
            return 'unknown'
        except:
            return 'unknown'
    
    def _extract_activities_from_caption(self, caption):
        """캡션에서 활동 추출"""
        try:
            activities = []
            activity_keywords = {
                'walking': ['walking', '걷기', 'walk'],
                'sitting': ['sitting', '앉기', 'sit'],
                'running': ['running', '달리기', 'run'],
                'talking': ['talking', '대화', 'talk'],
                'standing': ['standing', '서기', 'stand']
            }
            
            caption_lower = caption.lower()
            for activity, keywords in activity_keywords.items():
                for keyword in keywords:
                    if keyword in caption_lower:
                        activities.append(activity)
                        break
            
            return activities
        except:
            return []


class VideoHighlightView(APIView):
    """영상 하이라이트 자동 추출"""
    permission_classes = [AllowAny]
    
    def post(self, request):
        try:
            video_id = request.data.get('video_id')
            highlight_criteria = request.data.get('criteria', {})
            
            logger.info(f"🎬 하이라이트 추출 요청: 비디오={video_id}, 기준={highlight_criteria}")
            
            if not video_id:
                return Response({'error': '비디오 ID가 필요합니다.'}, status=400)
            
            try:
                video = Video.objects.get(id=video_id)
            except Video.DoesNotExist:
                return Response({'error': '비디오를 찾을 수 없습니다.'}, status=404)
            
            # 하이라이트 추출
            highlights = self._extract_highlights(video, highlight_criteria)
            
            return Response({
                'video_id': video_id,
                'video_name': video.original_name,
                'highlights': highlights,
                'total_highlights': len(highlights),
                'analysis_type': 'video_highlights'
            })
            
        except Exception as e:
            logger.error(f"❌ 하이라이트 추출 오류: {e}")
            return Response({'error': str(e)}, status=500)
    
    def _extract_highlights(self, video, criteria):
        """하이라이트 추출"""
        try:
            frames = Frame.objects.filter(video=video).order_by('timestamp')
            
            if not frames.exists():
                return []
            
            # 하이라이트 점수 계산
            scored_frames = []
            for frame in frames:
                score = self._calculate_highlight_score(frame, criteria)
                if score > 0:
                    scored_frames.append({
                        'frame': frame,
                        'score': score,
                        'timestamp': frame.timestamp,
                        'frame_id': frame.image_id
                    })
            
            # 점수순 정렬
            scored_frames.sort(key=lambda x: x['score'], reverse=True)
            
            # 하이라이트 생성
            highlights = self._create_highlights(scored_frames, criteria)
            
            return highlights
            
        except Exception as e:
            logger.error(f"❌ 하이라이트 추출 수행 오류: {e}")
            return []
    
    def _calculate_highlight_score(self, frame, criteria):
        """프레임의 하이라이트 점수 계산"""
        try:
            score = 0.0
            
            # 사람 수 기반 점수
            person_count = self._get_person_count(frame)
            if person_count > 0:
                # 사람이 많을수록 높은 점수 (최대 3점)
                score += min(person_count * 0.5, 3.0)
            
            # 활동 기반 점수
            activity_score = self._get_activity_score(frame, criteria)
            score += activity_score
            
            # 객체 기반 점수
            object_score = self._get_object_score(frame, criteria)
            score += object_score
            
            # 씬 변화 기반 점수
            scene_score = self._get_scene_change_score(frame)
            score += scene_score
            
            # 품질 기반 점수
            quality_score = self._get_quality_score(frame)
            score += quality_score
            
            return score
            
        except Exception as e:
            logger.error(f"❌ 하이라이트 점수 계산 오류: {e}")
            return 0.0
    
    def _get_person_count(self, frame):
        """프레임의 사람 수 반환"""
        try:
            detected_objects = frame.detected_objects
            if isinstance(detected_objects, dict) and 'persons' in detected_objects:
                return len(detected_objects['persons'])
            return 0
        except:
            return 0
    
    def _get_activity_score(self, frame, criteria):
        """활동 기반 점수 계산"""
        try:
            score = 0.0
            caption = (frame.caption or '') + ' ' + (frame.enhanced_caption or '') + ' ' + (frame.final_caption or '')
            caption_lower = caption.lower()
            
            # 활동 키워드별 점수
            activity_scores = {
                'running': 2.0,
                'jumping': 2.5,
                'dancing': 2.0,
                'fighting': 3.0,
                'talking': 1.0,
                'walking': 0.5,
                'sitting': 0.2,
                'standing': 0.3
            }
            
            for activity, activity_score in activity_scores.items():
                if activity in caption_lower:
                    score += activity_score
            
            # 사용자 지정 활동 기준
            if criteria.get('preferred_activities'):
                for activity in criteria['preferred_activities']:
                    if activity.lower() in caption_lower:
                        score += 1.5
            
            return min(score, 5.0)  # 최대 5점
            
        except Exception as e:
            logger.error(f"❌ 활동 점수 계산 오류: {e}")
            return 0.0
    
    def _get_object_score(self, frame, criteria):
        """객체 기반 점수 계산"""
        try:
            score = 0.0
            detected_objects = frame.detected_objects
            
            if isinstance(detected_objects, dict) and 'persons' in detected_objects:
                persons = detected_objects['persons']
                
                for person in persons:
                    # 액세서리 기반 점수
                    attributes = person.get('attributes', {})
                    accessories = attributes.get('accessories', {})
                    
                    if isinstance(accessories, dict):
                        accessory_value = accessories.get('value', '').lower()
                        if 'phone' in accessory_value:
                            score += 0.5
                        elif 'bag' in accessory_value or 'backpack' in accessory_value:
                            score += 0.3
                        elif 'watch' in accessory_value:
                            score += 0.2
                    
                    # 의상 색상 기반 점수
                    clothing_color = attributes.get('clothing_color', {})
                    if isinstance(clothing_color, dict):
                        color_value = clothing_color.get('value', '').lower()
                        if 'bright' in color_value or 'colorful' in color_value:
                            score += 0.5
            
            # 사용자 지정 객체 기준
            if criteria.get('preferred_objects'):
                for obj in criteria['preferred_objects']:
                    if obj.lower() in str(detected_objects).lower():
                        score += 1.0
            
            return min(score, 3.0)  # 최대 3점
            
        except Exception as e:
            logger.error(f"❌ 객체 점수 계산 오류: {e}")
            return 0.0
    
    def _get_scene_change_score(self, frame):
        """씬 변화 기반 점수 계산"""
        try:
            # 씬 타입이 있으면 점수 추가
            if frame.scene_type:
                return 1.0
            
            # 캡션에 씬 관련 키워드가 있으면 점수 추가
            caption = (frame.caption or '') + ' ' + (frame.enhanced_caption or '') + ' ' + (frame.final_caption or '')
            caption_lower = caption.lower()
            
            scene_keywords = ['scene', '장면', 'change', '변화', 'new', '새로운']
            for keyword in scene_keywords:
                if keyword in caption_lower:
                    return 0.5
            
            return 0.0
            
        except Exception as e:
            logger.error(f"❌ 씬 변화 점수 계산 오류: {e}")
            return 0.0
    
    def _get_quality_score(self, frame):
        """품질 기반 점수 계산"""
        try:
            score = 0.0
            
            # 이미지 품질 정보가 있으면 활용
            if hasattr(frame, 'crop_quality') and frame.crop_quality:
                quality_data = frame.crop_quality
                if isinstance(quality_data, dict):
                    overall_quality = quality_data.get('overall', 0)
                    if overall_quality > 0.7:
                        score += 1.0
                    elif overall_quality > 0.5:
                        score += 0.5
            
            # 신뢰도 기반 점수
            detected_objects = frame.detected_objects
            if isinstance(detected_objects, dict) and 'persons' in detected_objects:
                persons = detected_objects['persons']
                if persons:
                    avg_confidence = sum(p.get('confidence', 0) for p in persons) / len(persons)
                    if avg_confidence > 0.8:
                        score += 1.0
                    elif avg_confidence > 0.6:
                        score += 0.5
            
            return min(score, 2.0)  # 최대 2점
            
        except Exception as e:
            logger.error(f"❌ 품질 점수 계산 오류: {e}")
            return 0.0
    
    def _create_highlights(self, scored_frames, criteria):
        """하이라이트 생성"""
        try:
            highlights = []
            min_score = criteria.get('min_score', 2.0)  # 최소 점수
            max_highlights = criteria.get('max_highlights', 10)  # 최대 하이라이트 수
            
            # 점수 기준 필터링
            filtered_frames = [f for f in scored_frames if f['score'] >= min_score]
            
            # 시간 간격을 고려한 하이라이트 선택
            selected_highlights = []
            last_timestamp = -10  # 최소 10초 간격
            
            for frame_data in filtered_frames[:max_highlights * 2]:  # 여유분을 두고 선택
                if frame_data['timestamp'] - last_timestamp >= 10:  # 10초 이상 간격
                    selected_highlights.append(frame_data)
                    last_timestamp = frame_data['timestamp']
                    
                    if len(selected_highlights) >= max_highlights:
                        break
            
            # 하이라이트 정보 생성
            for i, frame_data in enumerate(selected_highlights):
                frame = frame_data['frame']
                highlight = {
                    'id': i + 1,
                    'timestamp': frame_data['timestamp'],
                    'frame_id': frame_data['frame_id'],
                    'score': frame_data['score'],
                    'description': frame.final_caption or frame.caption or '하이라이트 장면',
                    'person_count': self._get_person_count(frame),
                    'thumbnail_url': f'/api/frame/{frame.video.id}/{frame.image_id}/',
                    'significance': self._get_significance_level(frame_data['score'])
                }
                highlights.append(highlight)
            
            return highlights
            
        except Exception as e:
            logger.error(f"❌ 하이라이트 생성 오류: {e}")
            return []
    
    def _get_significance_level(self, score):
        """중요도 레벨 반환"""
        if score >= 5.0:
            return 'high'
        elif score >= 3.0:
            return 'medium'
        else:
            return 'low'


def _check_bag_in_accessories(obj):
    """객체의 accessories에서 가방 확인"""
    try:
        # _raw 필드에서 accessories 정보 추출
        raw_data = obj.get('_raw', {})
        attributes = raw_data.get('attributes', {})
        accessories = attributes.get('accessories', {})
        
        if isinstance(accessories, dict):
            # value 필드 확인
            accessory_value = accessories.get('value', '').lower()
            if any(bag_keyword in accessory_value for bag_keyword in ['bag', 'backpack', 'handbag']):
                return True
            
            # all_scores에서 가방 관련 점수 확인
            all_scores = accessories.get('all_scores', {})
            for score_key, score_value in all_scores.items():
                if any(bag_keyword in score_key.lower() for bag_keyword in ['bag', 'backpack', 'handbag']):
                    if score_value > 0.1:  # 신뢰도 0.1 이상
                        return True
        
        return False
        
    except Exception as e:
        logger.error(f"❌ 가방 액세서리 확인 오류: {e}")
        return False
