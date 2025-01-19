


from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import openai
import anthropic
from groq import Groq
import anthropic

class ChatBot:
    def __init__(self, api_key, model, api_type):
        self.conversation_history = []
        self.model = model
        self.api_type = api_type
        self.api_key = api_key  # api_key 속성 추가
        
        if api_type == 'openai':
            openai.api_key = api_key
        elif api_type == 'anthropic':
            self.client = anthropic.Client(api_key=api_key)
        elif api_type == 'groq':
            self.client = Groq(api_key=api_key)
    
    def chat(self, user_input):
        try:
            # 대화 시작 시 시스템 메시지 추가
            if not self.conversation_history:
                self.conversation_history.append({
                    "role": "system",
                    "content": """당신은 사용자가 입력한 언어에 맞춰 답변하는 AI 어시스턴트입니다.
                    1. 사용자가 한국어로 입력하면, 모든 답변을 한국어로 작성합니다.
                    2. 사용자가 다른 언어(예: 영어)로 입력하면, 해당 언어로 답변을 작성합니다.
                    3. 전문 용어가 필요할 경우, 한글로 추가 설명을 덧붙입니다.
                    """
                })

            # 사용자 입력 출력
            print(f"User input: {user_input}")
            
            self.conversation_history.append({"role": "user", "content": user_input})
            
            if self.api_type == 'openai':
                # OpenAI 방식 처리
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=self.conversation_history,
                    temperature=0.7,
                    max_tokens=1024
                )
                assistant_response = response['choices'][0]['message']['content']
            
            elif self.api_type == 'anthropic':
    # Anthropic Messages API 방식 처리
                client = anthropic.Client(api_key=self.api_key)
                
                # 메시지 구조 수정
                message = client.messages.create(
                    model="claude-3-5-haiku-20241022",  # 올바른 모델명 사용
                    max_tokens=4096,
                    temperature=0,
                    messages=[
                        {
                            "role": "user",
                            "content": f"""You are an AI assistant capable of communicating in multiple languages.
                            Your task is to respond in the same language as the user's input and answer their question or address their request.

                            Once you have identified the language, formulate your response in that same language.
                            Ensure that your grammar, vocabulary, and style are appropriate for the detected language.

                            Answer the user's question or address their request to the best of your ability. If you need clarification or additional information, ask for it in the same language as the user's input.

                            Here is the user's input:

                            <user_input>
                            {user_input}
                            </user_input>"""
                        }
                    ]
                )
                
                # 응답 추출 및 할당
           
                assistant_response = message.content[0].text 
                
                # 응답 확인
                print(f"Anthropic response: {assistant_response}")  # 응답 출력


            
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
            print(f"Error: {str(e)}")
            return f"오류가 발생했습니다: {str(e)}"




chatbots = {
    'gpt': ChatBot(OPENAI_API_KEY, 'gpt-3.5-turbo', 'openai'),
    'claude': ChatBot(ANTHROPIC_API_KEY, 'claude-3-5-haiku-20241022', 'anthropic'), 
    'mixtral': ChatBot(GROQ_API_KEY, 'mixtral-8x7b-32768', 'groq'),
}

class ChatView(APIView):
    def post(self, request, bot_name):
        try:
            data = request.data
            user_message = data.get('message')
            
            if not user_message:
                return Response({'error': 'No message provided'}, status=status.HTTP_400_BAD_REQUEST)
            
            chatbot = chatbots.get(bot_name)
            if not chatbot:
                return Response({'error': 'Invalid bot name'}, status=status.HTTP_400_BAD_REQUEST)

            response = chatbot.chat(user_message)
            return Response({'response': response})
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)