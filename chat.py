import os
from groq import Groq

class ChatBot:
    def __init__(self, api_key):
        self.client = Groq(api_key=api_key)
        self.conversation_history = []

    def chat(self, user_input):
        # 대화 기록에 사용자 입력 추가
        self.conversation_history.append({"role": "user", "content": user_input})
        
        try:
            # Groq API 호출
            response = self.client.chat.completions.create(
                model="mixtral-8x7b-32768",  # 또는 다른 지원되는 모델
                messages=self.conversation_history,
                temperature=0.7,
                max_tokens=1024
            )
            
            # 응답 추출
            assistant_response = response.choices[0].message.content
            
            # 대화 기록에 AI 응답 추가
            self.conversation_history.append({"role": "assistant", "content": assistant_response})
            
            return assistant_response
            
        except Exception as e:
            return f"오류가 발생했습니다: {str(e)}"

def main():
    # Groq API 키 설정
    api_key = "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"  # 실제 API 키로 교체해주세요
    
    # 챗봇 인스턴스 생성
    chatbot = ChatBot(api_key)
    
    print("챗봇과 대화를 시작합니다. 종료하려면 'quilst'를 입력하세요.")
    
    while True:
        user_input = input("사용자: ")
        
        if user_input.lower() == 'quit':
            print("대화를 종료합니다.")
            break
            
        response = chatbot.chat(user_input)
        print("챗봇:", response)

if __name__ == "__main__":
    main()