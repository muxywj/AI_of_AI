import React, { useState, useRef, useEffect } from 'react';
import { Send, Menu, Search, Clock, Settings } from 'lucide-react';

const ChatInterface = () => {
  const [messages, setMessages] = useState({
    gpt: [],
    claude: [],
    mixtral: [],
  });
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

//   const quickPrompts = [
//     { title: "코드 작성 도움", desc: "웹사이트의 스타일리시한 헤더를 위한 코드" },
//     { title: "이미지 생성", desc: "취침 시간 이야기와 그림 만들기" },
//     { title: "텍스트 분석", desc: "이력서를 위한 강력한 문구 생성" },
//     { title: "문제 해결", desc: "빠른 문제 해결 방법 제안" },
//   ];

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!inputMessage.trim()) return;

    const userMessage = inputMessage;
    setInputMessage('');

    const updatedMessages = {};
    setIsLoading(true);

    for (const botName of ['gpt', 'claude', 'mixtral']) {
      try {
        const response = await fetch(`http://localhost:8000/chat/${botName}/`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ message: userMessage }),
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        updatedMessages[botName] = [
          ...(messages[botName] || []),
          { text: userMessage, isUser: true },
          { text: data.response, isUser: false },
        ];
      } catch (error) {
        console.error('Error:', error);
        updatedMessages[botName] = [
          ...(messages[botName] || []),
          { text: userMessage, isUser: true },
          {
            text: `오류가 발생했습니다: ${error.message}`,
            isUser: false,
          },
        ];
      }
    }

    setMessages((prev) => ({
      ...prev,
      ...updatedMessages,
    }));

    setIsLoading(false);
  };

  return (
    <div className="flex flex-col h-screen bg-white">
      {/* 상단 네비게이션 바 */}
      <nav className="bg-white border-b px-6 py-3 flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <Menu className="w-6 h-6 text-gray-600" />
          <h1 className="text-xl font-semibold">AI Chatbot</h1>
        </div>
        <div className="flex items-center space-x-4">
          <Search className="w-5 h-5 text-gray-600" />
          <Clock className="w-5 h-5 text-gray-600" />
          <Settings className="w-5 h-5 text-gray-600" />
        </div>
      </nav>

      {/* 메인 컨텐츠 영역 */}
      <div className="flex-1 flex">
        {/* 왼쪽 사이드바 */}
        {/* <div className="w-64 border-r bg-gray-50 p-4">
          <h2 className="text-lg font-semibold mb-4">빠른 시작</h2>
          <div className="space-y-3">
            {quickPrompts.map((prompt, index) => (
              <div key={index} className="p-3 bg-white rounded-lg shadow-sm hover:shadow cursor-pointer">
                <h3 className="font-medium text-sm">{prompt.title}</h3>
                <p className="text-xs text-gray-600 mt-1">{prompt.desc}</p>
              </div>
            ))}
          </div>
        </div> */}

        {/* 채팅 영역 */}
        <div className="flex-1 grid grid-cols-3">
          {['gpt', 'claude', 'mixtral'].map((botName) => (
            <div key={botName} className="flex flex-col border-r last:border-r-0">
              <div className="p-4 border-b">
                <h2 className="text-xl font-semibold">{botName}</h2>
              </div>
              <div className="flex-1 overflow-y-auto p-6 space-y-6">
                {messages[botName].map((message, index) => (
                  <div
                    key={index}
                    className={`flex ${message.isUser ? 'justify-end' : 'justify-start'}`}
                  >
                    <div
                      className={`max-w-2xl p-4 rounded-2xl ${
                        message.isUser
                          ? 'bg-purple-600 text-white'
                          : 'bg-gray-100 text-gray-800'
                      }`}
                    >
                      {message.text}
                    </div>
                  </div>
                ))}
                {isLoading && (
                  <div className="flex justify-start">
                    <div className="bg-gray-100 text-gray-800 p-4 rounded-2xl">
                      입력 중...
                    </div>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* 입력 영역 */}
      <div className="border-t p-4">
        <form onSubmit={handleSubmit} className="max-w-4xl mx-auto">
          <div className="flex items-center bg-white border rounded-xl p-2">
            <input
              type="text"
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              placeholder="메시지를 입력하세요..."
              className="flex-1 px-3 py-2 focus:outline-none"
            />
            <button
              type="submit"
              disabled={isLoading}
              className="p-2 rounded-lg hover:bg-gray-100"
            >
              <Send className="w-5 h-5 text-gray-600" />
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default ChatInterface;