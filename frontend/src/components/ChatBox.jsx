import React, { useState, useEffect, useRef } from "react";
import { Send } from "lucide-react";
import { useChat } from "../context/ChatContext"; 

const ChatBox = () => {
  const { messages, sendMessage, isLoading } = useChat(); 
  const [inputMessage, setInputMessage] = useState("");
  const messagesEndRefs = {
    gpt: useRef(null),
    claude: useRef(null),
    mixtral: useRef(null)
  };

  // 메시지 전송
  const handleSendMessage = (e) => {
    e.preventDefault();
    sendMessage(inputMessage);
    setInputMessage("");
  };

  // 새 메시지가 추가될 때마다 스크롤을 맨 아래로 이동
  useEffect(() => {
    Object.values(messagesEndRefs).forEach(ref => {
      ref.current?.scrollIntoView({ behavior: "smooth" });
    });
  }, [messages]);

  return (
    <div className="flex flex-col h-screen w-full bg-white">
      {/* AI 이름 박스 - 상단 고정 */}
      <div className="grid grid-cols-3 border-b bg-white sticky top-0 z-20">
        {["gpt", "claude", "mixtral"].map((botName) => (
          <div key={botName} className="p-4 text-xl font-semibold text-center border-r last:border-r-0" style={{width: '100%'}}>
            {botName.toUpperCase()}
          </div>
        ))}
      </div>

      {/* 채팅 메시지 영역 */}
      <div className="flex-1 grid grid-cols-3 min-h-0 mt-14">
        {["gpt", "claude", "mixtral"].map((botName) => (
          <div key={botName} className="border-r last:border-r-0 overflow-hidden" style={{width: '100%'}}>
            <div className="h-full overflow-y-auto px-4">
              {messages[botName].map((message, index) => (
                <div key={index} className={`flex ${message.isUser ? "justify-end" : "justify-start"} mb-4`}>
                  <div
                    className={`max-w-[85%] p-4 rounded-2xl ${
                      message.isUser ? "bg-purple-600 text-white" : "bg-gray-100 text-gray-800"
                    }`}
                  >
                    {message.text}
                  </div>
                </div>
              ))}
              {isLoading && (
                <div className="flex justify-start mb-4">
                  <div className="bg-gray-100 text-gray-800 p-4 rounded-2xl">입력 중...</div>
                </div>
              )}
              <div ref={messagesEndRefs[botName]} />
            </div>
          </div>
        ))}
      </div>

      {/* 입력창을 화면 하단에 고정 */}
      <div className="border-t p-4 w-full flex-shrink-0 bg-white sticky bottom-0 z-20">
        <form onSubmit={handleSendMessage} className="max-w-4xl mx-auto flex items-center bg-white border rounded-xl p-2">
          <input
            type="text"
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            placeholder="메시지를 입력하세요..."
            className="flex-1 px-3 py-2 focus:outline-none"
          />
          <button type="submit" disabled={isLoading} className="p-2 rounded-lg hover:bg-gray-100">
            <Send className="w-5 h-5 text-gray-600" />
          </button>
        </form>
      </div>
    </div>
  );
};

export default ChatBox;