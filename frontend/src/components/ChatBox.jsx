import React, { useState } from "react";
import { Send } from "lucide-react";
import { useChat } from "../context/ChatContext"; 

const ChatBox = () => {
  const { messages, sendMessage, isLoading, messagesEndRef } = useChat(); 
  const [inputMessage, setInputMessage] = useState("");

  // 메시지 전송
  const handleSendMessage = (e) => {
    e.preventDefault();
    sendMessage(inputMessage);
    setInputMessage("");
  };

  return (
    <div className="flex flex-col h-full w-full bg-white">
      {/* 채팅 영역 + 입력창 포함하는 컨테이너 */}
      <div className="flex flex-col flex-1 w-full">
        {/* 채팅 메시지 영역 */}
        <div className="flex-1 grid grid-cols-3 overflow-y-auto">
          {["gpt", "claude", "mixtral"].map((botName) => (
            <div key={botName} className="flex flex-col border-r last:border-r-0">
              <div className="p-4 border-b">
                <h2 className="text-xl font-semibold text-center">{botName.toUpperCase()}</h2>
              </div>
              <div className="flex-1 overflow-y-auto p-6 space-y-6">
                {messages[botName].map((message, index) => (
                  <div key={index} className={`flex ${message.isUser ? "justify-end" : "justify-start"}`}>
                    <div
                      className={`max-w-2xl p-4 rounded-2xl ${
                        message.isUser ? "bg-purple-600 text-white" : "bg-gray-100 text-gray-800"
                      }`}
                    >
                      {message.text}
                    </div>
                  </div>
                ))}
                {isLoading && (
                  <div className="flex justify-start">
                    <div className="bg-gray-100 text-gray-800 p-4 rounded-2xl">입력 중...</div>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>
            </div>
          ))}
        </div>

        {/* 입력창을 화면 하단에 고정 */}
        <div className="border-t p-4 w-full flex-shrink-0">
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
    </div>
  );
};

export default ChatBox;