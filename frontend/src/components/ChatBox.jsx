import React, { useState, useEffect, useRef } from "react";
import { Send, CirclePlus } from "lucide-react";
import { useChat } from "../context/ChatContext";
import ModelSelectionModal from "./ModelSelectionModal";

const ChatBox = () => {
  const { messages, sendMessage, isLoading, selectedModels, setSelectedModels } = useChat();
  const [inputMessage, setInputMessage] = useState("");
  const [isModalOpen, setIsModalOpen] = useState(false);
  const messagesEndRefs = useRef({});

  useEffect(() => {
    selectedModels.concat("optimal").forEach((modelId) => {
      if (!messagesEndRefs.current[modelId]) {
        messagesEndRefs.current[modelId] = React.createRef();
      }
    });
  }, [selectedModels]);

  // 메시지 전송
  const handleSendMessage = (e) => {
    e.preventDefault();
    sendMessage(inputMessage);
    setInputMessage("");
  };

  // 새 메시지가 추가될 때마다 스크롤을 맨 아래로 이동
  useEffect(() => {
    selectedModels.concat("optimal").forEach((modelId) => {
      messagesEndRefs.current[modelId]?.current?.scrollIntoView({ behavior: "smooth" });
    });
  }, [messages, selectedModels]);

  return (
    <div className="h-screen w-full bg-white flex flex-col">
      {/* AI 이름 박스 - 상단 고정 */}
      <div className="flex-shrink-0 flex bg-white border-b w-full sticky top-0 z-10">
        {selectedModels.concat("optimal").map((modelId) => (
          <div key={modelId} className="p-4 text-xl font-semibold text-center border-r flex-1 whitespace-nowrap overflow-hidden text-ellipsis">
            {modelId === "optimal" ? "최적의 답변" : modelId.toUpperCase()}
          </div>
        ))}
      </div>

      {/* 채팅 메시지 영역 (유동적 크기 적용) */}
      <div className="flex-1 grid overflow-hidden" style={{ gridTemplateColumns: `repeat(${selectedModels.length + 1}, minmax(0, 1fr))` }}>
        {selectedModels.concat("optimal").map((modelId) => (
          <div key={modelId} className="border-r flex-1 overflow-y-auto">
            <div className="min-h-full px-4 pb-20 pt-[4rem]">
              {messages[modelId]?.map((message, index) => (
                <div key={index} className={`flex ${message.isUser ? "justify-end" : "justify-start"} mb-4`}>
                  <div className={`max-w-[85%] p-4 rounded-2xl ${message.isUser ? "bg-purple-600 text-white" : "bg-gray-100 text-gray-800"}`}>
                    {message.text}
                  </div>
                </div>
              ))}
              {isLoading && (
                <div className="flex justify-start mb-4">
                  <div className="bg-gray-100 text-gray-800 p-4 rounded-2xl">입력 중...</div>
                </div>
              )}
              <div ref={messagesEndRefs.current[modelId]} />
            </div>
          </div>
        ))}
      </div>

      {/* 입력창을 화면 하단에 고정 */}
      <div className="border-t p-4 w-full flex-shrink-0 bg-white sticky bottom-0 z-10">
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