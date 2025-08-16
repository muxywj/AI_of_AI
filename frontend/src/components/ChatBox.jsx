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
    <div 
      className="h-full w-full flex flex-col"
      style={{ background: 'rgba(245, 242, 234, 0.4)' }}
    >
      {/* CSS 스타일을 위한 style 태그 추가 */}
      <style jsx>{`
        .green-gradient-message {
          background: linear-gradient(135deg, #5d7c5b, #8ba88a);
          color: white;
          position: relative;
          overflow: hidden;
        }
        
        .green-gradient-message:hover::before {
          left: 100%;
        }
        
        /* 부드러운 호버 효과 */
        .green-gradient-message:hover {
          box-shadow: 0 8px 32px rgba(93, 124, 91, 0.3);
          transform: translateY(-1px);
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        /* 투명 배경 스타일 */
        .chat-header {
          background: rgba(245, 242, 234, 0.4);
          backdrop-filter: blur(10px);
          border-bottom: 1px solid rgba(139, 168, 138, 0.15);
          height: 60px;
        }

        .chat-column {
          background: rgba(255, 255, 255, 0.3);
          backdrop-filter: blur(5px);
        }

        .chat-container {
          height: calc(100% - 140px); /* 헤더(60px) + 입력창(80px) 제외 */
        }

        /* 세련된 입력 영역 */
        .aiofai-input-area {
          background: rgba(245, 242, 234, 0.4);
          backdrop-filter: blur(10px);
          border-top: 1px solid rgba(139, 168, 138, 0.15);
          padding: 1.2rem;
          height: 80px;
          display: flex;
          align-items: center;
        }

        /* 사용자 메시지 스타일 - 미리보기 코드 스타일 적용 */
        .aiofai-user-message {
          background: linear-gradient(135deg, #5d7c5b, #8ba88a);
          color: #ffffff;
          padding: 1.2rem 1.5rem;
          border-radius: 24px 24px 8px 24px;
          max-width: 85%;
          box-shadow: 0 8px 32px rgba(93, 124, 91, 0.3);
          font-weight: 500;
          line-height: 1.5;
          position: relative;
        }
        
        .aiofai-user-message::before {
          position: absolute;
          top: -10px;
          right: -5px;
          font-size: 0.8rem;
        }

        /* AI 봇 메시지 스타일 - 미리보기 코드 스타일 적용 */
        .aiofai-bot-message {
          background: rgba(255, 255, 255, 0.8);
          backdrop-filter: blur(10px);
          color: #2d3e2c;
          border: 1px solid rgba(139, 168, 138, 0.2);
          padding: 1.2rem 1.5rem;
          border-radius: 24px 24px 24px 8px;
          max-width: 85%;
          box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
          line-height: 1.6;
          position: relative;
        }
        
        .aiofai-bot-message::before {
          position: absolute;
          top: -8px;
          left: -5px;
          font-size: 0.8rem;
        }
        
        .aiofai-input-box {
          background: white;
          border: 1px solid #e5e7eb;
          border-radius: 12px;
          display: flex;
          align-items: center;
          padding: 0.4rem;
          gap: 0.6rem;
          max-width: 51.2rem;
          margin: 0 auto;
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
          width: 80%;
        }
        
        .aiofai-input-box:focus-within {
          border-color: #8ba88a;
          box-shadow: 0 0 0 3px rgba(93, 124, 91, 0.1);
        }
        
        .input-field {
          flex: 1;
          border: none;
          outline: none;
          padding: 0.6rem;
          background: transparent;
          color: #2d3e2c;
          font-size: 1rem;
          border-radius: 12px;
        }
        
        .input-field::placeholder {
          color: rgba(45, 62, 44, 0.5);
        }
        
        .aiofai-send-button {
          color: #5d7c5b;
          padding: 8px;
          border-radius: 12px;
          transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
          cursor: pointer;
          border: none;
          background: transparent;
        }
        
        .aiofai-send-button:hover {
          background: rgba(139, 168, 138, 0.1);
          color: #5d7c5b;
          transform: translateY(-1px) scale(1.05);
          box-shadow: 0 8px 25px rgba(93, 124, 91, 0.2);
          border-radius: 12px;
        }

        .aiofai-send-button:disabled {
          opacity: 0.5;
          cursor: not-allowed;
          transform: none;
        }

        .aiofai-send-button:disabled:hover {
          background: transparent;
          color: rgba(93, 124, 91, 0.5);
          transform: none;
          box-shadow: none;
        }
      `}</style>

      {/* AI 이름 박스 - 상단 고정 */}
      <div className="flex-shrink-0 flex chat-header w-full">
        {selectedModels.concat("optimal").map((modelId) => (
          <div 
            key={modelId} 
            className="px-4 py-2 text-lg font-semibold text-center border-r flex-1 whitespace-nowrap overflow-hidden text-ellipsis flex items-center justify-center"
            style={{
              color: '#2d3e2c',
              borderRightColor: 'rgba(139, 168, 138, 0.3)'
            }}
          >
            {modelId === "optimal" ? "최적의 답변" : modelId.toUpperCase()}
          </div>
        ))}
      </div>

      {/* 채팅 메시지 영역 (유동적 크기 적용) */}
      <div className="chat-container grid overflow-hidden" style={{ gridTemplateColumns: `repeat(${selectedModels.length + 1}, minmax(0, 1fr))` }}>
        {selectedModels.concat("optimal").map((modelId) => (
          <div key={modelId} className="border-r flex-1 overflow-y-auto chat-column">
            <div className="h-full px-4 py-3">
              {messages[modelId]?.map((message, index) => (
                <div key={index} className={`flex ${message.isUser ? "justify-end" : "justify-start"} mb-4`}>
                  <div className={`${
                    message.isUser 
                      ? "aiofai-user-message" 
                      : "aiofai-bot-message"
                  }`}>
                    {message.text}
                  </div>
                </div>
              ))}
              {isLoading && (
                <div className="flex justify-start mb-4">
                  <div className="bg-gray-100 text-gray-800 p-4 rounded-2xl">입력 중...</div>
                </div>
              )}
              {/* 하단 여백을 위한 스페이서 */}
              <div className="h-3"></div>
              <div ref={messagesEndRefs.current[modelId]} />
            </div>
          </div>
        ))}
      </div>

      {/* 세련된 입력창 - 미리보기 스타일 적용 */}
      <div className="aiofai-input-area">
        <form onSubmit={handleSendMessage} className="aiofai-input-box">
          <input
            type="text"
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            placeholder="메시지를 입력하세요..."
            className="input-field"
            disabled={isLoading}
          />
          <button 
            type="submit" 
            disabled={isLoading || !inputMessage.trim()} 
            className="aiofai-send-button"
          >
            <Send className="w-5 h-5" />
          </button>
        </form>
      </div>
    </div>
  );
};

export default ChatBox;