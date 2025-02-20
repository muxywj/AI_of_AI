import React, { createContext, useState, useContext, useRef, useEffect } from "react";

// ChatContext 생성
const ChatContext = createContext();

// ChatProvider 컴포넌트 (전역 상태 관리)
export const ChatProvider = ({ children }) => {
  const [messages, setMessages] = useState({
    gpt: [],
    claude: [],
    mixtral: [],
    optimal: []
  });
  const [isLoading, setIsLoading] = useState(false);
  const [selectedModels, setSelectedModels] = useState(["gpt", "claude", "mixtral"]);
  const messagesEndRef = useRef(null);

  // 스크롤을 자동으로 최신 메시지로 이동
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // 메시지 추가 함수
  const addMessage = (botName, text, isUser) => {
    setMessages((prev) => ({
      ...prev,
      [botName]: [...(prev[botName] || []), { text, isUser }],
    }));
  };

  // API 요청 함수 (AI 응답 받기)
  const sendMessage = async (userMessage) => {
    if (!userMessage.trim()) return;

    setIsLoading(true);

    // 선택된 모델들에 대해서만 메시지 처리
    for (const modelId of selectedModels) {
      // 사용자 메시지 추가
      setMessages(prev => ({
        ...prev,
        [modelId]: [...(prev[modelId] || []), { text: userMessage, isUser: true }]
      }));

      try {
        const response = await fetch(`http://localhost:8000/chat/${modelId}/`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ message: userMessage }),
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        
        // AI 응답 추가
        setMessages(prev => ({
          ...prev,
          [modelId]: [...prev[modelId], { text: data.response, isUser: false }]
        }));

      } catch (error) {
        console.error("Error:", error);
        // 에러 메시지 추가
        setMessages(prev => ({
          ...prev,
          [modelId]: [...prev[modelId], { text: "죄송합니다. 오류가 발생했습니다.", isUser: false }]
        }));
      }
    }

    setIsLoading(false);
  };

  return (
    <ChatContext.Provider value={{ 
      messages, 
      sendMessage, 
      isLoading,
      selectedModels,
      setSelectedModels
    }}>
      {children}
    </ChatContext.Provider>
  );
};

// ChatContext 사용을 위한 커스텀 훅
export const useChat = () => {
  const context = useContext(ChatContext);
  if (!context) {
    throw new Error("useChat must be used within a ChatProvider");
  }
  return context;
};