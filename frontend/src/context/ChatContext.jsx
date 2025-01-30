import React, { createContext, useState, useContext, useRef, useEffect } from "react";

// ChatContext 생성
const ChatContext = createContext();

// ChatProvider 컴포넌트 (전역 상태 관리)
export const ChatProvider = ({ children }) => {
  const [messages, setMessages] = useState({
    gpt: [],
    claude: [],
    mixtral: [],
  });
  const [isLoading, setIsLoading] = useState(false);
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
    const updatedMessages = {};

    for (const botName of ["gpt", "claude", "mixtral"]) {
      try {
        const response = await fetch(`http://localhost:8000/chat/${botName}/`, {
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
        updatedMessages[botName] = [
          ...(messages[botName] || []),
          { text: userMessage, isUser: true },
          { text: data.response, isUser: false },
        ];
      } catch (error) {
        console.error("Error:", error);
        updatedMessages[botName] = [
          ...(messages[botName] || []),
          { text: userMessage, isUser: true },
          { text: `오류가 발생했습니다: ${error.message}`, isUser: false },
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
    <ChatContext.Provider value={{ messages, addMessage, sendMessage, isLoading, messagesEndRef }}>
      {children}
    </ChatContext.Provider>
  );
};

// ChatContext 사용을 위한 커스텀 훅
export const useChat = () => useContext(ChatContext);