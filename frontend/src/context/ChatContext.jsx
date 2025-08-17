// context/ChatContext.jsx
import React, { createContext, useContext, useState, useEffect } from 'react';

const ChatContext = createContext();

export const ChatProvider = ({ children, initialModels = [] }) => {
  const [selectedModels, setSelectedModels] = useState(initialModels);
  const [messages, setMessages] = useState({});
  const [isLoading, setIsLoading] = useState(false);

  // initialModels가 변경되면 selectedModels 업데이트
  useEffect(() => {
    if (initialModels.length > 0) {
      setSelectedModels(initialModels);
    }
  }, [initialModels]);

  const sendMessage = async (messageText) => {
    if (!messageText.trim()) return;
    if (!selectedModels || selectedModels.length === 0) return;

    // 사용자 메시지 생성
    const userMessage = {
      text: messageText.trim(),
      isUser: true,
      timestamp: new Date().toISOString(),
      id: Date.now() + Math.random()
    };

    // 모든 선택된 모델 + optimal에 사용자 메시지 추가
    const modelsToUpdate = [...selectedModels, "optimal"];
    
    setMessages(prevMessages => {
      const newMessages = { ...prevMessages };
      
      modelsToUpdate.forEach(modelId => {
        if (!newMessages[modelId]) {
          newMessages[modelId] = [];
        }
        newMessages[modelId] = [...newMessages[modelId], userMessage];
      });
      
      return newMessages;
    });

    // 로딩 상태 시작
    setIsLoading(true);

    // 각 모델별로 AI 응답 처리
    try {
      const responsePromises = modelsToUpdate.map(async (modelId) => {
        try {
          // API 호출 (실제로는 실패할 것임)
          const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              model: modelId,
              message: messageText
            })
          });

          if (!response.ok) {
            throw new Error('API 호출 실패');
          }

          const data = await response.json();
          
          const aiMessage = {
            text: data.response || "응답을 받았습니다.",
            isUser: false,
            timestamp: new Date().toISOString(),
            id: Date.now() + Math.random() + modelId
          };

          // 각 모델별로 응답 추가
          setMessages(prevMessages => {
            const newMessages = { ...prevMessages };
            if (!newMessages[modelId]) {
              newMessages[modelId] = [];
            }
            newMessages[modelId] = [...newMessages[modelId], aiMessage];
            return newMessages;
          });

        } catch (error) {
          // 에러 메시지 추가
          const errorMessage = {
            text: `죄송합니다. ${modelId.toUpperCase()} 모델에서 오류가 발생했습니다. API 연결을 확인해주세요.`,
            isUser: false,
            timestamp: new Date().toISOString(),
            id: Date.now() + Math.random() + modelId + "_error"
          };

          setMessages(prevMessages => {
            const newMessages = { ...prevMessages };
            if (!newMessages[modelId]) {
              newMessages[modelId] = [];
            }
            newMessages[modelId] = [...newMessages[modelId], errorMessage];
            return newMessages;
          });
        }
      });

      // 모든 응답 완료 대기
      await Promise.all(responsePromises);
      
    } catch (error) {
      console.error("Error in sendMessage:", error);
    } finally {
      // 로딩 상태 종료
      setIsLoading(false);
    }
  };

  return (
    <ChatContext.Provider value={{
      selectedModels,
      setSelectedModels,
      messages,
      setMessages,
      isLoading,
      sendMessage
    }}>
      {children}
    </ChatContext.Provider>
  );
};

export const useChat = () => {
  const context = useContext(ChatContext);
  if (!context) {
    throw new Error('useChat must be used within a ChatProvider');
  }
  return context;
};