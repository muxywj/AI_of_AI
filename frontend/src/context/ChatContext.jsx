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

  const sendMessage = async (messageText, requestId = null, options = {}) => {
    // 파일이 options에 있는 경우 처리
    const filesBase64 = options.filesBase64 || [];
    const imagesBase64 = options.imagesBase64 || [];
    const hasFiles = filesBase64.length > 0 || imagesBase64.length > 0;
    
    if (!messageText.trim() && !hasFiles) return;
    if (!selectedModels || selectedModels.length === 0) return;

    // 파일명들을 추출
    const fileNames = [
      ...filesBase64.map(f => f.name),
      ...imagesBase64.map(f => f.name)
    ];

    // 사용자 메시지 생성
    const userMessage = {
      text: hasFiles ? `파일 업로드: ${fileNames.join(', ')}` : messageText.trim(),
      isUser: true,
      timestamp: new Date().toISOString(),
      id: Date.now() + Math.random(),
      // 파일이 있는 경우 파일 정보 추가
      files: hasFiles ? [...filesBase64, ...imagesBase64] : []
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
      // 먼저 optimal을 제외한 모델들의 응답을 수집
      const otherModels = modelsToUpdate.filter(modelId => modelId !== 'optimal');
      const otherResponses = {};
      
             // 다른 모델들의 응답 수집
             const otherResponsePromises = otherModels.map(async (modelId) => {
               try {
                 const formData = new FormData();
                 formData.append('message', messageText || '');
                 
                 // 파일이 있는 경우 FormData에 추가
                 if (hasFiles) {
                   // 첫 번째 파일만 전송 (백엔드에서 하나씩 처리)
                   const firstFile = filesBase64[0] || imagesBase64[0];
                   if (firstFile) {
                     // Base64를 Blob으로 변환
                     const byteCharacters = atob(firstFile.dataUrl.split(',')[1]);
                     const byteNumbers = new Array(byteCharacters.length);
                     for (let i = 0; i < byteCharacters.length; i++) {
                       byteNumbers[i] = byteCharacters.charCodeAt(i);
                     }
                     const byteArray = new Uint8Array(byteNumbers);
                     const blob = new Blob([byteArray], { type: firstFile.type });
                     formData.append('file', blob, firstFile.name);
                   }
                 }

                 const response = await fetch(`http://localhost:8000/chat/${modelId}/`, {
            method: 'POST',
            body: formData
          });

          if (!response.ok) {
            throw new Error('API 호출 실패');
          }

          const data = await response.json();
          const aiResponse = data.response || "응답을 받았습니다.";
          
          // 응답 저장
          otherResponses[modelId] = aiResponse;
          
          const aiMessage = {
            text: aiResponse,
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

          return aiResponse;

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
          
          return null;
        }
      });

             // 다른 모델들의 응답 완료 대기
             await Promise.all(otherResponsePromises);

             // 유사도 분석 수행 (2개 이상의 모델이 있는 경우)
             if (otherModels.length >= 2) {
               const modelResponses = {};
               
               // 최신 메시지 상태에서 AI 응답들 수집 및 유사도 분석 수행
               setMessages(prevMessages => {
                 const newMessages = { ...prevMessages };
                 
                 otherModels.forEach((modelId, index) => {
                   // 해당 모델의 최신 AI 응답 찾기
                   const modelMessages = newMessages[modelId] || [];
                   const lastAIMessage = modelMessages.filter(msg => !msg.isUser).pop();
                   if (lastAIMessage) {
                     modelResponses[modelId] = lastAIMessage.text;
                   }
                 });

                 console.log('Collected model responses for similarity analysis:', modelResponses);

                 if (Object.keys(modelResponses).length >= 2) {
                   // 비동기로 유사도 분석 수행
                   import('../utils/similarityAnalysis').then(({ calculateTextSimilarity, clusterResponses }) => {
                     try {
                       const clusters = clusterResponses(modelResponses, 0.7);
                       const similarityMatrix = {};
                       
                       // 유사도 행렬 계산
                       Object.keys(modelResponses).forEach(model1 => {
                         similarityMatrix[model1] = {};
                         Object.keys(modelResponses).forEach(model2 => {
                           if (model1 === model2) {
                             similarityMatrix[model1][model2] = 1;
                           } else {
                             similarityMatrix[model1][model2] = calculateTextSimilarity(
                               modelResponses[model1], 
                               modelResponses[model2]
                             );
                           }
                         });
                       });

                       // 유사도 분석 결과 생성
                       const analysisResult = {
                         messageId: userMessage.id,
                         clusters,
                         similarityMatrix,
                         modelResponses,
                         averageSimilarity: Object.values(similarityMatrix)
                           .flatMap(row => Object.values(row))
                           .filter(val => val < 1)
                           .reduce((sum, val) => sum + val, 0) / (Object.keys(modelResponses).length * (Object.keys(modelResponses).length - 1))
                       };

                       // 유사도 분석 결과를 전역 상태에 저장
                       console.log('Saving similarity analysis result for userMessage ID:', userMessage.id);
                       console.log('Analysis result:', analysisResult);
                       
                       setMessages(prevMessages => {
                         const newMessages = { ...prevMessages };
                         if (!newMessages['_similarityData']) {
                           newMessages['_similarityData'] = {};
                         }
                         newMessages['_similarityData'][userMessage.id] = analysisResult;
                         console.log('Similarity data saved. Current _similarityData:', newMessages['_similarityData']);
                         return newMessages;
                       });
                     } catch (error) {
                       console.error('유사도 분석 오류:', error);
                     }
                   }).catch(error => {
                     console.error('유사도 분석 모듈 로드 오류:', error);
                   });
                 }
                 
                 return newMessages;
               });
             }

             // optimal 모델 처리 (다른 AI들의 응답을 포함하여)
             if (modelsToUpdate.includes('optimal')) {
               try {
                 const formData = new FormData();
                 formData.append('message', messageText || '');
                 
                 // other_responses가 있는 경우에만 추가
                 if (Object.keys(otherResponses).length > 0) {
                   formData.append('other_responses', JSON.stringify(otherResponses));
                 }
                 
                 // 파일이 있는 경우 FormData에 추가
                 if (hasFiles) {
                   // 첫 번째 파일만 전송 (백엔드에서 하나씩 처리)
                   const firstFile = filesBase64[0] || imagesBase64[0];
                   if (firstFile) {
                     // Base64를 Blob으로 변환
                     const byteCharacters = atob(firstFile.dataUrl.split(',')[1]);
                     const byteNumbers = new Array(byteCharacters.length);
                     for (let i = 0; i < byteCharacters.length; i++) {
                       byteNumbers[i] = byteCharacters.charCodeAt(i);
                     }
                     const byteArray = new Uint8Array(byteNumbers);
                     const blob = new Blob([byteArray], { type: firstFile.type });
                     formData.append('file', blob, firstFile.name);
                   }
                 }

                 const response = await fetch(`http://localhost:8000/chat/optimal/`, {
            method: 'POST',
            body: formData
          });

          if (!response.ok) {
            throw new Error('API 호출 실패');
          }

          const data = await response.json();
          
          // optimal 응답 추가 (유사도 분석 데이터 포함)
          setMessages(prevMessages => {
            const newMessages = { ...prevMessages };
            if (!newMessages['optimal']) {
              newMessages['optimal'] = [];
            }

            // 유사도 분석 데이터 가져오기 (최신 상태에서)
            const similarityData = newMessages['_similarityData'] && newMessages['_similarityData'][userMessage.id] 
              ? newMessages['_similarityData'][userMessage.id] 
              : null;

            console.log('Creating optimal message for userMessage ID:', userMessage.id);
            console.log('Available similarity data:', newMessages['_similarityData']);
            console.log('Retrieved similarity data:', similarityData);

            const optimalMessage = {
              text: data.response || "최적화된 응답을 받았습니다.",
              isUser: false,
              timestamp: new Date().toISOString(),
              id: Date.now() + Math.random() + 'optimal',
              similarityData: similarityData
            };

            newMessages['optimal'] = [...newMessages['optimal'], optimalMessage];
            return newMessages;
          });

        } catch (error) {
          // 에러 메시지 추가
          const errorMessage = {
            text: `죄송합니다. OPTIMAL 모델에서 오류가 발생했습니다. API 연결을 확인해주세요.`,
            isUser: false,
            timestamp: new Date().toISOString(),
            id: Date.now() + Math.random() + 'optimal_error'
          };

          setMessages(prevMessages => {
            const newMessages = { ...prevMessages };
            if (!newMessages['optimal']) {
              newMessages['optimal'] = [];
            }
            newMessages['optimal'] = [...newMessages['optimal'], errorMessage];
            return newMessages;
          });
        }
      }
      
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