// src/App.js
import React, { useMemo, useState } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom'; // Router는 index.js에서 감싸고 있으므로 여기선 Routes/Route만
import WelcomePage from './pages/WelcomePage';
import MainPage from './pages/MainPage';
import OCRToolPage from './pages/OCRToolPage';
import { ChatProvider } from './context/ChatContext';

function App() {
  // 기존 동작 유지: Welcome → onStartChat 호출 시 Main 전환
  const [showWelcome, setShowWelcome] = useState(true);
  const [selectedModels, setSelectedModels] = useState([]);

  const handleStartChat = (models) => {
    setSelectedModels(models || []);
    setShowWelcome(false);
  };

  // "/"에서 원래 조건부 렌더링 유지
  const HomeElement = useMemo(
    () =>
      showWelcome ? (
        <WelcomePage onStartChat={handleStartChat} />
      ) : (
        <ChatProvider initialModels={selectedModels}>
          <MainPage />
        </ChatProvider>
      ),
    [showWelcome, selectedModels]
  );

  return (
    <Routes>
      {/* 홈 */}
      <Route path="/" element={HomeElement} />

      {/* ✅ OCR 페이지는 useChat()을 쓰므로 Provider로 감쌈 */}
      <Route
        path="/ocr-tool"
        element={
          <ChatProvider initialModels={[]}>
            <OCRToolPage />
          </ChatProvider>
        }
      />

      {/* 필요하면 소셜 콜백 라우트 여기에 추가
      <Route path="/auth/kakao/callback" element={<KakaoCallback />} />
      <Route path="/auth/naver/callback" element={<NaverCallback />} />
      <Route path="/auth/google/callback" element={<GoogleCallback />} />
      */}

      {/* 그 외 경로는 홈으로 */}
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}

export default App;