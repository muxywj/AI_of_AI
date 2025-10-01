// src/App.js
import React, { useMemo, useState, useEffect } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom'; // Router는 index.js에서 감싸고 있으므로 여기선 Routes/Route만
import { useDispatch } from 'react-redux';
import WelcomePage from './pages/WelcomePage';
import MainPage from './pages/MainPage';
import OCRToolPage from './pages/OCRToolPage';
import VideoChatPage from './pages/VideoChatPage';
import KakaoCallback from './components/KakaoCallback';
import NaverCallback from './components/NaverCallback';
import { ChatProvider } from './context/ChatContext';
import { loginSuccess } from './store/authSlice';

function App() {
  const dispatch = useDispatch();
  
  // 기존 동작 유지: Welcome → onStartChat 호출 시 Main 전환
  const [showWelcome, setShowWelcome] = useState(true);
  const [selectedModels, setSelectedModels] = useState([]);

  // 앱 시작 시 로컬 스토리지에서 사용자 정보 복원
  useEffect(() => {
    const savedUser = localStorage.getItem('user');
    if (savedUser) {
      try {
        const userData = JSON.parse(savedUser);
        dispatch(loginSuccess(userData));
      } catch (error) {
        console.error('사용자 정보 복원 실패:', error);
        localStorage.removeItem('user');
      }
    }
  }, [dispatch]);

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

      {/* 영상 채팅 페이지 */}
      <Route path="/video-chat/:videoId?" element={<VideoChatPage />} />

      {/* 소셜 로그인 콜백 라우트 */}
      <Route path="/auth/kakao/callback" element={<KakaoCallback />} />
      <Route path="/auth/naver/callback" element={<NaverCallback />} />

      {/* 그 외 경로는 홈으로 */}
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}

export default App;