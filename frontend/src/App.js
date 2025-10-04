// src/App.js
import React, { useMemo, useState, useEffect } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { useDispatch } from 'react-redux';
import WelcomePage from './pages/WelcomePage';
import MainPage from './pages/MainPage';
import OCRToolPage from './pages/OCRToolPage';
// 👇 기존 단일 페이지 import 제거하고 아래 두 개로 교체
import VideoListPage from './pages/VideoListPage';
import VideoChatDetailPage from './pages/VideoChatDetailPage';

import KakaoCallback from './components/KakaoCallback';
import NaverCallback from './components/NaverCallback';
import { ChatProvider } from './context/ChatContext';
import { loginSuccess } from './store/authSlice';

function App() {
  const dispatch = useDispatch();

  const [showWelcome, setShowWelcome] = useState(true);
  const [selectedModels, setSelectedModels] = useState([]);

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

      {/* OCR 페이지는 Provider 필요 */}
      <Route
        path="/ocr-tool"
        element={
          <ChatProvider initialModels={[]}>
            <OCRToolPage />
          </ChatProvider>
        }
      />


      <Route path="/video-chat" element={<VideoListPage />} />
      <Route path="/video-chat/:videoId" element={<VideoChatDetailPage />} />

      {/* 소셜 로그인 콜백 */}
      <Route path="/auth/kakao/callback" element={<KakaoCallback />} />
      <Route path="/auth/naver/callback" element={<NaverCallback />} />

      {/* 그 외 경로는 홈으로 */}
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}

export default App;