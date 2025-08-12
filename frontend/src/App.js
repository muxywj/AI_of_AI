import React from 'react';
import MainPage from './pages/MainPage';
import { ChatProvider } from './context/ChatContext'; // ← 이 경로가 맞다면 그대로 사용

function App() {
  return (
    <ChatProvider>
      <MainPage />
    </ChatProvider>
  );
}

export default App;