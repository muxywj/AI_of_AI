import React, { useState, useEffect } from 'react';
import WelcomePage from './pages/WelcomePage';
import MainPage from './pages/MainPage';
import { ChatProvider } from './context/ChatContext';

function App() {
  const [showWelcome, setShowWelcome] = useState(true);
  const [initialModels, setInitialModels] = useState([]);

  const handleStartChat = (models) => {
    setInitialModels(models);
    setShowWelcome(false);
  };

  return (
    <ChatProvider initialModels={initialModels}>
      {showWelcome ? (
        <WelcomePage onStartChat={handleStartChat} />
      ) : (
        <MainPage />
      )}
    </ChatProvider>
  );
}

export default App;