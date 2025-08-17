import React, { useState } from 'react';
import WelcomePage from './pages/WelcomePage';
import MainPage from './pages/MainPage';
import { ChatProvider } from './context/ChatContext';

function App() {
  const [showWelcome, setShowWelcome] = useState(true);
  const [selectedModels, setSelectedModels] = useState([]);

  const handleStartChat = (models) => {
    setSelectedModels(models);
    setShowWelcome(false);
  };

  return (
    <div>
      {showWelcome ? (
        <WelcomePage onStartChat={handleStartChat} />
      ) : (
        <ChatProvider initialModels={selectedModels}>
          <MainPage />
        </ChatProvider>
      )}
    </div>
  );
}

export default App;