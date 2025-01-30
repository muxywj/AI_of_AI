// src/App.js
import React from 'react';
import { Provider } from 'react-redux';
import { GoogleOAuthProvider } from '@react-oauth/google';
import { BrowserRouter as Router } from 'react-router-dom';
import ChatInterface from './components/ChatInterface';
import { store } from './store';

function App() {
  return (
    <Provider store={store}>
      <GoogleOAuthProvider clientId={process.env.REACT_APP_GOOGLE_CLIENT_ID}>
        <Router>
          <div className="App">
            <ChatInterface />
          </div>
        </Router>
      </GoogleOAuthProvider>
    </Provider>
  );
}

export default App;