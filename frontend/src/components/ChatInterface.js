// src/components/ChatInterface.js 상단의 import 및 변수 선언 부분만 수정
import React, { useState, useRef, useEffect } from 'react';
<<<<<<< Updated upstream
=======
<<<<<<< HEAD
import { Send, Menu, Search, Clock, Settings, X} from 'lucide-react';
=======
>>>>>>> Stashed changes
import { Send, Menu, Search, Clock, Settings, X } from 'lucide-react';
import { useDispatch, useSelector } from 'react-redux';
import { useGoogleLogin } from '@react-oauth/google';
import { loginSuccess, loginFailure } from '../store/authSlice';
<<<<<<< Updated upstream
=======
>>>>>>> 4a47049b6c71701a6a5c214835ba99eddadbc2ac
>>>>>>> Stashed changes

const ChatInterface = () => {
  const [messages, setMessages] = useState({
    gpt: [],
    claude: [],
    mixtral: [],
  });
  const [inputMessage, setInputMessage] = useState('');
<<<<<<< Updated upstream
  const [isLoading, setLoading] = useState(false);
=======
<<<<<<< HEAD
  const [isLoading, setIsLoading] = useState(false);
=======
  const [isLoading, setLoading] = useState(false);
>>>>>>> 4a47049b6c71701a6a5c214835ba99eddadbc2ac
>>>>>>> Stashed changes
  const [isSidebarVisible, setIsSidebarVisible] = useState(false);
  const [isLoginModalOpen, setIsLoginModalOpen] = useState(false);
  const messagesEndRef = useRef(null);
  
  const dispatch = useDispatch();
  const { user } = useSelector((state) => state.auth);  // error 제거

  const quickPrompts = [
    { title: "코드 작성 도움", desc: "웹사이트의 스타일리시한 헤더를 위한 코드" },
    { title: "이미지 생성", desc: "취침 시간 이야기와 그림 만들기" },
    { title: "텍스트 분석", desc: "이력서를 위한 강력한 문구 생성" },
    { title: "문제 해결", desc: "빠른 문제 해결 방법 제안" },
  ];
<<<<<<< HEAD
=======

  const handleKakaoLogin = () => {
    const kakaoAuthUrl = `https://kauth.kakao.com/oauth/authorize?response_type=code&client_id=${process.env.REACT_APP_KAKAO_CLIENT_ID}&redirect_uri=${process.env.REACT_APP_KAKAO_REDIRECT_URI}&scope=profile_nickname,account_email&prompt=login`;
    window.location.href = kakaoAuthUrl;
  };

  const googleLogin = useGoogleLogin({
    onSuccess: async (codeResponse) => {
      setLoading(true);
      try {
        const backendResponse = await fetch('http://localhost:8000/api/auth/google/callback/', {
          method: 'GET',
          headers: {
            'Authorization': `Bearer ${codeResponse.access_token}`,
            'Content-Type': 'application/json',
          },
          credentials: 'include',
        });

        if (!backendResponse.ok) {
          const errorData = await backendResponse.json();
          throw new Error(errorData.error || '로그인 실패');
        }

        const data = await backendResponse.json();
        dispatch(loginSuccess(data.user));
        setIsLoginModalOpen(false);  // 로그인 성공 시 모달 닫기
      } catch (error) {
        console.error('로그인 에러:', error);
        dispatch(loginFailure(error.message));
      } finally {
        setLoading(false);
      }
    },
    onError: (error) => {
      console.error('로그인 실패:', error);
      dispatch(loginFailure('구글 로그인 실패'));
    },
  });
>>>>>>> 4a47049b6c71701a6a5c214835ba99eddadbc2ac

  const handleKakaoLogin = () => {
    const kakaoAuthUrl = `https://kauth.kakao.com/oauth/authorize?response_type=code&client_id=${process.env.REACT_APP_KAKAO_CLIENT_ID}&redirect_uri=${process.env.REACT_APP_KAKAO_REDIRECT_URI}&scope=profile_nickname,account_email&prompt=login`;
    window.location.href = kakaoAuthUrl;
  };

  const googleLogin = useGoogleLogin({
    onSuccess: async (codeResponse) => {
      setLoading(true);
      try {
        const backendResponse = await fetch('http://localhost:8000/api/auth/google/callback/', {
          method: 'GET',
          headers: {
            'Authorization': `Bearer ${codeResponse.access_token}`,
            'Content-Type': 'application/json',
          },
          credentials: 'include',
        });

        if (!backendResponse.ok) {
          const errorData = await backendResponse.json();
          throw new Error(errorData.error || '로그인 실패');
        }

        const data = await backendResponse.json();
        dispatch(loginSuccess(data.user));
        setIsLoginModalOpen(false);  // 로그인 성공 시 모달 닫기
      } catch (error) {
        console.error('로그인 에러:', error);
        dispatch(loginFailure(error.message));
      } finally {
        setLoading(false);
      }
    },
    onError: (error) => {
      console.error('로그인 실패:', error);
      dispatch(loginFailure('구글 로그인 실패'));
    },
  });

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!inputMessage.trim()) return;

    const userMessage = inputMessage;
    setInputMessage('');

    const updatedMessages = {};
    setLoading(true);

    for (const botName of ['gpt', 'claude', 'mixtral']) {
      try {
        const response = await fetch(`http://localhost:8000/chat/${botName}/`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
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
        console.error('Error:', error);
        updatedMessages[botName] = [
          ...(messages[botName] || []),
          { text: userMessage, isUser: true },
          {
            text: `오류가 발생했습니다: ${error.message}`,
            isUser: false,
          },
        ];
      }
    }

    setMessages((prev) => ({
      ...prev,
      ...updatedMessages,
    }));

    setLoading(false);
  };

  return (
    <div className="flex flex-col h-screen bg-white">
      {/* 상단 네비게이션 바 */}
      <nav className="bg-white border-b px-6 py-3 flex items-center justify-between">
        <div className="flex items-center space-x-4">
<<<<<<< Updated upstream
=======
<<<<<<< HEAD
          <Menu className="w-6 h-6 text-gray-600 cursor-pointer" onClick={() => setIsSidebarVisible(!isSidebarVisible)} />
=======
>>>>>>> Stashed changes
          <Menu 
            className="w-6 h-6 text-gray-600 cursor-pointer" 
            onClick={() => setIsSidebarVisible(!isSidebarVisible)} 
          />
<<<<<<< Updated upstream
=======
>>>>>>> 4a47049b6c71701a6a5c214835ba99eddadbc2ac
>>>>>>> Stashed changes
          <h1 className="text-xl font-semibold">AI Chatbot</h1>
        </div>
        <div className="flex items-center space-x-4">
          <Search className="w-5 h-5 text-gray-600" />
          <Clock className="w-5 h-5 text-gray-600" />
<<<<<<< Updated upstream
=======
<<<<<<< HEAD
          <Settings className="w-5 h-5 text-gray-600 cursor-pointer" onClick={() => setIsLoginModalOpen(true)} />
=======
>>>>>>> Stashed changes
          <Settings 
            className="w-5 h-5 text-gray-600 cursor-pointer" 
            onClick={() => setIsLoginModalOpen(true)} 
          />
<<<<<<< Updated upstream
=======
>>>>>>> 4a47049b6c71701a6a5c214835ba99eddadbc2ac
>>>>>>> Stashed changes
        </div>
      </nav>

      {/* 메인 컨텐츠 영역 */}
      <div className="flex flex-1">
        {/* 왼쪽 사이드바 */}
        {isSidebarVisible && (
          <div className="w-64 border-r bg-gray-50 p-4">
            <h2 className="text-lg font-semibold mb-4">메뉴</h2>
            <div className="space-y-3">
              {quickPrompts.map((prompt, index) => (
                <div key={index} className="p-3 bg-white rounded-lg shadow-sm hover:shadow cursor-pointer">
                  <h3 className="font-medium text-sm">{prompt.title}</h3>
                  <p className="text-xs text-gray-600 mt-1">{prompt.desc}</p>
                </div>
              ))}
            </div>
          </div>
        )}
<<<<<<< Updated upstream
=======
<<<<<<< HEAD
=======
>>>>>>> Stashed changes
{/* 로그인 모달 */}
{isLoginModalOpen && (
  <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50 z-50">
    <div className="bg-white p-6 rounded-lg shadow-lg w-96 relative flex flex-col items-center">
      <X className="absolute top-3 right-3 w-6 h-6 cursor-pointer" onClick={() => setIsLoginModalOpen(false)} />
      <h2 className="text-2xl font-bold">AI OF AI</h2>
      <p className="text-sm text-gray-600 mb-4">AI 통합 기반 답변 최적화 플랫폼</p>

      {user ? (
        // 로그인된 상태
        <div className="w-full space-y-4">
          <div className="bg-green-100 border border-green-400 text-green-700 p-4 rounded">
            <p>환영합니다, {user.nickname || user.username}님!</p>
            <p>이메일: {user.email}</p>
          </div>
          <button
            onClick={() => setIsLoginModalOpen(false)}
            className="w-full bg-indigo-600 text-white p-2 rounded hover:bg-indigo-700"
          >
            계속하기
          </button>
      
   
        </div>
      ) : (
        // 로그인되지 않은 상태
        <>
          <button onClick={googleLogin} disabled={isLoading} className="w-full bg-gray-300 p-2 rounded mb-2">
            {isLoading ? "로그인 중..." : "Google로 로그인"}
          </button>
          <button onClick={handleKakaoLogin} disabled={isLoading} className="w-full bg-gray-300 p-2 rounded mb-2">
            Kakao로 로그인
          </button>
          <button className="w-full bg-gray-300 p-2 rounded mb-4">Naver로 로그인</button>
          <hr className="w-full border-gray-400 mb-4" />
          <input type="email" placeholder="이메일" className="w-full p-2 border rounded mb-2" />
          <input type="password" placeholder="비밀번호" className="w-full p-2 border rounded mb-2" />
          <div className="text-xs text-gray-600 flex justify-between w-full">
            <span>비밀번호를 잊으셨나요?</span> <span className="text-blue-500 cursor-pointer">비밀번호 찾기</span>
          </div>
          <button className="w-full bg-gray-800 text-white p-2 rounded mt-4">로그인</button>
        </>
      )}
    </div>
  </div>
)}

        {/* 로그인 모달
        {isLoginModalOpen && (
           <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50 z-50">
           <div className="bg-white p-6 rounded-lg shadow-lg w-96 relative flex flex-col items-center">
             <X className="absolute top-3 right-3 w-6 h-6 cursor-pointer" onClick={() => setIsLoginModalOpen(false)} />
             <h2 className="text-2xl font-bold">AI OF AI</h2>
             <p className="text-sm text-gray-600 mb-4">AI 통합 기반 답변 최적화 플랫폼</p>
       
              
              {user ? (
                <div className="w-full space-y-4">
                  <div className="bg-green-100 border border-green-400 text-green-700 p-4 rounded">
                    <p>환영합니다, {user.nickname || user.username}님!</p>
                    <p>이메일: {user.email}</p>
                  </div>
                  <button
                    onClick={() => setIsLoginModalOpen(false)}
                    className="w-full bg-indigo-600 text-white p-2 rounded hover:bg-indigo-700"
                  >
                    계속하기
                  </button>
                </div>
              ) : (
                <div className="w-full space-y-4">
                  <button onClick={googleLogin} disabled={isLoading} className="w-full bg-gray-300 p-2 rounded mb-2">
        {isLoading ? "로그인 중..." : "Google로 로그인"}
      </button>
      <button onClick={handleKakaoLogin} disabled={isLoading} className="w-full bg-gray-300 p-2 rounded mb-2">
        Kakao로 로그인
      </button>
      <button className="w-full bg-gray-300 p-2 rounded mb-4">Naver로 로그인</button>

      <hr className="w-full border-gray-400 mb-4" />
                  
      <input type="email" placeholder="이메일" className="w-full p-2 border rounded mb-2" />
      <input type="password" placeholder="비밀번호" className="w-full p-2 border rounded mb-2" />
      <div className="text-xs text-gray-600 flex justify-between w-full">
        <span>비밀번호를 잊으셨나요?</span>
        <span className="text-blue-500 cursor-pointer">비밀번호 찾기</span>
      </div>
                    <button className="w-full bg-gray-800 text-white p-2 rounded">
                      로그인
                    </button>
                
                </div>
              )}
            </div>
          </div>
        )} */}
<<<<<<< Updated upstream
=======
>>>>>>> 4a47049b6c71701a6a5c214835ba99eddadbc2ac
>>>>>>> Stashed changes

        {/* 채팅 영역 */}
        <div className="flex-1 grid grid-cols-3">
          {['gpt', 'claude', 'mixtral'].map((botName) => (
            <div key={botName} className="flex flex-col border-r last:border-r-0">
              <div className="p-4 border-b">
                <h2 className="text-xl font-semibold">{botName}</h2>
              </div>
              <div className="flex-1 overflow-y-auto p-6 space-y-6">
                {messages[botName].map((message, index) => (
                  <div
                    key={index}
                    className={`flex ${message.isUser ? 'justify-end' : 'justify-start'}`}
                  >
                    <div
                      className={`max-w-2xl p-4 rounded-2xl ${
                        message.isUser
                          ? 'bg-purple-600 text-white'
                          : 'bg-gray-100 text-gray-800'
                      }`}
                    >
                      {message.text}
                    </div>
                  </div>
                ))}
                {isLoading && (
                  <div className="flex justify-start">
                    <div className="bg-gray-100 text-gray-800 p-4 rounded-2xl">
                      입력 중...
                    </div>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* 입력 영역 */}
      <div className="border-t p-4">
        <form onSubmit={handleSubmit} className="max-w-4xl mx-auto">
          <div className="flex items-center bg-white border rounded-xl p-2">
            <input
              type="text"
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              placeholder="메시지를 입력하세요..."
              className="flex-1 px-3 py-2 focus:outline-none"
            />
            <button
              type="submit"
              disabled={isLoading}
              className="p-2 rounded-lg hover:bg-gray-100"
            >
              <Send className="w-5 h-5 text-gray-600" />
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default ChatInterface;