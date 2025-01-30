// src/components/Login.js
import { useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { useNavigate } from 'react-router-dom';
import { useGoogleLogin } from '@react-oauth/google';
import { loginSuccess, loginFailure } from '../store/authSlice';

const Login = () => {
  const dispatch = useDispatch();
  const navigate = useNavigate();
  const { error, user } = useSelector((state) => state.auth);
  const [loading, setLoading] = useState(false);

  const login = useGoogleLogin({
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
        setLoading(false);
      } catch (error) {
        console.error('로그인 에러:', error);
        dispatch(loginFailure(error.message));
        setLoading(false);
      }
    },
    onError: (error) => {
      console.error('로그인 실패:', error);
      dispatch(loginFailure('구글 로그인 실패'));
    },
  });

  const handleChatClick = () => {
    navigate('/chat');
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-md w-full space-y-8">
        <div>
          <h2 className="mt-6 text-center text-3xl font-extrabold text-gray-900">
            계정에 로그인하세요
          </h2>
        </div>
        {error && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative">
            {error}
          </div>
        )}
        {user ? (
          <div className="space-y-4">
            <div className="bg-green-100 border border-green-400 text-green-700 px-4 py-3 rounded relative">
              <p>환영합니다, {user.username}님!</p>
              <p>이메일: {user.email}</p>
            </div>
            <button
              onClick={handleChatClick}
              className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
            >
              채팅 시작하기
            </button>
          </div>
        ) : (
          <div className="mt-8 space-y-6">
            <button
              onClick={() => login()}
              disabled={loading}
              className="group relative w-full flex justify-center py-2 px-4 border border-transparent text-sm font-medium rounded-md text-white bg-red-600 hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500 disabled:opacity-50"
            >
              {loading ? '로그인 중...' : '구글로 로그인'}
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default Login;