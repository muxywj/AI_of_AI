// src/components/KakaoCallback.js
import { useEffect, useRef } from 'react';
import { useDispatch } from 'react-redux';
import { useNavigate, useLocation } from 'react-router-dom';
import { loginSuccess, loginFailure } from '../store/authSlice';

const KakaoCallback = () => {
  const dispatch = useDispatch();
  const navigate = useNavigate();
  const location = useLocation();
  const processedRef = useRef(false);

  useEffect(() => {
    const processKakaoLogin = async () => {
      if (processedRef.current) return;
      
      const code = new URLSearchParams(location.search).get('code');
      if (!code) return;

      processedRef.current = true;

      try {
        const response = await fetch(`http://localhost:8000/api/auth/kakao/callback/?code=${code}`);
        const data = await response.json();
        
        if (!response.ok) {
          throw new Error(data.error || '카카오 로그인 실패');
        }
        
        dispatch(loginSuccess(data.user));
        navigate('/');  // ChatInterface로 리다이렉트
      } catch (error) {
        console.error('Kakao login error:', error);
        dispatch(loginFailure(error.message));
        navigate('/');  // 에러 발생 시에도 메인 페이지로 리다이렉트
      }
    };

    processKakaoLogin();
  }, [dispatch, navigate, location]);

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-100">
      <div className="bg-white p-8 rounded-lg shadow-md">
        <div className="flex flex-col items-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-gray-900"></div>
          <h2 className="mt-4 text-xl font-semibold text-gray-700">
            카카오 로그인 처리중...
          </h2>
        </div>
      </div>
    </div>
  );
};

export default KakaoCallback;