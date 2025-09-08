import React, { useCallback } from 'react';
import { X } from 'lucide-react';

const Loginbar = ({ onClose }) => {
  // 쿼리스트링 빌더
  const buildQuery = (obj) =>
    Object.entries(obj)
      .filter(([, v]) => v !== undefined && v !== null && v !== '')
      .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(v)}`)
      .join('&');

  // ✅ 구글: 리다이렉트(Authorization Code) 방식
  const handleGoogleLogin = useCallback(() => {
    const clientId = process.env.REACT_APP_GOOGLE_CLIENT_ID;
    const redirectUri = process.env.REACT_APP_GOOGLE_REDIRECT_URI; // 예: http://localhost:3000/
    if (!clientId || !redirectUri) {
      console.warn('GOOGLE env가 비어 있습니다. REACT_APP_GOOGLE_CLIENT_ID/REDIRECT_URI 확인');
    }
    const q = buildQuery({
      client_id: clientId,
      redirect_uri: redirectUri,
      response_type: 'code',
      access_type: 'offline',
      include_granted_scopes: 'true',
      scope: ['openid', 'email', 'profile'].join(' '),
      prompt: 'consent',
    });
    window.location.href = `https://accounts.google.com/o/oauth2/v2/auth?${q}`;
  }, []);

  // ✅ 카카오: 리다이렉트 방식
  const handleKakaoLogin = useCallback(() => {
    const clientId = process.env.REACT_APP_KAKAO_CLIENT_ID;
    const redirectUri = process.env.REACT_APP_KAKAO_REDIRECT_URI; // 예: http://localhost:3000/
    if (!clientId || !redirectUri) {
      console.warn('KAKAO env가 비어 있습니다. REACT_APP_KAKAO_CLIENT_ID/REDIRECT_URI 확인');
    }
    const q = buildQuery({
      response_type: 'code',
      client_id: clientId,
      redirect_uri: redirectUri,
      scope: 'profile_nickname,account_email',
      prompt: 'login',
    });
    window.location.href = `https://kauth.kakao.com/oauth/authorize?${q}`;
  }, []);

  // ✅ 네이버: 리다이렉트 + state 저장
  const handleNaverLogin = useCallback(() => {
    const clientId = process.env.REACT_APP_NAVER_CLIENT_ID;
    const redirectUri = process.env.REACT_APP_NAVER_REDIRECT_URI; // 예: http://localhost:3000/
    if (!clientId || !redirectUri) {
      console.warn('NAVER env가 비어 있습니다. REACT_APP_NAVER_CLIENT_ID/REDIRECT_URI 확인');
    }
    const state = Math.random().toString(36).slice(2, 13);
    localStorage.setItem('naverState', state);
    const q = buildQuery({
      response_type: 'code',
      client_id: clientId,
      redirect_uri: redirectUri,
      state,
      auth_type: 'reauthenticate',
      prompt: 'consent',
      service_provider: 'NAVER',
      access_type: 'offline',
      include_granted_scopes: 'true',
    });
    window.location.href = `https://nid.naver.com/oauth2.0/authorize?${q}`;
  }, []);

  return (
    <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50 z-40">
      <div className="bg-white p-6 rounded-lg shadow-lg w-96 relative">
        <X
          className="absolute top-3 right-3 w-6 h-6 cursor-pointer text-gray-500 hover:text-gray-700"
          onClick={onClose}
        />
        <h3 className="text-xl font-bold mb-2 text-left" style={{ color: '#2d3e2c' }}>로그인</h3>
        <p className="text-sm text-gray-600 mb-4 text-left">소셜 계정으로 간편하게 로그인하세요.</p>
        <hr className="w-full border-gray-300 mb-4" />

        {/* 소셜 로그인 버튼들 */}
        <div className="w-full flex flex-col space-y-3">
          <button 
            className="w-full p-3 border border-gray-200 rounded-lg bg-white transition-colors font-medium"
            style={{ color: '#2d3e2c' }}
            onMouseEnter={(e) => {
              e.target.style.backgroundColor = 'rgba(139, 168, 138, 0.05)';
              e.target.style.borderColor = 'rgba(139, 168, 138, 0.4)';
            }}
            onMouseLeave={(e) => {
              e.target.style.backgroundColor = 'white';
              e.target.style.borderColor = '#d1d5db';
            }}
            onClick={handleGoogleLogin}
          >
            Google로 로그인
          </button>

          <button 
            className="w-full p-3 border border-gray-200 rounded-lg bg-white transition-colors font-medium"
            style={{ color: '#2d3e2c' }}
            onMouseEnter={(e) => {
              e.target.style.backgroundColor = 'rgba(139, 168, 138, 0.05)';
              e.target.style.borderColor = 'rgba(139, 168, 138, 0.4)';
            }}
            onMouseLeave={(e) => {
              e.target.style.backgroundColor = 'white';
              e.target.style.borderColor = '#d1d5db';
            }}
            onClick={handleKakaoLogin}
          >
            Kakao로 로그인
          </button>

          <button 
            className="w-full p-3 border border-gray-200 rounded-lg bg-white transition-colors font-medium"
            style={{ color: '#2d3e2c' }}
            onMouseEnter={(e) => {
              e.target.style.backgroundColor = 'rgba(139, 168, 138, 0.05)';
              e.target.style.borderColor = 'rgba(139, 168, 138, 0.4)';
            }}
            onMouseLeave={(e) => {
              e.target.style.backgroundColor = 'white';
              e.target.style.borderColor = '#d1d5db';
            }}
            onClick={handleNaverLogin}
          >
            Naver로 로그인
          </button>
        </div>
      </div>
    </div>
  );
};

export default Loginbar;