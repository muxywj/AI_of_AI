import React, { useEffect } from 'react';
import { useDispatch } from 'react-redux';
import { loginSuccess } from '../store/authSlice';

const NaverCallback = () => {
  const dispatch = useDispatch();

  useEffect(() => {
    const handleNaverCallback = async () => {
      try {
        // URL에서 코드 추출
        const urlParams = new URLSearchParams(window.location.search);
        const code = urlParams.get('code');
        const state = urlParams.get('state');
        const error = urlParams.get('error');

        if (error) {
          console.error('네이버 로그인 오류:', error);
          alert('네이버 로그인 중 오류가 발생했습니다.');
          window.location.href = '/';
          return;
        }

        if (!code) {
          console.error('네이버 인증 코드가 없습니다.');
          alert('네이버 인증 코드를 받지 못했습니다.');
          window.location.href = '/';
          return;
        }

        // 저장된 state와 비교
        const savedState = localStorage.getItem('naverState');
        if (state !== savedState) {
          console.error('네이버 state 불일치');
          alert('보안 오류가 발생했습니다.');
          window.location.href = '/';
          return;
        }

        // 코드를 액세스 토큰으로 교환
        const clientId = process.env.REACT_APP_NAVER_CLIENT_ID;
        const clientSecret = process.env.REACT_APP_NAVER_CLIENT_SECRET;
        const redirectUri = process.env.REACT_APP_NAVER_REDIRECT_URI || 'http://localhost:3000';

        if (!clientId || !clientSecret) {
          throw new Error('네이버 클라이언트 설정이 누락되었습니다.');
        }

        const tokenResponse = await fetch('https://nid.naver.com/oauth2.0/token', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
          },
          body: new URLSearchParams({
            grant_type: 'authorization_code',
            client_id: clientId,
            client_secret: clientSecret,
            redirect_uri: redirectUri,
            code: code,
            state: state,
          }),
        });

        if (!tokenResponse.ok) {
          throw new Error('네이버 토큰 교환 실패');
        }

        const tokenData = await tokenResponse.json();
        const accessToken = tokenData.access_token;

        // 백엔드로 토큰 전송
        const response = await fetch('http://localhost:8000/api/auth/naver/callback/', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            access_token: accessToken
          }),
        });

        if (response.ok) {
          const data = await response.json();
          console.log('네이버 로그인 성공:', data);

          // Redux store에 사용자 정보 저장
          dispatch(loginSuccess(data.user));

          // 로컬 스토리지에도 저장
          localStorage.setItem('user', JSON.stringify(data.user));

          // 메인 페이지로 이동
          window.location.href = '/';
        } else {
          const errorData = await response.json();
          console.error('네이버 로그인 실패:', errorData);
          alert('네이버 로그인 처리 중 오류가 발생했습니다: ' + (errorData.error || '알 수 없는 오류'));
          window.location.href = '/';
        }
      } catch (error) {
        console.error('네이버 로그인 처리 오류:', error);
        alert('네이버 로그인 중 오류가 발생했습니다: ' + error.message);
        window.location.href = '/';
      }
    };

    handleNaverCallback();
  }, [dispatch]);

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50">
      <div className="text-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-green-500 mx-auto mb-4"></div>
        <p className="text-gray-600">네이버 로그인 처리 중...</p>
      </div>
    </div>
  );
};

export default NaverCallback;