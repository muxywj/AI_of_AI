// // // src/components/KakaoCallback.js
// // import { useEffect, useRef } from 'react';
// // import { useDispatch } from 'react-redux';
// // import { useNavigate, useLocation } from 'react-router-dom';
// // import { loginSuccess, loginFailure } from '../store/authSlice';

// // const KakaoCallback = () => {
// //   const dispatch = useDispatch();
// //   const navigate = useNavigate();
// //   const location = useLocation();
// //   const processedRef = useRef(false);
// //   useEffect(() => {
// //     const processKakaoLogin = async () => {
// //       if (processedRef.current) return;
      
// //       const code = new URLSearchParams(location.search).get('code');
// //       console.log('Received code:', code);  // 코드 로깅
// //       if (!code) return;

// //       processedRef.current = true;

// //       try {
// //         const response = await fetch(`http://localhost:8000/api/auth/kakao/callback/?code=${code}`);
// //         console.log('Response status:', response.status);  // 응답 상태 로깅
// //         const data = await response.json();
// //         console.log('Response data:', data);  // 응답 데이터 로깅
        
// //         if (!response.ok) {
// //           throw new Error(data.error || '카카오 로그인 실패');
// //         }
        
// //         dispatch(loginSuccess(data.user));
// //         navigate('/');
// //       } catch (error) {
// //         console.error('Kakao login error:', error);
// //         dispatch(loginFailure(error.message));
// //         navigate('/');
// //       }
// //     };

// //     processKakaoLogin();
// // }, [dispatch, navigate, location]);
// //   return (
// //     <div className="min-h-screen flex items-center justify-center bg-gray-100">
// //       <div className="bg-white p-8 rounded-lg shadow-md">
// //         <div className="flex flex-col items-center">
// //           <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-gray-900"></div>
// //           <h2 className="mt-4 text-xl font-semibold text-gray-700">
// //             카카오 로그인 처리중...
// //           </h2>
// //         </div>
// //       </div>
// //     </div>
// //   );
// // };

// // export default KakaoCallback;

// import React, { useEffect, useRef, useState } from 'react';
// import { useLocation, useNavigate } from 'react-router-dom';
// import { useDispatch } from 'react-redux';
// import { loginSuccess, loginFailure } from '../store/authSlice';

// const KakaoCallback = () => {
//   const location = useLocation();
//   const navigate = useNavigate();
//   const dispatch = useDispatch();
//   const processedRef = useRef(false);
//   const [loading, setLoading] = useState(true);

//   useEffect(() => {
//     const code = new URLSearchParams(location.search).get('code');
//     if (!code) {
//       setLoading(false);
//       return;
//     }
//     if (processedRef.current) return;
//     processedRef.current = true;
    
//     fetch(`http://localhost:8000/api/auth/kakao/callback/?code=${code}`, {
//       method: 'GET',
//       credentials: 'include',
//     })
//       .then(async (response) => {
//         if (!response.ok) {
//           const errorData = await response.json();
//           throw new Error(errorData.error || '카카오 로그인 실패');
//         }
//         return response.json();
//       })
//       .then((data) => {
//         // 로그인 성공 처리
//         dispatch(loginSuccess({
//           user: data.user,
//           token: data.access_token,
//         }));
//         localStorage.setItem('accessToken', data.access_token);
//         // 쿼리 파라미터 제거 후 홈으로 이동
//         navigate('/');
//         // 페이지 새로고침으로 최신 회원정보 표시
//         window.location.reload();
//       })
//       .catch((error) => {
//         console.error('카카오 로그인 에러:', error);
//         dispatch(loginFailure(error.message));
//         navigate('/');
//       })
//       .finally(() => setLoading(false));
//   }, [location, dispatch, navigate]);

//   if (loading) {
//     return (
//       <div className="min-h-screen flex items-center justify-center bg-gray-100">
//         <div className="bg-white p-8 rounded-lg shadow-md">
//           <p>로그인 처리중...</p>
//         </div>
//       </div>
//     );
//   }

//   return null;
// };

// export default KakaoCallback;
// src/components/KakaoCallback.js
import React, { useEffect, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { useDispatch } from "react-redux";
import { loginSuccess, loginFailure } from "../store/authSlice";

const KakaoCallback = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const dispatch = useDispatch();
  const processedRef = useRef(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const code = new URLSearchParams(location.search).get("code");
    if (!code) {
      setLoading(false);
      return;
    }
    if (processedRef.current) return;
    processedRef.current = true;
    
    fetch(`http://localhost:8000/api/auth/kakao/callback/?code=${code}`, {
      method: "GET",
      credentials: "include",
    })
      .then(async (response) => {
        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.error || "카카오 로그인 실패");
        }
        return response.json();
      })
      .then((data) => {
        localStorage.setItem("accessToken", data.access_token);
        localStorage.setItem("user", JSON.stringify(data.user));
        dispatch(
          loginSuccess({
            user: data.user,
            token: data.access_token,
          })
        );
        navigate("/");
      })
      .catch((error) => {
        console.error("카카오 로그인 에러:", error);
        dispatch(loginFailure(error.message));
        navigate("/");
      })
      .finally(() => {
        setLoading(false);
      });
  }, [location, dispatch, navigate]);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-100">
        <div className="bg-white p-8 rounded-lg shadow-md">
          <p>카카오 로그인 처리중...</p>
        </div>
      </div>
    );
  }

  return null;
};

export default KakaoCallback;
