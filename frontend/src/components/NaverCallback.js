// src/components/NaverCallback.js
import React, { useEffect, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { useDispatch } from "react-redux";
import { loginSuccess, loginFailure } from "../store/authSlice";

const NaverCallback = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const dispatch = useDispatch();
  const processedRef = useRef(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const params = new URLSearchParams(location.search);
    const code = params.get("code");
    const state = params.get("state");
    if (!code || !state) {
      setLoading(false);
      return;
    }
    if (processedRef.current) return;
    processedRef.current = true;
    fetch(`http://localhost:8000/auth/naver/callback/?code=${code}&state=${state}`, {
      method: "GET",
      headers: {
        "Authorization": `Bearer ${code.access_token}`,
        "Content-Type": "application/json",
      },
      credentials: "include",
    })
      .then(async (response) => {
        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.error || "네이버 로그인 실패");
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
        console.error("네이버 로그인 에러:", error);
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
          <p>네이버 로그인 처리중...</p>
        </div>
      </div>
    );
  }

  return null;
};

export default NaverCallback;
