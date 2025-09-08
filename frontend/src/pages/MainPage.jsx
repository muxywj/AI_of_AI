// src/pages/MainPage.js
import React, { useState } from "react";
import { useSelector, useDispatch } from "react-redux";
import Loginbar from "../components/Loginbar";
import Settingbar from "../components/Settingbar";
import Sidebar from "../components/Sidebar";
import ChatBox from "../components/ChatBox";
import { Menu, Settings, UserCircle, CirclePlus } from "lucide-react";
import { logout } from "../store/authSlice";
import { useNavigate } from "react-router-dom";
import ModelSelectionModal from "../components/ModelSelectionModal";
import { useChat } from "../context/ChatContext";

const MainPage = () => {
  const [isSidebarVisible, setIsSidebarVisible] = useState(false);
  const [isSettingVisible, setIsSettingVisible] = useState(false);
  const [isLoginVisible, setIsLoginVisible] = useState(false);
  const [isModelModalOpen, setIsModelModalOpen] = useState(false);
  const { selectedModels, setSelectedModels } = useChat();

  const user = useSelector((state) => state.auth.user);
  const dispatch = useDispatch();
  const navigate = useNavigate();

  const toggleSetting = () => {
    setIsSettingVisible((v) => !v);
    setIsLoginVisible(false);
  };

  const toggleLogin = () => {
    setIsLoginVisible((v) => !v);
    setIsSettingVisible(false);
  };

  const handleLogout = () => {
    localStorage.removeItem("accessToken");
    localStorage.removeItem("user");
    dispatch(logout());
    navigate("/");
  };

  // 배경 애니메이션 스타일
  const backgroundOverlayStyle = {
    position: "fixed",
    top: 0,
    left: 0,
    width: "100%",
    height: "100%",
    background: `
      radial-gradient(circle at 20% 50%, rgba(139, 168, 138, 0.05) 0%, transparent 50%),
      radial-gradient(circle at 80% 20%, rgba(93, 124, 91, 0.05) 0%, transparent 50%),
      radial-gradient(circle at 40% 80%, rgba(155, 181, 154, 0.05) 0%, transparent 50%)
    `,
    pointerEvents: "none",
    zIndex: -1,
  };

  const displayName = user?.nickname || user?.username || "";

  return (
    <div
      className="flex flex-col h-screen relative"
      style={{
        background: "linear-gradient(135deg, #fefefe 0%, #f8f6f0 100%)",
        color: "#2d3e2c",
        overflowX: "hidden",
      }}
    >
      {/* 배경 애니메이션 오버레이 */}
      <div style={backgroundOverlayStyle} />

      <nav
        className="border-b px-6 py-3 flex items-center justify-between sticky top-0 z-100"
        style={{
          background: "rgba(248, 246, 240, 0.8)",
          backdropFilter: "blur(20px)",
          borderBottomColor: "rgba(139, 168, 138, 0.2)",
          boxShadow: "0 8px 32px rgba(93, 124, 91, 0.1)",
        }}
      >
        <div className="flex items-center space-x-4">
          <Menu
            className="w-6 h-6 text-gray-600 cursor-pointer transition-all duration-300 hover:scale-110"
            onClick={() => setIsSidebarVisible((v) => !v)}
          />
          <h1
            className="text-xl font-semibold cursor-pointer hover:opacity-80 transition-opacity"
            style={{ color: "#5d7c5b", fontWeight: 800 }}
          >
            AI OF AI
          </h1>
        </div>

        {/* ▼ 여기부터 조건부 렌더(user ? ... : ...) */}
        <div className="flex items-center space-x-4">
          {user ? (
            // 로그인된 상태
            <div className="flex items-center space-x-3">
              <span className="text-gray-700 font-medium">
                {displayName}
              </span>

              <button
                onClick={handleLogout}
                className="text-sm text-gray-600 hover:text-gray-800 transition-colors"
                title="로그아웃"
              >
                로그아웃
              </button>

              <CirclePlus
                className="w-6 h-6 text-gray-600 cursor-pointer transition-all duration-300 hover:scale-110"
                onClick={() => setIsModelModalOpen(true)}
                title="AI 모델 선택"
              />
              <Settings
                className="w-6 h-6 text-gray-600 cursor-pointer transition-all duration-300 hover:scale-110"
                onClick={toggleSetting}
                title="설정"
              />
            </div>
          ) : (
            // 로그인되지 않은 상태
            <>
              <CirclePlus
                className="w-6 h-6 text-gray-600 cursor-pointer transition-all duration-300 hover:scale-110"
                onClick={() => setIsModelModalOpen(true)}
                title="AI 모델 선택"
              />
              <UserCircle
                className="w-6 h-6 text-gray-600 cursor-pointer transition-all duration-300 hover:scale-110"
                onClick={toggleLogin}
                title="로그인"
              />
              <Settings
                className="w-6 h-6 text-gray-600 cursor-pointer transition-all duration-300 hover:scale-110"
                onClick={toggleSetting}
                title="설정"
              />
            </>
          )}
        </div>
        {/* ▲ 조건부 렌더 끝 */}
      </nav>

      <div className="flex flex-1 min-h-0 overflow-hidden">
        {isSidebarVisible && <Sidebar />}
        <div className="flex-1 overflow-hidden">
          <ChatBox />
        </div>
      </div>

      <ModelSelectionModal
        isOpen={isModelModalOpen}
        onClose={() => setIsModelModalOpen(false)}
        selectedModels={selectedModels}
        onModelSelect={setSelectedModels}
      />

      {isLoginVisible && (
        <Loginbar onClose={() => setIsLoginVisible(false)} />
      )}
      <Settingbar
        isOpen={isSettingVisible}
        onClose={() => setIsSettingVisible(false)}
      />
    </div>
  );
};

export default MainPage;