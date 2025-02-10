import React, { useState } from "react";
import Sidebar from "../components/Sidebar";
import ChatBox from "../components/ChatBox";
import Settingbar from "../components/Settingbar"; // ✅ default import 가능
import Loginbar from "../components/Loginbar";
import { Menu, Search, Clock, Settings, UserCircle } from "lucide-react";

const MainPage = () => {
  const [isSidebarVisible, setIsSidebarVisible] = useState(false);
  const [isSettingVisible, setIsSettingVisible] = useState(false);
  const [isLoginVisible, setIsLoginVisible] = useState(false);

  // 설정창과 로그인창이 동시에 열리지 않도록 조정
  const toggleSetting = () => {
    setIsSettingVisible(!isSettingVisible);
    setIsLoginVisible(false); // 설정창을 열 때 로그인창은 닫기
  };

  const toggleLogin = () => {
    setIsLoginVisible(!isLoginVisible);
    setIsSettingVisible(false); // 로그인창을 열 때 설정창은 닫기
  };

  return (
    <div className="flex flex-col h-screen bg-white relative">
      {/* 상단 네비게이션 바 */}
      <nav className="bg-white border-b px-6 py-3 flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <Menu
            className="w-6 h-6 text-gray-600 cursor-pointer"
            onClick={() => setIsSidebarVisible(!isSidebarVisible)}
          />
          <h1 className="text-xl font-semibold">AI Chatbot</h1>
        </div>
        <div className="flex items-center space-x-4">
          {/*<Search className="w-5 h-5 text-gray-600" />
          <Clock className="w-5 h-5 text-gray-600" />*/}
          <UserCircle
            className="w-5 h-5 text-gray-600 cursor-pointer"
            onClick={toggleLogin}
          />
          <Settings
            className="w-5 h-5 text-gray-600 cursor-pointer"
            onClick={toggleSetting}
          />
        </div>
      </nav>

      {/* 메인 컨텐츠 영역 */}
      <div className="flex flex-1 min-h-0 overflow-hidden"> 
        {isSidebarVisible && <Sidebar />}
        <ChatBox />
      </div>

      {/* 로그인 모달 */}
      {isLoginVisible && <Loginbar onClose={() => setIsLoginVisible(false)} />}

      {/* 설정 모달 */}
      <Settingbar isOpen={isSettingVisible} onClose={() => setIsSettingVisible(false)} />
    </div>
  );
};

export default MainPage;