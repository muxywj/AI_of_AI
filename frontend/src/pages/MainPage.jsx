import React, { useState } from "react";
import Sidebar from "../components/Sidebar";
import ChatBox from "../components/ChatBox";
import Settingbar from "../components/Settingbar";
import { Menu, Search, Clock, Settings } from "lucide-react";

const MainPage = () => {
  const [isSidebarVisible, setIsSidebarVisible] = useState(false);
  const [isSettingVisible, setIsSettingVisible] = useState(false);

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
          <Search className="w-5 h-5 text-gray-600" />
          <Clock className="w-5 h-5 text-gray-600" />
          <Settings
            className="w-5 h-5 text-gray-600 cursor-pointer"
            onClick={() => setIsSettingVisible(true)}
          />
        </div>
      </nav>

      {/* 메인 컨텐츠 영역 */}
      <div className="flex flex-1 min-h-0"> 
        {isSidebarVisible && <Sidebar />}
        <ChatBox />
      </div>

      {/* 설정 모달 */}
      {isSettingVisible && <Settingbar onClose={() => setIsSettingVisible(false)} />}
    </div>
  );
};

export default MainPage;