import React, { useState } from "react";
import Sidebar from "../components/Sidebar";
import ChatBox from "../components/ChatBox";
import Settingbar from "../components/Settingbar";
import Loginbar from "../components/Loginbar";
import ModelSelectionModal from "../components/ModelSelectionModal";
import { Menu, UserCircle, Settings, CirclePlus } from "lucide-react";
import { useChat } from "../context/ChatContext";

const MainPage = () => {
  const [isSidebarVisible, setIsSidebarVisible] = useState(false);
  const [isSettingVisible, setIsSettingVisible] = useState(false);
  const [isLoginVisible, setIsLoginVisible] = useState(false);
  const [isModelModalOpen, setIsModelModalOpen] = useState(false);
  const { selectedModels, setSelectedModels } = useChat();

  const toggleSetting = () => {
    setIsSettingVisible(!isSettingVisible);
    setIsLoginVisible(false);
  };

  const toggleLogin = () => {
    setIsLoginVisible(!isLoginVisible);
    setIsSettingVisible(false);
  };

  return (
    <div className="flex flex-col h-screen bg-white relative">
      <nav className="bg-white border-b px-6 py-3 flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <Menu className="w-6 h-6 text-gray-600 cursor-pointer" onClick={() => setIsSidebarVisible(!isSidebarVisible)} />
          <h1 className="text-xl font-semibold">AI Chatbot</h1>
        </div>
        <div className="flex items-center space-x-4">
          <CirclePlus className="w-6 h-6 text-gray-600 cursor-pointer" onClick={() => setIsModelModalOpen(true)} title="AI 모델 선택" />
          <UserCircle className="w-6 h-6 text-gray-600 cursor-pointer" onClick={() => setIsLoginVisible(!isLoginVisible)} />
          <Settings className="w-6 h-6 text-gray-600 cursor-pointer" onClick={() => setIsSettingVisible(!isSettingVisible)} />
        </div>
      </nav>

      <div className="flex flex-1 min-h-0 overflow-hidden">
        {isSidebarVisible && <Sidebar />}
        <div className="flex-1 overflow-hidden">
          <ChatBox selectedModels={selectedModels} />
        </div>
      </div>

      <ModelSelectionModal 
        isOpen={isModelModalOpen} 
        onClose={() => setIsModelModalOpen(false)}
        selectedModels={selectedModels}
        onModelSelect={setSelectedModels}
      />

      {isLoginVisible && <Loginbar onClose={() => setIsLoginVisible(false)} />}
      <Settingbar isOpen={isSettingVisible} onClose={() => setIsSettingVisible(false)} />
    </div>
  );
};

export default MainPage;