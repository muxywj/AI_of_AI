import React, { useState } from "react";
import { Menu } from "lucide-react";

const Sidebar = () => {
  const [isSidebarVisible, setIsSidebarVisible] = useState(false);

  const quickPrompts = [
    { title: "코드 작성 도움", desc: "웹사이트의 스타일리시한 헤더를 위한 코드" },
    { title: "이미지 생성", desc: "취침 시간 이야기와 그림 만들기" },
    { title: "텍스트 분석", desc: "이력서를 위한 강력한 문구 생성" },
    { title: "문제 해결", desc: "빠른 문제 해결 방법 제안" },
  ];

  return (
    <div>
      {/* 메뉴 버튼 (고정) */}
      <div className="fixed top-4 left-4 z-50">
        <Menu 
          className="w-6 h-6 text-gray-600 cursor-pointer" 
          onClick={() => setIsSidebarVisible(!isSidebarVisible)}
        />
      </div>
      
      {/* 왼쪽 사이드바 (조금 아래로 이동) */}
      <div 
        className={`w-64 border-r bg-gray-50 p-4 fixed left-0 top-12 h-full shadow-lg transition-transform duration-300 ${isSidebarVisible ? "translate-x-0" : "-translate-x-full"}`}
      >
        <h2 className="text-lg font-semibold mb-4">메뉴</h2>
        <div className="space-y-3">
          {quickPrompts.map((prompt, index) => (
            <div 
              key={index} 
              className="p-3 bg-white rounded-lg shadow-sm hover:shadow-md cursor-pointer"
            >
              <h3 className="font-medium text-sm">{prompt.title}</h3>
              <p className="text-xs text-gray-600 mt-1">{prompt.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Sidebar;
