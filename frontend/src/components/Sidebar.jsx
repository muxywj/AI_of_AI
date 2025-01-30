import React from "react";

const Sidebar = () => {
  const quickPrompts = [
    { title: "코드 작성 도움", desc: "웹사이트의 스타일리시한 헤더를 위한 코드" },
    { title: "이미지 생성", desc: "취침 시간 이야기와 그림 만들기" },
    { title: "텍스트 분석", desc: "이력서를 위한 강력한 문구 생성" },
    { title: "문제 해결", desc: "빠른 문제 해결 방법 제안" },
  ];

  return (
    <div className="w-64 border-r bg-gray-50 p-4 absolute left-0 top-16 h-full shadow-lg">
      <h2 className="text-lg font-semibold mb-4">메뉴</h2>
      <div className="space-y-3">
        {quickPrompts.map((prompt, index) => (
          <div key={index} className="p-3 bg-white rounded-lg shadow-sm hover:shadow-md cursor-pointer">
            <h3 className="font-medium text-sm">{prompt.title}</h3>
            <p className="text-xs text-gray-600 mt-1">{prompt.desc}</p>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Sidebar;