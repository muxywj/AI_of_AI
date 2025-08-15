import React from "react";

const Sidebar = () => {
  const quickPrompts = [
    { title: "코드 작성 도움", desc: "웹사이트의 스타일리시한 헤더를 위한 코드" },
    { title: "이미지 생성", desc: "취침 시간 이야기와 그림 만들기" },
    { title: "텍스트 분석", desc: "이력서를 위한 강력한 문구 생성" },
    { title: "문제 해결", desc: "빠른 문제 해결 방법 제안" }
  ];

  return (
    <div 
      className="w-64 border-r p-4 h-full flex-shrink-0 transition-all duration-300"
      style={{
        background: 'rgba(245, 242, 234, 0.4)',
        backdropFilter: 'blur(20px)',
        borderRightColor: 'rgba(139, 168, 138, 0.15)'
      }}
    >
      <h2 
        className="text-lg font-semibold mb-4"
        style={{
          color: '#2d3e2c',
          fontSize: '1.2rem',
          fontWeight: 600
        }}
      >
        메뉴
      </h2>
      <div className="space-y-3">
        {quickPrompts.map((prompt, index) => (
          <div 
            key={index} 
            className="sidebar-item p-3 rounded-lg cursor-pointer transition-all duration-400 relative overflow-hidden"
            style={{
              background: 'rgba(255, 255, 255, 0.8)',
              backdropFilter: 'blur(10px)',
              border: '1px solid rgba(139, 168, 138, 0.2)',
              borderRadius: '16px'
            }}
          >
            {/* 반짝이는 애니메이션 효과 */}
            <div 
              className="shine-effect absolute top-0 left-0 w-full h-full pointer-events-none"
              style={{
                background: 'linear-gradient(90deg, transparent, rgba(139, 168, 138, 0.1), transparent)',
                transform: 'translateX(-100%)',
                transition: 'transform 0.6s ease'
              }}
            />
            
            <h3 
              className="font-medium text-sm mb-1 transition-colors duration-300"
              style={{
                color: '#5d7c5b',
                fontSize: '0.95rem',
                fontWeight: 600
              }}
            >
              {prompt.title}
            </h3>
            <p 
              className="text-xs mt-1 transition-colors duration-300"
              style={{
                color: 'rgba(45, 62, 44, 0.5)',
                fontSize: '0.8rem',
                lineHeight: 1.4
              }}
            >
              {prompt.desc}
            </p>
          </div>
        ))}
      </div>

      <style jsx>{`
        .sidebar-item:hover {
          background: rgba(255, 255, 255, 0.9) !important;
          border-color: #8ba88a !important;
          transform: translateY(-4px) scale(1.02);
          box-shadow: 0 12px 40px rgba(93, 124, 91, 0.15);
        }

        .sidebar-item:hover .shine-effect {
          transform: translateX(100%);
        }

        .sidebar-item:hover h3 {
          color: #5d7c5b !important;
        }

        .sidebar-item:hover p {
          color: rgba(45, 62, 44, 0.7) !important;
        }

        /* 추가 애니메이션 효과 */
        .sidebar-item {
          position: relative;
        }

        .sidebar-item::before {
          content: '';
          position: absolute;
          top: 0;
          left: -100%;
          width: 100%;
          height: 100%;
          background: linear-gradient(90deg, transparent, rgba(139, 168, 138, 0.1), transparent);
          transition: left 0.6s;
          pointer-events: none;
        }

        .sidebar-item:hover::before {
          left: 100%;
        }

        /* 메뉴 제목에 그라데이션 효과 추가 (선택사항) */
        .menu-title-gradient {
          background: linear-gradient(135deg, #2d3e2c, #5d7c5b);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
        }
      `}</style>
    </div>
  );
};

export default Sidebar;