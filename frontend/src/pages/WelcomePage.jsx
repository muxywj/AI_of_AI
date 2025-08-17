import React, { useState } from 'react';
import { Plus, Sparkles, MessageCircle, Brain, Zap, X } from 'lucide-react';

const WelcomePage = ({ onStartChat }) => {
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [selectedModels, setSelectedModels] = useState([]);

  const availableModels = [
    { id: 'gpt', name: 'GPT', description: 'OpenAI의 강력한 언어 모델' },
    { id: 'claude', name: 'Claude', description: 'Anthropic의 안전하고 유용한 AI' },
    { id: 'mixtral', name: 'Mixtral', description: 'Mistral의 혼합 전문가 모델' },
    { id: 'gemini', name: 'Gemini', description: 'Google의 멀티모달 AI' },
    { id: 'llama', name: 'Llama', description: 'Meta의 오픈소스 모델' },
    { id: 'palm', name: 'PaLM', description: 'Google의 대규모 언어 모델' },
    { id: 'allama', name: 'Ollama', description: '로컬 AI 모델 실행 플랫폼' },
    { id: 'deepseek', name: 'DeepSeek', description: '딥시크의 고성능 AI 모델' },
    { id: 'bloom', name: 'BLOOM', description: '다국어 오픈소스 언어 모델' },
    { id: 'labs', name: 'AI21 Labs', description: 'AI21의 Jurassic 언어 모델' },
  ];

  const handleModelToggle = (modelId) => {
    if (selectedModels.includes(modelId)) {
      if (selectedModels.length > 1) {
        setSelectedModels(selectedModels.filter(id => id !== modelId));
      }
    } else {
      if (selectedModels.length < 3) {
        setSelectedModels([...selectedModels, modelId]);
      }
    }
  };

  const handleStartChat = () => {
    if (selectedModels.length > 0 && onStartChat) {
      onStartChat(selectedModels);
    }
  };

  const handleCloseModal = () => {
    setIsModalOpen(false);
    setSelectedModels([]);
  };

  const backgroundOverlayStyle = {
    position: 'fixed',
    top: 0,
    left: 0,
    width: '100%',
    height: '100%',
    background: `
      radial-gradient(circle at 20% 50%, rgba(139, 168, 138, 0.05) 0%, transparent 50%),
      radial-gradient(circle at 80% 20%, rgba(93, 124, 91, 0.05) 0%, transparent 50%),
      radial-gradient(circle at 40% 80%, rgba(155, 181, 154, 0.05) 0%, transparent 50%)
    `,
    pointerEvents: 'none',
    zIndex: -1
  };

  return (
    <div 
      className="h-screen w-full flex flex-col"
      style={{ 
        background: 'linear-gradient(135deg, #fefefe 0%, #f8f6f0 100%)',
        minHeight: '100vh'
      }}
    >
      {/* 배경 애니메이션 오버레이 */}
      <div style={backgroundOverlayStyle}></div>

      <style jsx>{`
        .welcome-card {
          background: rgba(255, 255, 255, 0.8);
          backdrop-filter: blur(10px);
          border: 1px solid rgba(139, 168, 138, 0.2);
          box-shadow: 0 8px 32px rgba(93, 124, 91, 0.1);
        }

        .feature-card {
          background: rgba(255, 255, 255, 0.6);
          backdrop-filter: blur(5px);
          border: 1px solid rgba(139, 168, 138, 0.15);
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .feature-card:hover {
          background: rgba(255, 255, 255, 0.9);
          border-color: rgba(139, 168, 138, 0.3);
          transform: translateY(-2px);
          box-shadow: 0 8px 25px rgba(93, 124, 91, 0.15);
        }

        .add-ai-button {
          background: linear-gradient(135deg, #5d7c5b, #8ba88a);
          color: white;
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .add-ai-button:hover {
          transform: translateY(-2px);
          box-shadow: 0 12px 32px rgba(93, 124, 91, 0.3);
        }

        .floating-particles {
          position: absolute;
          width: 100%;
          height: 100%;
          overflow: hidden;
          pointer-events: none;
        }

        .particle {
          position: absolute;
          background: rgba(139, 168, 138, 0.1);
          border-radius: 50%;
          animation: float 6s ease-in-out infinite;
        }

        @keyframes float {
          0%, 100% { transform: translateY(0px) rotate(0deg); }
          50% { transform: translateY(-20px) rotate(180deg); }
        }

        .gradient-text {
          background: linear-gradient(135deg, #5d7c5b, #8ba88a);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
        }

        .model-option {
          background: rgba(255, 255, 255, 0.8);
          border: 1px solid rgba(139, 168, 138, 0.2);
          transition: all 0.3s ease;
        }

        .model-option:hover {
          background: rgba(139, 168, 138, 0.05);
          border-color: rgba(139, 168, 138, 0.4);
        }

        .model-option.selected {
          background: rgba(139, 168, 138, 0.1);
          border-color: #8ba88a;
        }

        .close-button {
          transition: all 0.2s ease;
        }
      `}</style>

      {/* 네비게이션 바 */}
      <nav className="flex-shrink-0" style={{ 
        background: 'rgba(248, 246, 240, 0.8)', 
        backdropFilter: 'blur(10px)',
        borderBottom: '1px solid rgba(139, 168, 138, 0.15)',
        height: '60px'
      }}>
        <div className="h-full flex items-center justify-center">
          <h1 className="text-2xl font-bold gradient-text">AI OF AI</h1>
        </div>
      </nav>

      {/* 메인 컨텐츠 */}
      <div className="flex-1 flex items-center justify-center p-8 relative">
        {/* 플로팅 파티클 */}
        <div className="floating-particles">
          <div className="particle" style={{ left: '10%', top: '20%', width: '8px', height: '8px', animationDelay: '0s' }}></div>
          <div className="particle" style={{ left: '20%', top: '60%', width: '6px', height: '6px', animationDelay: '2s' }}></div>
          <div className="particle" style={{ left: '70%', top: '30%', width: '10px', height: '10px', animationDelay: '4s' }}></div>
          <div className="particle" style={{ left: '80%', top: '70%', width: '7px', height: '7px', animationDelay: '1s' }}></div>
          <div className="particle" style={{ left: '50%', top: '10%', width: '5px', height: '5px', animationDelay: '3s' }}></div>
        </div>

        <div className="welcome-card rounded-3xl p-12 max-w-4xl w-full text-center relative z-10">
          {/* 환영 메시지 */}
          <div className="mb-12">
            <div className="flex justify-center mb-6">
              <div className="p-4 rounded-full" style={{ background: 'rgba(139, 168, 138, 0.1)' }}>
                <Sparkles className="w-12 h-12" style={{ color: '#5d7c5b' }} />
              </div>
            </div>
            <h1 className="text-4xl font-bold mb-4 gradient-text">
              AI OF AI에 오신 것을 환영합니다! ✨
            </h1>
            <p className="text-lg mb-6" style={{ color: '#2d3e2c' }}>
              여러 AI 모델과 동시에 대화하며 최고의 답변을 얻어보세요
            </p>
          </div>

          {/* 기능 소개 */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12 ">
            <div className="feature-card p-6 rounded-2xl ">
              <MessageCircle className="w-8 h-8 mx-auto mb-4" style={{ color: '#5d7c5b' }} />
              <h3 className="font-semibold mb-2" style={{ color: '#2d3e2c' }}>동시 채팅</h3>
              <p className="text-sm opacity-80" style={{ color: 'rgba(45, 62, 44, 0.7)' }}>
                여러 AI와 한번에 대화
              </p>
            </div>
            <div className="feature-card p-6 rounded-2xl">
              <Brain className="w-8 h-8 mx-auto mb-4" style={{ color: '#5d7c5b' }} />
              <h3 className="font-semibold mb-2" style={{ color: '#2d3e2c' }}>최적의 답변</h3>
              <p className="text-sm opacity-80" style={{ color: 'rgba(45, 62, 44, 0.7)' }}>
                AI가 선택한 최고의 답변
              </p>
            </div>
            <div className="feature-card p-6 rounded-2xl">
              <Zap className="w-8 h-8 mx-auto mb-4" style={{ color: '#5d7c5b' }} />
              <h3 className="font-semibold mb-2" style={{ color: '#2d3e2c' }}>빠른 응답</h3>
              <p className="text-sm opacity-80" style={{ color: 'rgba(45, 62, 44, 0.7)' }}>
                효율적인 AI 활용
              </p>
            </div>
          </div>

          {/* AI 추가 버튼 */}
          <button
            onClick={() => setIsModalOpen(true)}
            className="add-ai-button px-8 py-4 rounded-2xl font-semibold text-lg flex items-center gap-3 mx-auto"
          >
            <Plus className="w-6 h-6" />
            AI 모델 선택
          </button>
        </div>
      </div>

      {/* AI 모델 선택 모달 */}
      {isModalOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50" style={{ backdropFilter: 'blur(8px)' }}>
          <div className="bg-white rounded-2xl p-8 w-full max-w-2xl max-h-[80vh] overflow-y-auto relative" style={{
            background: 'rgba(255, 255, 255, 0.95)',
            backdropFilter: 'blur(20px)',
            border: '1px solid rgba(139, 168, 138, 0.2)'
          }}>
            {/* X 버튼 */}
            <button
              onClick={handleCloseModal}
              className="close-button absolute top-4 right-4 w-8 h-8 rounded-full flex items-center justify-center"
              style={{ color: '#6b7280' }}
            >
              <X className="w-5 h-5" />
            </button>

            <h2 className="text-2xl font-bold mb-2 text-center gradient-text">
              AI 모델 선택
            </h2>
            <p className="text-center mb-8" style={{ color: 'rgba(45, 62, 44, 0.7)' }}>
              최소 1개, 최대 3개의 AI 모델을 선택하세요
            </p>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-8">
              {availableModels.map((model) => (
                <div
                  key={model.id}
                  onClick={() => handleModelToggle(model.id)}
                  className={`model-option p-4 rounded-xl cursor-pointer ${
                    selectedModels.includes(model.id) ? 'selected' : ''
                  } ${
                    selectedModels.length >= 3 && !selectedModels.includes(model.id) 
                      ? 'opacity-50 cursor-not-allowed' 
                      : ''
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="font-semibold text-lg" style={{ color: '#2d3e2c' }}>
                        {model.name}
                      </h3>
                      <p className="text-sm" style={{ color: 'rgba(45, 62, 44, 0.7)' }}>
                        {model.description}
                      </p>
                    </div>
                    {selectedModels.includes(model.id) && (
                      <div 
                        className="w-6 h-6 rounded-full flex items-center justify-center"
                        style={{ background: '#5d7c5b' }}
                      >
                        <div className="w-2 h-2 bg-white rounded-full"></div>
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>

            {/* 채팅 시작하기 버튼 */}
            <div className="flex justify-center">
              <button
                onClick={handleStartChat}
                disabled={selectedModels.length === 0}
                className={`py-3 px-8 rounded-xl font-semibold transition-colors ${
                  selectedModels.length === 0 
                    ? 'bg-gray-300 text-gray-500 cursor-not-allowed' 
                    : 'text-white'
                }`}
                style={selectedModels.length > 0 ? { backgroundColor: '#8ba88a' } : {}}
                onMouseEnter={(e) => selectedModels.length > 0 && (e.target.style.backgroundColor = '#5d7c5b')}
                onMouseLeave={(e) => selectedModels.length > 0 && (e.target.style.backgroundColor = '#8ba88a')}
              >
                채팅 시작하기 ({selectedModels.length})
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default WelcomePage;