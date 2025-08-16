import React from 'react';
import { X } from 'lucide-react';

const ModelSelectionModal = ({ isOpen, onClose, selectedModels, onModelSelect, onConfirm }) => {
  const availableModels = [
    { id: 'gpt', name: 'GPT', description: 'OpenAI의 강력한 언어 모델' },
    { id: 'claude', name: 'Claude', description: 'Anthropic의 안전하고 유용한 AI' },
    { id: 'gemini', name: 'Gemini', description: 'Google의 멀티모달 AI' },
    { id: 'mixtral', name: 'Mixtral', description: 'Mistral의 혼합 전문가 모델' },
    { id: 'llama', name: 'Llama', description: 'Meta의 오픈소스 모델' },
    { id: 'palm', name: 'PaLM', description: 'Google의 대규모 언어 모델' },
    { id: 'allama', name: 'Ollama', description: '로컬 AI 모델 실행 플랫폼' },
    { id: 'deepseek', name: 'DeepSeek', description: '딥시크의 고성능 AI 모델' },
    { id: 'bloom', name: 'BLOOM', description: '다국어 오픈소스 언어 모델' },
    { id: 'labs', name: 'AI21 Labs', description: 'AI21의 Jurassic 언어 모델' },
  ];

  const handleModelToggle = (modelId) => {
    if (selectedModels.includes(modelId)) {
      // Remove model if already selected
      if (selectedModels.length > 1) { // Ensure at least one model remains selected
        onModelSelect(selectedModels.filter(id => id !== modelId));
      }
    } else {
      // Add model if not selected and less than 3 models are currently selected
      if (selectedModels.length < 3) {
        onModelSelect([...selectedModels, modelId]);
      }
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg w-80 max-h-[60vh] flex flex-col relative"> 
        <div className="p-6">
          <X className="absolute top-3 right-3 w-6 h-6 cursor-pointer text-gray-500 hover:text-gray-700" onClick={onClose} />
          <h3 className="text-xl font-bold mb-2 text-left" style={{ color: '#2d3e2c' }}>AI 모델 선택</h3>
          <p className="text-sm text-gray-600 mb-0.1 text-left">기본 응답을 제공할 AI 모델을 선택하세요.<br/>(최소 1개, 최대 3개)</p>
        </div>
        
        <div className="flex-1 overflow-y-auto px-6 border-t">
          <div className="space-y-3 pb-4 pt-6">
            {availableModels.map((model) => (
              <label
                key={model.id}
                className={`flex items-center p-3 rounded-lg border cursor-pointer transition-all duration-200
                  ${selectedModels.includes(model.id) 
                    ? '' 
                    : 'border-gray-200'}
                  ${selectedModels.length >= 3 && !selectedModels.includes(model.id) 
                    ? 'opacity-50 cursor-not-allowed' 
                    : ''}`}
                style={selectedModels.includes(model.id) 
                  ? { 
                      borderColor: 'rgba(139, 168, 138, 0.4)', 
                      backgroundColor: 'rgba(139, 168, 138, 0.05)' 
                    } 
                  : {}}
                onMouseEnter={(e) => {
                  if (!selectedModels.includes(model.id) && selectedModels.length < 3) {
                    e.currentTarget.style.backgroundColor = 'rgba(139, 168, 138, 0.05)';
                    e.currentTarget.style.borderColor = 'rgba(139, 168, 138, 0.4)';
                  }
                }}
                onMouseLeave={(e) => {
                  if (!selectedModels.includes(model.id)) {
                    e.currentTarget.style.backgroundColor = 'transparent';
                    e.currentTarget.style.borderColor = '#d1d5db';
                  }
                }}
              >
                <input
                  type="checkbox"
                  checked={selectedModels.includes(model.id)}
                  onChange={() => handleModelToggle(model.id)}
                  disabled={selectedModels.length >= 3 && !selectedModels.includes(model.id)}
                  className="hidden"
                />
                <div className="flex-1">
                  <div style={{ color: '#2d3e2c' }}>{model.name}</div>
                  <div className="text-xs mt-1" style={{ color: '#6b7280' }}>{model.description}</div>
                </div>
                {selectedModels.includes(model.id) && (
                  <div 
                    className="w-4 h-4 rounded-full flex items-center justify-center ml-2"
                    style={{ background: '#5d7c5b' }}
                  >
                    <div className="w-1.5 h-1.5 bg-white rounded-full"></div>
                  </div>
                )}
              </label>
            ))}
          </div>
        </div>

        <div className="p-6 border-t flex justify-center">
          <button
            onClick={onClose}
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
            확인
          </button>
        </div>
      </div>
    </div>
  );
};

export default ModelSelectionModal;