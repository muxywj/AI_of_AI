import React from 'react';

const ModelSelectionModal = ({ isOpen, onClose, selectedModels, onModelSelect }) => {
  const availableModels = [
    { id: 'gpt', name: 'GPT' },
    { id: 'claude', name: 'Claude' },
    { id: 'mixtral', name: 'Mixtral' },
    { id: 'gemini', name: 'Gemini' },
    { id: 'llama', name: 'Llama' },
    { id: 'palm', name: 'PaLM' },
    { id: 'allama', name: 'Ollama' },
    { id: 'deepseek', name: 'DeepSeek' },
    { id: 'bloom', name: 'BLOOM' },
    { id: 'labs', name: 'AI21 Labs' },
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
      <div className="bg-white rounded-lg w-80 max-h-[80vh] flex flex-col">
        <div className="p-6">
          <h2 className="text-xl font-semibold mb-4">AI 모델 선택</h2>
          <p className="text-sm text-gray-600 mb-4">최소 1개, 최대 3개까지 선택 가능합니다.</p>
        </div>
        
        <div className="flex-1 overflow-y-auto px-6">
          <div className="space-y-3 pb-4">
            {availableModels.map((model) => (
              <label
                key={model.id}
                className={`flex items-center p-3 rounded-lg border cursor-pointer transition-colors
                  ${selectedModels.includes(model.id) 
                    ? 'border-purple-500 bg-purple-50' 
                    : 'border-gray-200 hover:bg-gray-50'}
                  ${selectedModels.length >= 3 && !selectedModels.includes(model.id) 
                    ? 'opacity-50 cursor-not-allowed' 
                    : ''}`}
              >
                <input
                  type="checkbox"
                  checked={selectedModels.includes(model.id)}
                  onChange={() => handleModelToggle(model.id)}
                  disabled={selectedModels.length >= 3 && !selectedModels.includes(model.id)}
                  className="hidden"
                />
                <div className="flex-1">{model.name}</div>
                {selectedModels.includes(model.id) && (
                  <div className="w-4 h-4 rounded-full bg-purple-500"></div>
                )}
              </label>
            ))}
          </div>
        </div>

        <div className="p-6 border-t">
          <button
            onClick={onClose}
            className="w-full py-2 px-4 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors"
          >
            확인
          </button>
        </div>
      </div>
    </div>
  );
};

export default ModelSelectionModal;
