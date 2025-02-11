import React, { useState } from "react";
import { X } from "lucide-react";
const Settingbar = ({ isOpen, onClose }) => {
  const [isAISelectionOpen, setIsAISelectionOpen] = useState(false);
  const [isLanguageSelectionOpen, setIsLanguageSelectionOpen] = useState(false);
  const languages = [
    "한국어", "English", "Español", "Français", "Deutsch", "中文", "日本語", 
    "Русский", "Português", "Italiano", "Türkçe", "हिन्दी", "العربية"
  ];

  return (
    <>
      {/* 메인 설정 모달 */}
      {isOpen && !isAISelectionOpen && !isLanguageSelectionOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-96 shadow-lg relative flex flex-col items-center">
            <X
              className="absolute top-3 right-3 w-6 h-6 cursor-pointer"
              onClick={onClose}
            />
            <h2 className="text-xl font-bold mb-4">설정</h2>
            <div className="space-y-4 w-full">
              <button
                className="w-full p-4 border rounded-lg hover:bg-blue-50 transition-colors font-bold"
                onClick={() => setIsLanguageSelectionOpen(true)}
              >
                언어 선택
              </button>
              <button
                className="w-full p-4 border rounded-lg hover:bg-blue-50 transition-colors font-bold"
                onClick={() => setIsAISelectionOpen(true)}
              >
                최적화 모델 선택
              </button>
            </div>
          </div>
        </div>
      )}

      {/* AI 선택 모달 */}
      {isAISelectionOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-2xl shadow-lg">
            <h2 className="text-xl font-bold mb-4">최적화 모델 선택</h2>
            <div className="grid grid-cols-3 gap-4 mb-6">
              {/* GPT-3.5 */}
              <button
                onClick={() => console.log("GPT-3.5 선택됨")}
                className="p-6 border rounded-lg hover:bg-blue-50 transition-colors"
              >
                <h3 className="font-bold text-lg mb-2">GPT-3.5</h3>
                <p className="text-sm text-gray-600 mb-2">OpenAI의 GPT-3.5 모델</p>
                <ul className="text-xs text-gray-500 list-disc pl-4">
                  <li>빠른 응답 속도</li>
                  <li>일관된 답변 품질</li>
                  <li>다양한 주제 처리</li>
                </ul>
              </button>

              {/* Claude */}
              <button
                onClick={() => console.log("Claude 선택됨")}
                className="p-6 border rounded-lg hover:bg-blue-50 transition-colors"
              >
                <h3 className="font-bold text-lg mb-2">Claude</h3>
                <p className="text-sm text-gray-600 mb-2">Anthropic의 Claude 모델</p>
                <ul className="text-xs text-gray-500 list-disc pl-4">
                  <li>높은 분석 능력</li>
                  <li>정확한 정보 제공</li>
                  <li>상세한 설명 제공</li>
                </ul>
              </button>

              {/* Mixtral */}
              <button
                onClick={() => console.log("Mixtral 선택됨")}
                className="p-6 border rounded-lg hover:bg-blue-50 transition-colors"
              >
                <h3 className="font-bold text-lg mb-2">Mixtral</h3>
                <p className="text-sm text-gray-600 mb-2">Mixtral-8x7B 모델</p>
                <ul className="text-xs text-gray-500 list-disc pl-4">
                  <li>균형잡힌 성능</li>
                  <li>다국어 지원</li>
                  <li>코드 분석 특화</li>
                </ul>
              </button>
            </div>
          </div>
        </div>
      )}

      {/* 언어 선택 모달 */}
      {isLanguageSelectionOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-md shadow-lg">
            <h2 className="text-xl font-bold mb-4">언어 선택</h2>
            <div className="grid grid-cols-2 gap-2 mb-6">
              {languages.map((lang, index) => (
                <button
                  key={index}
                  onClick={() => console.log(`${lang} 선택됨`)}
                  className="p-2 border rounded-lg hover:bg-blue-50 transition-colors"
                >
                  {lang}
                </button>
              ))}
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default Settingbar;