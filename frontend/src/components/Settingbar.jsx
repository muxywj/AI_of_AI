import React, { useState } from "react";
import { X } from "lucide-react";

const Settingbar = ({ isOpen, onClose }) => {
  const [isAISelectionOpen, setIsAISelectionOpen] = useState(false);
  const [isLanguageSelectionOpen, setIsLanguageSelectionOpen] = useState(false);
  const [selectedAI, setSelectedAI] = useState(null);
  const [selectedLanguage, setSelectedLanguage] = useState(null);
  const [showConfirmButton, setShowConfirmButton] = useState(false);
  const languages = [
    "Afrikaans", "Bahasa Indonesia", "Bahasa Melayu", "Català", "Čeština", "Dansk", "Deutsch", 
    "Eesti", "English (United Kingdom)", "English (United States)", "Español (España)", "Español (Latinoamérica)", 
    "Euskara", "Filipino", "Français (Canada)", "Français (France)", "Galego", "Hrvatski", "IsiZulu", "Íslenska", 
    "Italiano", "Kiswahili", "Latviešu", "Lietuvių", "Magyar", "Nederlands", "Norsk", "Polski", 
    "Português (Brasil)", "Português (Portugal)", "Română", "Slovenčina", "Slovenščina", "Suomi", "Svenska", 
    "Tiếng Việt", "Türkçe", "Ελληνικά", "Български", "Русский", "Српски", "Українська", "Հայերեն", "עברית", 
    "اردو", "العربية", "فارسی", "मराठी", "हिन्दी", "বাংলা", "ગુજરાતી", "தமிழ்", "తెలుగు", "ಕನ್ನಡ", "മലയാളം", 
    "ไทย", "한국어", "中文 (简体)", "中文 (繁體)", "日本語"
  ];

const handleConfirm = () => {
  setIsAISelectionOpen(false);
  setIsLanguageSelectionOpen(false);
  setSelectedAI(null);
  setSelectedLanguage(null);
  onClose();
};

  return (
    <>
      {/* 메인 설정 모달 */}
      {isOpen && !isAISelectionOpen && !isLanguageSelectionOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-96 shadow-lg relative flex flex-col items-center">
            <X className="absolute top-3 right-3 w-6 h-6 cursor-pointer" onClick={onClose} />
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
          <div className="bg-white rounded-lg p-6 w-full max-w-2xl shadow-lg relative pb-20">
            <h2 className="text-xl font-bold mb-4">최적화 모델 선택</h2>
            <div className="grid grid-cols-3 gap-4 mb-6">
              {["GPT-3.5", "Claude", "Mixtral"].map((model) => (
                <button
                  key={model}
                  onClick={() => setSelectedAI(model)}
                  className={`p-6 border rounded-lg transition-colors ${selectedAI === model ? "bg-blue-300" : "hover:bg-blue-50"}`}
                >
                  <h3 className="font-bold text-lg mb-2">{model}</h3>
                  <p className="text-sm text-gray-600 mb-2">{model === "GPT-3.5" ? "OpenAI의 GPT-3.5 모델" : model === "Claude" ? "Anthropic의 Claude 모델" : "Mixtral-8x7B 모델"}</p>
                  <ul className="text-xs text-gray-500 list-disc pl-4">
                    {model === "GPT-3.5" && (<>
                      <li>빠른 응답 속도</li>
                      <li>일관된 답변 품질</li>
                      <li>다양한 주제 처리</li>
                    </>)}
                    {model === "Claude" && (<>
                      <li>높은 분석 능력</li>
                      <li>정확한 정보 제공</li>
                      <li>상세한 설명 제공</li>
                    </>)}
                    {model === "Mixtral" && (<>
                      <li>균형잡힌 성능</li>
                      <li>다국어 지원</li>
                      <li>코드 분석 특화</li>
                    </>)}
                  </ul>
                </button>
              ))}
            </div>
            <button
              onClick={handleConfirm}
              className={`absolute bottom-6 right-6 px-6 py-3 rounded-lg transition-colors ${selectedAI ? "bg-blue-500 text-white hover:bg-blue-600" : "bg-gray-300 text-gray-500 cursor-not-allowed"}`}
              disabled={!selectedAI}
            >
              확인
            </button>
          </div>
        </div>
      )}
      {/* 언어 선택 모달 */}
    {isLanguageSelectionOpen && (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div className="bg-white rounded-lg p-6 w-full max-w-md shadow-lg relative h-96 overflow-y-auto flex flex-col" onScroll={(e) => setShowConfirmButton(e.target.scrollTop + e.target.clientHeight >= e.target.scrollHeight)}>
          <h2 className="text-xl font-bold mb-4">언어 선택</h2>
          <div className="grid grid-cols-2 gap-2 mb-6">
            {languages.map((lang) => (
              <button
                key={lang}
                onClick={() => setSelectedLanguage(lang)}
                className={`p-2 border rounded-lg transition-colors ${selectedLanguage === lang ? "bg-blue-300" : "hover:bg-blue-50"}`}
              >
                {lang}
              </button>
            ))}
          </div>
          <button 
            className={`px-6 py-3 rounded-lg transition-colors self-end mt-auto  shadow-md ${selectedLanguage ? "bg-blue-500 text-white hover:bg-blue-600" : "bg-gray-300 text-gray-500 cursor-not-allowed"}`} 
            onClick={handleConfirm}
            disabled={!selectedLanguage}
          >
            확인
          </button>
        </div>
      </div>
    )}
    </>
  );
};

export default Settingbar;
