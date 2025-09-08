
import React, { useState } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";
import Sidebar from "../components/Sidebar";

const OCRToolPage = () => {
  const navigate = useNavigate();
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [isPdf, setIsPdf] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  
  // Ollama 모델 선택 상태
  const [selectedModels, setSelectedModels] = useState(['llama3', 'phi3', 'gemma']);
  
  // 사용 가능한 Ollama 모델 목록
  const availableModels = [
    { id: 'llama3', name: 'Llama 3' },
    { id: 'phi3', name: 'Phi-3' },
    { id: 'gemma', name: 'Gemma' },
    { id: 'mistral', name: 'Mistral' },
    { id: 'codellama', name: 'CodeLlama' }
  ];

  // 파일 선택 처리
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      
      // PDF 파일인지 확인
      const isPdfFile = file.type === 'application/pdf';
      setIsPdf(isPdfFile);
      
      // 이미지만 미리보기 생성
      if (!isPdfFile) {
        setPreview(URL.createObjectURL(file));
      } else {
        setPreview(null); // PDF의 경우 미리보기 초기화
      }
      
      setError(null);
    }
  };

  // 모델 선택 토글
  const handleModelToggle = (modelId) => {
    setSelectedModels(prev => {
      if (prev.includes(modelId)) {
        // 최소 하나의 모델은 선택되어 있어야 함
        if (prev.length <= 1) return prev;
        return prev.filter(id => id !== modelId);
      } else {
        return [...prev, modelId];
      }
    });
  };

  // 폼 제출 처리
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!selectedFile) {
      setError('파일을 선택해주세요');
      return;
    }

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('selectedModels', JSON.stringify(selectedModels));
    formData.append('language', 'ko'); // 언어 설정

    setLoading(true);
    setError(null);

    try {
      const response = await axios.post('/api/ocr/process-file/', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setResult(response.data);
    } catch (err) {
      console.error('파일 처리 오류:', err);
      setError(
        err.response?.data?.error || 
        '파일 처리 중 오류가 발생했습니다. 다시 시도해주세요.'
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100">
      <div className="flex">
        <Sidebar />
        <div className="ml-64 w-full p-6 pt-20">
          <div className="max-w-4xl mx-auto">
            <h1 className="text-2xl font-bold mb-6">OCR 및 LLM 텍스트 분석 도구</h1>
            
            <div className="bg-white shadow rounded-lg p-6">
              <form onSubmit={handleSubmit}>
                <div className="space-y-6">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      이미지 또는 PDF 업로드
                    </label>
                    <div className="mt-1 flex justify-center px-6 pt-5 pb-6 border-2 border-gray-300 border-dashed rounded-md">
                      <div className="space-y-1 text-center">
                        {preview ? (
                          <div className="flex justify-center">
                            <img 
                              src={preview} 
                              alt="Preview" 
                              className="h-64 object-contain"
                            />
                          </div>
                        ) : isPdf && selectedFile ? (
                          <div className="flex flex-col items-center justify-center">
                            <svg 
                              className="h-16 w-16 text-red-500" 
                              fill="currentColor" 
                              viewBox="0 0 20 20"
                            >
                              <path fillRule="evenodd" d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4zm2 6a1 1 0 011-1h6a1 1 0 110 2H7a1 1 0 01-1-1zm1 3a1 1 0 100 2h6a1 1 0 100-2H7z" clipRule="evenodd" />
                            </svg>
                            <p className="mt-2 text-sm text-gray-600">
                              {selectedFile.name}
                            </p>
                          </div>
                        ) : (
                          <svg
                            className="mx-auto h-12 w-12 text-gray-400"
                            stroke="currentColor"
                            fill="none"
                            viewBox="0 0 48 48"
                            aria-hidden="true"
                          >
                            <path
                              d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02"
                              strokeWidth={2}
                              strokeLinecap="round"
                              strokeLinejoin="round"
                            />
                          </svg>
                        )}
                        <div className="flex text-sm text-gray-600 justify-center">
                          <label
                            htmlFor="file-upload"
                            className="relative cursor-pointer bg-white rounded-md font-medium text-indigo-600 hover:text-indigo-500 focus-within:outline-none focus-within:ring-2 focus-within:ring-offset-2 focus-within:ring-indigo-500"
                          >
                            <span>파일 업로드</span>
                            <input
                              id="file-upload"
                              name="file-upload"
                              type="file"
                              className="sr-only"
                              accept="image/*,.pdf"
                              onChange={handleFileChange}
                            />
                          </label>
                          <p className="pl-1">또는 드래그 앤 드롭</p>
                        </div>
                        <p className="text-xs text-gray-500">PNG, JPG, GIF, PDF (최대 10MB)</p>
                      </div>
                    </div>
                  </div>

                  {/* Ollama 모델 선택 UI */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      분석에 사용할 Ollama 모델 선택
                    </label>
                    <div className="flex flex-wrap gap-2">
                      {availableModels.map(model => (
                        <button
                          key={model.id}
                          type="button"
                          onClick={() => handleModelToggle(model.id)}
                          className={`px-3 py-1.5 text-sm rounded-full transition-colors ${
                            selectedModels.includes(model.id)
                              ? 'bg-indigo-600 text-white'
                              : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                          }`}
                        >
                          {model.name}
                        </button>
                      ))}
                    </div>
                    <p className="text-xs text-gray-500 mt-1">
                      여러 모델을 선택하면 다양한 관점에서 분석 결과를 얻을 수 있습니다.
                    </p>
                  </div>

                  {error && (
                    <div className="rounded-md bg-red-50 p-4">
                      <div className="flex">
                        <div className="ml-3">
                          <h3 className="text-sm font-medium text-red-800">오류</h3>
                          <div className="text-sm text-red-700">
                            <p>{error}</p>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}

                  <div>
                    <button
                      type="submit"
                      className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                      disabled={loading}
                    >
                      {loading ? '처리 중...' : '파일 처리하기'}
                    </button>
                  </div>
                </div>
              </form>

              {/* 결과 표시 */}
              {result && (
                <div className="mt-8 border-t border-gray-200 pt-8">
                  <h2 className="text-xl font-semibold text-gray-900 mb-4">분석 결과</h2>
                  
                  {/* 원본 파일 */}
                  <div className="bg-gray-50 p-4 rounded-md mb-6">
                    <h3 className="text-lg font-medium text-gray-900 mb-2">
                      원본 {result.file_type === 'pdf' ? 'PDF' : '이미지'}
                    </h3>
                    <div className="bg-white p-3 rounded border border-gray-200">
                      {result.file_type === 'pdf' ? (
                        <div className="flex items-center justify-center">
                          <div className="p-4 border border-gray-300 rounded-md bg-gray-50 text-center">
                            <svg className="h-12 w-12 text-red-500 mx-auto" fill="currentColor" viewBox="0 0 20 20">
                              <path fillRule="evenodd" d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4zm2 6a1 1 0 011-1h6a1 1 0 110 2H7a1 1 0 01-1-1zm1 3a1 1 0 100 2h6a1 1 0 100-2H7z" clipRule="evenodd" />
                            </svg>
                            <p className="mt-2 text-sm text-gray-600">PDF 문서</p>
                            {result.file && (
                              <a 
                                href={axios.defaults.baseURL + result.file} 
                                target="_blank" 
                                rel="noopener noreferrer"
                                className="mt-2 inline-flex items-center px-3 py-1 border border-transparent text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700"
                              >
                                PDF 보기
                              </a>
                            )}
                          </div>
                        </div>
                      ) : (
                        result.file && (
                          <div className="flex justify-center">
                            <img 
                              src={axios.defaults.baseURL + result.file} 
                              alt="Uploaded" 
                              className="max-h-64 object-contain"
                            />
                          </div>
                        )
                      )}
                    </div>
                  </div>
                  
                  {/* 추출된 텍스트 */}
                  <div className="bg-gray-50 p-4 rounded-md mb-6">
                    <h3 className="text-lg font-medium text-gray-900 mb-2">추출된 텍스트</h3>
                    <div className="bg-white p-3 rounded border border-gray-200 max-h-60 overflow-y-auto">
                      <pre className="whitespace-pre-wrap text-sm text-gray-800">
                        {result.ocr_text || '텍스트가 추출되지 않았습니다'}
                      </pre>
                    </div>
                  </div>
                  
                  {/* Ollama LLM 분석 */}
                  <div className="bg-gray-50 p-4 rounded-md mb-6">
                    <h3 className="text-lg font-medium text-gray-900 mb-2">Ollama LLM 분석</h3>
                    <div className="bg-white p-3 rounded border border-gray-200">
                      <div className="prose prose-sm max-w-none text-gray-800">
                        {result.llm_response || 'LLM 분석을 사용할 수 없습니다'}
                      </div>
                    </div>
                  </div>
                  
                  {/* 모델별 응답 (확장된 결과) */}
                  {result.model_responses && Object.keys(result.model_responses).length > 0 && (
                    <div className="bg-gray-50 p-4 rounded-md">
                      <h3 className="text-lg font-medium text-gray-900 mb-2">각 모델별 응답</h3>
                      <div className="space-y-4">
                        {Object.entries(result.model_responses).map(([modelId, response]) => {
                          // 모델 이름 가져오기
                          const modelInfo = availableModels.find(m => m.id === modelId) || { id: modelId, name: modelId.toUpperCase() };
                          
                          return (
                            <div key={modelId} className="bg-white p-4 rounded border border-gray-200">
                              <h4 className="font-medium text-gray-900 mb-2 flex items-center">
                                <span className="h-3 w-3 rounded-full bg-green-500 mr-2"></span>
                                {modelInfo.name}
                              </h4>
                              <div className="prose prose-sm max-w-none text-gray-800">
                                {typeof response === 'string' && !response.startsWith('오류:') ? (
                                  <div className="whitespace-pre-wrap">{response}</div>
                                ) : (
                                  <div className="text-red-500">{response}</div>
                                )}
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default OCRToolPage;