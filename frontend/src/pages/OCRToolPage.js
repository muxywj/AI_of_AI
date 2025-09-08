

import React, { useState } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";
import { useSelector, useDispatch } from "react-redux";
import { Menu, Settings, UserCircle, CirclePlus, Languages, CheckCircle, XCircle } from "lucide-react";
import { logout } from "../store/authSlice";
import Sidebar from "../components/Sidebar";
import Loginbar from "../components/Loginbar";
import Settingbar from "../components/Settingbar";
import ModelSelectionModal from "../components/ModelSelectionModal";
import { useChat } from "../context/ChatContext";
import HeaderLogo from "../components/HeaderLogo";

const OCRToolPage = () => {
  const navigate = useNavigate();
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [isPdf, setIsPdf] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [pdfPageOptions, setPdfPageOptions] = useState({
    startPage: 1,
    endPage: 0,  // 0은 전체 페이지를 의미
    totalPages: 0
  });
  
  // 분석 유형 선택 상태
  const [analysisType, setAnalysisType] = useState("both"); // "ocr", "ollama", "both"
  
  // 페이지별 분석 옵션
  const [analyzeByPage, setAnalyzeByPage] = useState(true);
  
  // 번역 옵션 추가
  const [enableTranslation, setEnableTranslation] = useState(true);
  
  // 번역 결과 표시 모드
  const [showTranslation, setShowTranslation] = useState(true);

  // MainPage에서 가져온 상태 관리
  const [isSidebarVisible, setIsSidebarVisible] = useState(false);
  const [isSettingVisible, setIsSettingVisible] = useState(false);
  const [isLoginVisible, setIsLoginVisible] = useState(false);
  const [isModelModalOpen, setIsModelModalOpen] = useState(false);
  
  const user = useSelector((state) => state.auth.user);
  const dispatch = useDispatch();
  const { selectedModels, setSelectedModels } = useChat();

  // MainPage에서 가져온 함수들
  const toggleSetting = () => {
    setIsSettingVisible(!isSettingVisible);
    setIsLoginVisible(false);
  };

  const toggleLogin = () => {
    setIsLoginVisible(!isLoginVisible);
    setIsSettingVisible(false);
  };

  const handleLogout = () => {
    localStorage.removeItem("accessToken");
    localStorage.removeItem("user");
    dispatch(logout());
    navigate("/");
  };

  // 파일 선택 처리
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      
      // PDF 파일인지 확인
      const isPdfFile = file.type === 'application/pdf';
      setIsPdf(isPdfFile);
      
      // PDF 파일이면 페이지 선택 옵션 초기화
      if (isPdfFile) {
        setPdfPageOptions({
          startPage: 1,
          endPage: 0,  // 0은 전체 페이지를 의미
          totalPages: 0  // 실제 총 페이지 수는 서버에서 확인됨
        });
        setPreview(null);
      } else {
        // 이미지만 미리보기 생성
        setPreview(URL.createObjectURL(file));
      }
      
      setError(null);
    }
  };
// OCRToolPage.js의 handleSubmit 함수에서 결과 처리 부분 수정

const handleSubmit = async (e) => {
  e.preventDefault();
  if (!selectedFile) {
    setError('파일을 선택해주세요');
    return;
  }

  const formData = new FormData();
  formData.append('file', selectedFile);
  formData.append('analysis_type', analysisType);
  formData.append('analyze_by_page', analyzeByPage.toString());
  formData.append('enable_translation', enableTranslation.toString());
  
  // PDF 페이지 선택 옵션 추가
  if (isPdf) {
    formData.append('start_page', pdfPageOptions.startPage);
    formData.append('end_page', pdfPageOptions.endPage);
  }

  setLoading(true);
  setError(null);

  try {
    const response = await axios.post('/api/ocr/process-file/', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    console.log('=== 서버 응답 데이터 확인 ===');
    console.log('전체 응답:', response.data);
    console.log('영어 원문 존재:', !!response.data.llm_response);
    console.log('한국어 번역 존재:', !!response.data.llm_response_korean);
    console.log('번역 성공:', response.data.translation_success);
    console.log('영어 원문 길이:', response.data.llm_response?.length || 0);
    console.log('한국어 번역 길이:', response.data.llm_response_korean?.length || 0);
    console.log('영어 원문 일부:', response.data.llm_response?.substring(0, 100));
    console.log('한국어 번역 일부:', response.data.llm_response_korean?.substring(0, 100));
    
    setResult(response.data);
    
    // 번역 결과에 따른 초기 표시 언어 설정 수정
    if (response.data.translation_success && response.data.llm_response_korean) {
      setShowTranslation(true);
      console.log('초기 언어 설정: 한국어 번역');
    } else {
      setShowTranslation(false);
      console.log('초기 언어 설정: 영어 원문');
    }
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


const formatLLMResponse = (response, isTranslated = false) => {
    if (!response) return isTranslated ? '번역 결과가 없습니다.' : '분석 결과가 없습니다.';
    
    try {
      // 이미지/텍스트 분석 결과 구분
      if (typeof response === 'string' && 
          (response.includes("이미지 분석 결과:") || response.includes("Image Analysis Result:")) && 
          (response.includes("텍스트 분석 결과:") || response.includes("Text Analysis Result:"))) {
        
        // 언어에 따른 구분자 설정
        const imageSeparator = isTranslated ? "이미지 분석 결과:" : "Image Analysis Result:";
        const textSeparator = isTranslated ? "텍스트 분석 결과:" : "Text Analysis Result:";
        
        // 안전하게 문자열 분리
        const parts = response.split(textSeparator);
        const imageAnalysisPart = parts[0] || '';
        const textAnalysisPart = parts[1] || '';
        
        // 각 부분 정리
        const imageAnalysis = imageAnalysisPart.replace(imageSeparator, "").trim();
        const textAnalysis = textAnalysisPart.trim();
        
        return (
          <div>
            <div className="mb-4">
              <h4 className="text-md font-medium text-gray-900 mb-2">
                {isTranslated ? "이미지 내용 분석" : "Image Content Analysis"}
              </h4>
              <div className="bg-blue-50 p-3 rounded border border-blue-200 whitespace-pre-wrap">
                {imageAnalysis || (isTranslated ? "이미지 분석 결과가 없습니다." : "No image analysis result.")}
              </div>
            </div>
            <div>
              <h4 className="text-md font-medium text-gray-900 mb-2">
                {isTranslated ? "텍스트 분석" : "Text Analysis"}
              </h4>
              <div className="bg-green-50 p-3 rounded border border-green-200 whitespace-pre-wrap">
                {textAnalysis || (isTranslated ? "텍스트 분석 결과가 없습니다." : "No text analysis result.")}
              </div>
            </div>
          </div>
        );
      }
      
      // 일반 텍스트 분석 결과 (다른 형식)
      return (
        <div className="whitespace-pre-wrap">
          {response}
        </div>
      );
    } catch (err) {
      console.error('LLM 응답 형식 오류:', err);
      // 오류 발생 시 원본 응답 반환
      return (
        <div className="whitespace-pre-wrap">
          {response}
        </div>
      );
    }
  };
const TranslationStatus = ({ translationEnabled, translationSuccess }) => {
    if (!translationEnabled) return null;
    
    return (
      <div className="flex items-center space-x-2 text-sm">
        {translationSuccess ? (
          <>
            <CheckCircle className="w-4 h-4 text-green-500" />
            <span className="text-green-600">번역 완료</span>
          </>
        ) : (
          <>
            <XCircle className="w-4 h-4 text-red-500" />
            <span className="text-red-600">번역 실패</span>
          </>
        )}
      </div>
    );
  };
const parseImageAndTextAnalysis = (analysisText, isTranslated = false) => {
    if (!analysisText) return { imageAnalysis: "", textAnalysis: "" };
    
    const imageSectionPattern = isTranslated 
      ? /이미지\s*분석\s*결과:[\s\S]*?(?=텍스트\s*분석\s*결과:|$)/i
      : /(?:Image\s*Analysis\s*Result|이미지\s*분석\s*결과):[\s\S]*?(?=(?:Text\s*Analysis\s*Result|텍스트\s*분석\s*결과):|$)/i;
      
    const textSectionPattern = isTranslated 
      ? /텍스트\s*분석\s*결과:[\s\S]*/i
      : /(?:Text\s*Analysis\s*Result|텍스트\s*분석\s*결과):[\s\S]*/i;
    
    const imageSectionMatch = analysisText.match(imageSectionPattern);
    const textSectionMatch = analysisText.match(textSectionPattern);
    
    const imageAnalysis = imageSectionMatch 
      ? imageSectionMatch[0].replace(/(?:Image\s*Analysis\s*Result|이미지\s*분석\s*결과):/i, "").trim()
      : "";
      
    const textAnalysis = textSectionMatch
      ? textSectionMatch[0].replace(/(?:Text\s*Analysis\s*Result|텍스트\s*분석\s*결과):/i, "").trim()
      : "";
      
    return {
      imageAnalysis,
      textAnalysis
    };
  };
const OCRResultDisplay = ({ ocrText, textRelevant }) => {
    // textRelevant가 false거나 undefined인 경우 관련 없는 텍스트로 간주
    const isRelevant = textRelevant === true;
    
    // OCR 텍스트가 없는 경우 표시하지 않음
    if (!ocrText || ocrText.trim() === '') {
      return null;
    }
    
    // 관련 없는 텍스트인 경우 표시하지 않음
    if (!isRelevant) {
      return null;
    }
    
    return (
      <div className="bg-gray-50 p-4 rounded-md mb-6">
        <h3 className="text-lg font-medium text-gray-900 mb-2">
          이미지에서 추출된 텍스트 (OCR)
        </h3>
        <div className="bg-white p-3 rounded border border-gray-200 max-h-60 overflow-y-auto">
          <pre className="whitespace-pre-wrap text-sm text-gray-800">
            {ocrText || '텍스트가 추출되지 않았습니다'}
          </pre>
        </div>
      </div>
    );
  };
// parsePagedAnalysis 함수 수정 - 영어와 한국어 모두 지원
const parsePagedAnalysis = (analysisText, isTranslated = false) => {
  if (!analysisText) return [];
  
  // 언어에 따른 페이지 구분 패턴 - 더 포괄적으로 수정
  const pageHeaderPatterns = [
    // 한국어 패턴
    /={3,5}\s*페이지\s*(\d+(?:-\d+)?)\s*(?:분석)?\s*={3,5}/gi,
    // 영어 패턴  
    /={3,5}\s*Page\s*(\d+(?:-\d+)?)\s*(?:Analysis)?\s*={3,5}/gi,
    // 혼합 패턴
    /={3,5}\s*(?:Page|페이지)\s*(\d+(?:-\d+)?)\s*(?:Analysis|분석)?\s*={3,5}/gi
  ];
  
  let matches = [];
  
  // 모든 패턴으로 매치 찾기
  pageHeaderPatterns.forEach(pattern => {
    let match;
    const regex = new RegExp(pattern.source, pattern.flags);
    while ((match = regex.exec(analysisText)) !== null) {
      matches.push({
        pageNum: match[1],
        position: match.index,
        length: match[0].length,
        fullMatch: match[0]
      });
    }
  });
  
  // 결과가 없으면 전체 텍스트를 하나의 페이지로 취급
  if (matches.length === 0) {
    return [{
      pageNum: isTranslated ? "전체 문서" : "Full Document",
      content: analysisText
    }];
  }
  
  // 위치순으로 정렬하고 중복 제거
  matches.sort((a, b) => a.position - b.position);
  
  // 동일한 위치의 중복 매치 제거
  matches = matches.filter((match, index) => {
    if (index === 0) return true;
    return match.position !== matches[index - 1].position;
  });
  
  // 페이지별로 내용 분할
  const pages = [];
  for (let i = 0; i < matches.length; i++) {
    const currentMatch = matches[i];
    const nextMatch = i < matches.length - 1 ? matches[i + 1] : null;
    
    const startPos = currentMatch.position + currentMatch.length;
    const endPos = nextMatch ? nextMatch.position : analysisText.length;
    
    let content = analysisText.substring(startPos, endPos).trim();
    
    // 중복 패턴 제거
    content = content.replace(/\*\*📄\s*(?:Page|페이지)\s*\d+(?:-\d+)?\*\*/gi, "");
    content = content.replace(/(?:Page|페이지)\s*\d+(?:-\d+)?$/gm, "");
    content = content.trim();
    
    if (content) {
      pages.push({
        pageNum: currentMatch.pageNum,
        content: content
      });
    }
  }
  
  return pages;
};
const LanguageToggle = ({ showTranslation, onToggle, hasTranslation }) => {
  return (
    <div className="flex items-center space-x-2 mb-4">
      <button
        onClick={() => onToggle(false)}
        className={`px-3 py-1 rounded-md text-sm transition-colors ${
          !showTranslation 
            ? "bg-blue-600 text-white" 
            : "bg-gray-200 text-gray-700 hover:bg-gray-300"
        }`}
      >
        영어 원문
      </button>
      <button
        onClick={() => onToggle(true)}
        className={`px-3 py-1 rounded-md text-sm transition-colors ${
          showTranslation 
            ? "bg-blue-600 text-white" 
            : "bg-gray-200 text-gray-700 hover:bg-gray-300"
        }`}
        disabled={!hasTranslation}
        title={!hasTranslation ? "번역이 실패했거나 비활성화되었습니다" : ""}
      >
        한국어 번역
        {!hasTranslation && (
          <span className="ml-1 text-xs">❌</span>
        )}
      </button>
      <Languages className="w-4 h-4 text-gray-500" />
    </div>
  );
};

// renderAnalysisResult 함수 수정 - 언어 전환 지원 강화
const renderAnalysisResult = () => {
  if (!result) return null;
  
  // 번역된 결과가 있고 번역 표시 모드인 경우
  const currentResponse = showTranslation && result.llm_response_korean 
    ? result.llm_response_korean 
    : result.llm_response;
  
  if (!currentResponse) return null;
  console.log('=== renderAnalysisResult 디버깅 ===');
  console.log('showTranslation:', showTranslation);
  console.log('result.llm_response_korean 존재:', !!result.llm_response_korean);
  console.log('result.llm_response 존재:', !!result.llm_response);
  console.log('currentResponse 길이:', currentResponse?.length || 0);
  console.log('currentResponse 일부:', currentResponse?.substring(0, 100));
  
  if (!currentResponse) {
    console.log('⚠️ currentResponse가 비어있음');
    return null;
  }
  // 이미지 파일인 경우 이미지/텍스트 분석을 구분
  if (result.file_type === 'image') {
    const { imageAnalysis, textAnalysis } = parseImageAndTextAnalysis(
      currentResponse, 
      showTranslation && result.llm_response_korean
    );
    
    return (
      <div className="space-y-6">
        {imageAnalysis && (
          <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
            <h3 className="text-lg font-medium text-blue-900 mb-2">
              {showTranslation && result.llm_response_korean ? "이미지 분석 결과" : "Image Analysis Result"}
            </h3>
            <div className="prose prose-blue max-w-none">
              {imageAnalysis.split("\n").map((line, idx) => (
                <p key={idx} className="mb-2">{line}</p>
              ))}
            </div>
          </div>
        )}
        
        {textAnalysis && (
          <div className="bg-purple-50 p-4 rounded-lg border border-purple-200">
            <h3 className="text-lg font-medium text-purple-900 mb-2">
              {showTranslation && result.llm_response_korean ? "텍스트 분석 결과" : "Text Analysis Result"}
            </h3>
            <div className="prose prose-purple max-w-none">
              {parsePagedAnalysis(textAnalysis, showTranslation && result.llm_response_korean).map((page, idx) => (
                <div key={idx} className="mb-6 bg-white p-3 rounded shadow-sm">
                  <h4 className="text-md font-semibold mb-2 pb-2 border-b border-purple-200">
                    {page.pageNum === "전체 요약" || page.pageNum === "Overall Summary" 
                      ? (showTranslation && result.llm_response_korean ? "전체 요약" : "Overall Summary")
                      : `${showTranslation && result.llm_response_korean ? "페이지" : "Page"} ${page.pageNum}`}
                  </h4>
                  <div>
                    {page.content.split("\n").map((line, lineIdx) => (
                      <p key={lineIdx} className="mb-1">{line}</p>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    );
  } else {
    // PDF 파일인 경우 페이지별 분석 결과 표시
    const pages = parsePagedAnalysis(currentResponse, showTranslation && result.llm_response_korean);
    
    return (
      <div className="space-y-4">
        <div className="flex justify-between items-center mb-2">
          <h3 className="text-lg font-medium text-gray-900">
            {showTranslation && result.llm_response_korean ? "페이지별 분석 결과" : "Page-by-Page Analysis Results"}
          </h3>
          {pages.length > 1 && (
            <span className="text-sm text-gray-500">
              {showTranslation && result.llm_response_korean ? `총 ${pages.length}개 섹션` : `Total ${pages.length} sections`}
            </span>
          )}
        </div>
        
        <div className="bg-gradient-to-r from-indigo-50 to-purple-50 p-6 rounded-lg border border-indigo-100">
          <div className="space-y-6">
            {pages.map((page, idx) => (
              <div 
                key={idx} 
                className={`mb-4 p-4 rounded-lg shadow-md ${
                  page.pageNum === "전체 요약" || page.pageNum === "Overall Summary"
                    ? "bg-gradient-to-r from-emerald-50 to-teal-50 border border-emerald-200" 
                    : "bg-white"
                }`}
              >
                <h4 className={`text-lg font-semibold mb-3 pb-2 border-b ${
                  page.pageNum === "전체 요약" || page.pageNum === "Overall Summary"
                    ? "text-emerald-800 border-emerald-200" 
                    : "text-indigo-800 border-indigo-100"
                }`}>
                  {page.pageNum === "전체 요약" || page.pageNum === "Overall Summary"
                    ? `📋 ${showTranslation && result.llm_response_korean ? "전체 문서 요약" : "Overall Document Summary"}`
                    : `📄 ${showTranslation && result.llm_response_korean ? "페이지" : "Page"} ${page.pageNum}`}
                </h4>
                <div className="prose max-w-none whitespace-pre-wrap">
                  {page.content}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }
};
  
  return (
    <div className="min-h-screen bg-gray-100">
      {/* MainPage에서 가져온 네비게이션 바 */}
      <nav className="bg-white border-b px-6 py-3 flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <Menu className="w-6 h-6 text-gray-600 cursor-pointer" onClick={() => setIsSidebarVisible(!isSidebarVisible)} />
          <HeaderLogo/>
        </div>
        <div className="flex items-center space-x-4">
          {user ? (
            <div className="flex items-center space-x-2">
              <span className="text-gray-600 cursor-pointer">{user.nickname || user.username}</span>
              <button onClick={handleLogout} className="text-sm text-gray-600 cursor-pointer">로그아웃</button>
              <CirclePlus className="w-6 h-6 text-gray-600 cursor-pointer" onClick={() => setIsModelModalOpen(true)} title="AI 모델 선택" />
              <Settings className="w-5 h-5 text-gray-600 cursor-pointer" onClick={toggleSetting} />
            </div>
          ) : (
            <>
              <CirclePlus className="w-6 h-6 text-gray-600 cursor-pointer" onClick={() => setIsModelModalOpen(true)} title="AI 모델 선택" />
              <UserCircle className="w-6 h-6 text-gray-600 cursor-pointer" onClick={toggleLogin} />
              <Settings className="w-6 h-6 text-gray-600 cursor-pointer" onClick={toggleSetting} />
            </>
          )}
        </div>
      </nav>

      <div className="flex">
        {isSidebarVisible && <Sidebar />}
        <div className={`${isSidebarVisible ? 'ml-64' : ''} w-full p-6 pt-20`}>
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
                  
                  {/* PDF 페이지 선택 옵션 - PDF 파일일 경우에만 표시 */}
                  {isPdf && selectedFile && (
                    <div className="bg-gray-50 p-4 rounded-lg">
                      <h4 className="text-sm font-medium text-gray-700 mb-2">PDF 페이지 선택</h4>
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <label className="block text-xs text-gray-600">시작 페이지</label>
                          <input
                            type="number"
                            min="1"
                            max={pdfPageOptions.totalPages || 9999}
                            value={pdfPageOptions.startPage}
                            onChange={(e) => setPdfPageOptions({
                              ...pdfPageOptions,
                              startPage: parseInt(e.target.value) || 1
                            })}
                            className="mt-1 block w-20 rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"
                          />
                        </div>
                        <div>
                          <label className="block text-xs text-gray-600">끝 페이지</label>
                          <input
                            type="number"
                            min={pdfPageOptions.startPage}
                            max={pdfPageOptions.totalPages || 9999}
                            value={pdfPageOptions.endPage}
                            onChange={(e) => setPdfPageOptions({
                              ...pdfPageOptions,
                              endPage: parseInt(e.target.value) || pdfPageOptions.startPage
                            })}
                            className="mt-1 block w-20 rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"
                          />
                        </div>
                      </div>
                      <p className="mt-2 text-xs text-gray-500">
                        선택한 페이지 범위만 처리합니다. 끝 페이지를 0으로 설정하면 문서 끝까지 처리합니다.
                      </p>
                    </div>
                  )}
                  
                  {/* 분석 유형 선택 */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      분석 유형 선택
                    </label>
                    <div className="flex space-x-4">
                      <button
                        type="button"
                        onClick={() => setAnalysisType("both")}
                        className={`py-2 px-4 rounded-md transition-colors ${
                          analysisType === "both" 
                            ? "bg-indigo-600 text-white" 
                            : "bg-gray-200 text-gray-700 hover:bg-gray-300"
                        }`}
                      >
                        OCR + Ollama 분석
                      </button>
                      <button
                        type="button"
                        onClick={() => setAnalysisType("ocr")}
                        className={`py-2 px-4 rounded-md transition-colors ${
                          analysisType === "ocr" 
                            ? "bg-indigo-600 text-white" 
                            : "bg-gray-200 text-gray-700 hover:bg-gray-300"
                        }`}
                      >
                        OCR만 수행
                      </button>
                      <button
                        type="button"
                        onClick={() => setAnalysisType("ollama")}
                        className={`py-2 px-4 rounded-md transition-colors ${
                          analysisType === "ollama" 
                            ? "bg-indigo-600 text-white" 
                            : "bg-gray-200 text-gray-700 hover:bg-gray-300"
                        }`}
                      >
                        Ollama만 사용
                      </button>
                    </div>
                    <p className="mt-1 text-xs text-gray-500">
                      {analysisType === "both" 
                        ? "텍스트 추출과 내용 분석을 모두 수행합니다."
                        : analysisType === "ocr" 
                        ? "텍스트 추출만 수행합니다." 
                        : "내용을 AI로 직접 분석합니다."}
                    </p>
                  </div>
                  
                  {/* 페이지별 분석 옵션 */}
                  <div className="flex items-center">
                    <input
                      id="analyze-by-page"
                      name="analyze-by-page"
                      type="checkbox"
                      checked={analyzeByPage}
                      onChange={(e) => setAnalyzeByPage(e.target.checked)}
                      className="h-4 w-4 text-indigo-600 focus:ring-indigo-500 border-gray-300 rounded"
                    />
                    <label htmlFor="analyze-by-page" className="ml-2 block text-sm text-gray-700">
                      페이지별 분석 수행 (PDF 또는 페이지 구분 가능한 텍스트)
                    </label>
                  </div>

                  {/* 번역 옵션 추가 */}
                  <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
                    <div className="flex items-center mb-2">
                      <Languages className="w-5 h-5 text-blue-600 mr-2" />
                      <h4 className="text-sm font-medium text-blue-800">번역 설정</h4>
                    </div>
                    <div className="flex items-center">
                      <input
                        id="enable-translation"
                        name="enable-translation"
                        type="checkbox"
                        checked={enableTranslation}
                        onChange={(e) => setEnableTranslation(e.target.checked)}
                        className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                      />
                      <label htmlFor="enable-translation" className="ml-2 block text-sm text-gray-700">
                        GPT로 한국어 번역 수행
                      </label>
                    </div>
                    <p className="mt-1 text-xs text-blue-600">
                      {enableTranslation 
                        ? "Ollama 분석 결과를 GPT-4로 한국어로 번역합니다."
                        : "번역을 비활성화하면 영어 원문만 제공됩니다."}
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
                      {loading ? (
                        <div className="flex items-center">
                          <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                          </svg>
                          처리 중...
                        </div>
                      ) : '파일 처리하기'}
                    </button>
                  </div>
                </div>
              </form>

              {/* 결과 표시 */}
              {result && (
                <div className="mt-8 border-t border-gray-200 pt-8">
                  <div className="flex justify-between items-center mb-4">
                    <h2 className="text-xl font-semibold text-gray-900">분석 결과</h2>
                    <TranslationStatus 
                      translationEnabled={result.translation_enabled}
                      translationSuccess={result.translation_success}
                    />
                  </div>
                  
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
                                href={result.file} 
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
                              src={result.file} 
                              alt="Uploaded" 
                              className="max-h-64 object-contain"
                            />
                          </div>
                        )
                      )}
                    </div>
                  </div>
                  
                  {/* 추출된 텍스트 - OCR이나 Both 모드일 때만 표시, 그리고 관련 있는 텍스트만 표시 */}
                  {(analysisType === "ocr" || analysisType === "both") && (
                    <OCRResultDisplay 
                      ocrText={result.ocr_text}
                      textRelevant={result.text_relevant}
                    />
                  )}
                  
                  {/* LLM 분석 - Ollama나 Both 모드일 때만 표시 */}
               {/* LLM 분석 - Ollama나 Both 모드일 때만 표시 */}
{(analysisType === "ollama" || analysisType === "both") && (result.llm_response || result.llm_response_korean) && (
  <div className="bg-gray-50 p-4 rounded-md mb-6">
    <div className="flex justify-between items-center mb-4">
      <h3 className="text-lg font-medium text-gray-900">Ollama 분석 결과</h3>
      
      {/* 언어 전환 버튼 - 번역이 활성화되었고 영어 원문이 있으면 항상 표시 */}
      {result.translation_enabled && result.llm_response && (
        <LanguageToggle 
          showTranslation={showTranslation}
          onToggle={setShowTranslation}
          hasTranslation={result.translation_success && result.llm_response_korean}
        />
      )}
    </div>
    
    <div className="bg-white p-3 rounded border border-gray-200">
      <div className="prose prose-sm max-w-none text-gray-800">
        {/* 번역 실패 시 안내 메시지 추가 */}
        {result.translation_enabled && !result.translation_success && showTranslation && (
          <div className="mb-4 p-3 bg-yellow-50 border border-yellow-200 rounded-md">
            <p className="text-yellow-800 text-sm">
              ⚠️ 번역에 실패했습니다. 영어 원문을 표시합니다.
            </p>
          </div>
        )}
        
        {result.file_type === 'pdf'
          ? renderAnalysisResult()
          : formatLLMResponse(
              // 번역 실패 시 영어 원문 표시
              (showTranslation && result.llm_response_korean && result.translation_success) 
                ? result.llm_response_korean 
                : result.llm_response,
              showTranslation && result.llm_response_korean && result.translation_success
            )}
      </div>
    </div>
  </div>
)}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* MainPage에서 가져온 모달 컴포넌트들 */}
      {isModelModalOpen && (
        <ModelSelectionModal 
          isOpen={isModelModalOpen} 
          onClose={() => setIsModelModalOpen(false)}
          selectedModels={selectedModels}
          onModelSelect={setSelectedModels}
        />
      )}
      {isLoginVisible && <Loginbar onClose={() => setIsLoginVisible(false)} />}
      {isSettingVisible && <Settingbar isOpen={isSettingVisible} onClose={() => setIsSettingVisible(false)} />}
    </div>
  );
};

export default OCRToolPage;