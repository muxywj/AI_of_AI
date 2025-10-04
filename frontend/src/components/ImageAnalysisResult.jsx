// src/components/ImageAnalysisResult.jsx
import React, { useState } from 'react';
import { FileText, Image as ImageIcon, List, AlertCircle, Check, X } from 'lucide-react';

const ImageAnalysisResult = ({ result, modelResponses }) => {
  const [expandedModel, setExpandedModel] = useState(null);
  
  if (!result) return null;
  
  const { 
    bestResponse, 
    analysis, 
    reasoning, 
    imageAnalysisMode = 'describe',
    botName
  } = result;
  
  // 분석 모드에 따른 아이콘과 타이틀 설정
  const getModeIcon = () => {
    switch (imageAnalysisMode) {
      case 'ocr':
        return <FileText className="text-purple-500" />;
      case 'objects':
        return <List className="text-green-500" />;
      default:
        return <ImageIcon className="text-blue-500" />;
    }
  };
  
  const getModeTitle = () => {
    switch (imageAnalysisMode) {
      case 'ocr':
        return '텍스트 추출 결과';
      case 'objects':
        return '객체 인식 결과';
      default:
        return '이미지 설명 결과';
    }
  };
  
  // 응답 콘텐츠 포맷 (코드 블록, 리스트 등 처리)
  const formatContent = (content) => {
    if (!content) return null;
    
    // 코드 블록 처리
    const codeBlockRegex = /```(.*?)\n([\s\S]*?)```/g;
    let formattedContent = content;
    let match;
    
    // 코드 블록 변환
    while ((match = codeBlockRegex.exec(content)) !== null) {
      const language = match[1] || '';
      const code = match[2];
      
      const formattedBlock = (
        `<div class="bg-gray-100 rounded-md p-3 my-2 font-mono text-sm overflow-auto">
          ${code.replace(/\n/g, '<br/>')}
        </div>`
      );
      
      formattedContent = formattedContent.replace(match[0], formattedBlock);
    }
    
    // 목록 처리
    formattedContent = formattedContent.replace(
      /^(\s*[-*+]\s+.*?)(?=\n[-*+]|\n\n|$)/gm, 
      '<li class="ml-4">$1</li>'
    );
    formattedContent = formattedContent.replace(/<li/g, '<ul><li');
    formattedContent = formattedContent.replace(/li><ul>/g, 'li>');
    formattedContent = formattedContent.replace(/li>\n/g, 'li></ul>\n');
    
    // 새 줄 처리
    formattedContent = formattedContent.replace(/\n\n/g, '<br/><br/>');
    formattedContent = formattedContent.replace(/\n/g, '<br/>');
    
    return (
      <div 
        className="prose max-w-none"
        dangerouslySetInnerHTML={{ __html: formattedContent }} 
      />
    );
  };
  
  // 모델별 응답 토글
  const toggleModelResponse = (modelId) => {
    if (expandedModel === modelId) {
      setExpandedModel(null);
    } else {
      setExpandedModel(modelId);
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-md overflow-hidden border">
      {/* 헤더 */}
      <div className="bg-blue-50 p-4 border-b flex items-center justify-between">
        <div className="flex items-center gap-2">
          {getModeIcon()}
          <h3 className="font-semibold text-lg">{getModeTitle()}</h3>
        </div>
        <div className="text-sm text-gray-500">
          분석 엔진: {botName || '자동'}
        </div>
      </div>
      
      {/* 최적의 응답 */}
      <div className="p-4 border-b">
        <div className="font-medium mb-2 flex items-center gap-2">
          <Check className="text-green-500" />
          <span>최적의 응답</span>
        </div>
        <div className="bg-gray-50 p-3 rounded-md">
          {formatContent(bestResponse)}
        </div>
      </div>
      
      {/* 각 모델별 응답 요약 */}
      <div className="p-4 border-b">
        <div className="font-medium mb-2">모델별 응답 분석</div>
        <div className="space-y-2">
          {Object.entries(analysis || {}).map(([modelId, modelAnalysis]) => (
            <div key={modelId} className="bg-gray-50 p-2 rounded-md">
              <div 
                className="flex justify-between items-center cursor-pointer"
                onClick={() => toggleModelResponse(modelId)}
              >
                <div className="font-medium">{modelId.toUpperCase()}</div>
                {expandedModel === modelId ? (
                  <X size={16} className="text-gray-500" />
                ) : (
                  <span className="text-xs text-blue-500">자세히 보기</span>
                )}
              </div>
              
              {/* 분석 내용 표시 */}
              <div className="mt-1 text-sm">
                <div className="text-green-600">장점: {modelAnalysis?.장점 || '정보 없음'}</div>
                <div className="text-red-600">단점: {modelAnalysis?.단점 || '정보 없음'}</div>
              </div>
              
              {/* 확장된 모델 응답 표시 */}
              {expandedModel === modelId && modelResponses && modelResponses[modelId] && (
                <div className="mt-3 pt-3 border-t">
                  <div className="text-sm font-medium mb-1">전체 응답:</div>
                  <div className="text-sm bg-white p-2 rounded">
                    {formatContent(modelResponses[modelId])}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
      
      {/* 분석 근거 */}
      <div className="p-4">
        <div className="font-medium mb-2 flex items-center gap-2">
          <AlertCircle className="text-blue-500" size={16} />
          <span>분석 근거</span>
        </div>
        <div className="bg-gray-50 p-3 rounded-md text-sm">
          {formatContent(reasoning) || '분석 근거가 제공되지 않았습니다.'}
        </div>
      </div>
    </div>
  );
};

export default ImageAnalysisResult;