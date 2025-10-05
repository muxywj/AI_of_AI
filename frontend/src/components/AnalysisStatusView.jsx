import React from 'react';
import { Loader2, XCircle, RefreshCw, Clock } from 'lucide-react';

const AnalysisStatusView = ({ 
  status, 
  progress, 
  message, 
  onRetry, 
  onBackToList 
}) => {
  if (status === 'pending' || status === 'analyzing') {
    return (
      <div className="flex flex-col items-center justify-center min-h-[500px] bg-gradient-to-br from-blue-50 to-indigo-50 rounded-xl border border-blue-200 shadow-lg">
        <div className="relative mb-6">
          <div className="w-20 h-20 bg-blue-100 rounded-full flex items-center justify-center shadow-lg">
            <Loader2 className="w-10 h-10 text-blue-500 animate-spin" />
          </div>
          <div className="absolute inset-0 w-20 h-20 bg-blue-200 rounded-full animate-ping opacity-20"></div>
        </div>
        
        <h3 className="text-2xl font-bold text-gray-800 mb-3">🎬 영상 분석 중</h3>
        
        <div className="bg-white/80 backdrop-blur-sm rounded-xl p-6 mb-6 shadow-md max-w-md w-full">
          <div className="text-center mb-4">
            <p className="text-gray-700 font-medium mb-2">
              {message || '영상을 분석하고 있습니다...'}
            </p>
            <div className="flex items-center justify-center text-sm text-gray-500">
              <Clock className="w-4 h-4 mr-1" />
              <span>잠시만 기다려주세요</span>
            </div>
          </div>
          
          <div className="mb-4">
            <div className="flex justify-between text-sm text-gray-600 mb-2">
              <span className="font-medium">분석 진행률</span>
              <span className="font-bold text-blue-600">{progress}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-4 shadow-inner">
              <div 
                className="bg-gradient-to-r from-blue-500 via-blue-600 to-indigo-600 h-4 rounded-full transition-all duration-700 ease-out shadow-sm relative overflow-hidden" 
                style={{ width: `${progress}%` }}
              >
                <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white to-transparent opacity-30 animate-pulse"></div>
              </div>
            </div>
          </div>
          
          <div className="space-y-2">
            <div className={`flex items-center text-sm ${progress >= 20 ? 'text-green-600' : 'text-gray-400'}`}>
              <div className={`w-2 h-2 rounded-full mr-2 ${progress >= 20 ? 'bg-green-500' : 'bg-gray-300'}`}></div>
              프레임 추출 중...
            </div>
            <div className={`flex items-center text-sm ${progress >= 50 ? 'text-green-600' : 'text-gray-400'}`}>
              <div className={`w-2 h-2 rounded-full mr-2 ${progress >= 50 ? 'bg-green-500' : 'bg-gray-300'}`}></div>
              객체 감지 중...
            </div>
            <div className={`flex items-center text-sm ${progress >= 80 ? 'text-green-600' : 'text-gray-400'}`}>
              <div className={`w-2 h-2 rounded-full mr-2 ${progress >= 80 ? 'bg-green-500' : 'bg-gray-300'}`}></div>
              분석 데이터 저장 중...
            </div>
            <div className={`flex items-center text-sm ${progress >= 100 ? 'text-green-600' : 'text-gray-400'}`}>
              <div className={`w-2 h-2 rounded-full mr-2 ${progress >= 100 ? 'bg-green-500' : 'bg-gray-300'}`}></div>
              완료!
            </div>
          </div>
        </div>
        
        <div className="text-center max-w-lg">
          <div className="bg-blue-100/50 rounded-lg p-4 mb-4">
            <p className="text-sm text-blue-700 font-medium">
              💡 분석이 완료되면 자동으로 채팅 화면으로 이동됩니다
            </p>
          </div>
          <p className="text-xs text-gray-500">
            영상 길이와 복잡도에 따라 분석 시간이 달라질 수 있습니다
          </p>
        </div>
      </div>
    );
  }

  if (status === 'failed') {
    return (
      <div className="flex flex-col items-center justify-center h-64 bg-gradient-to-br from-red-50 to-pink-50 rounded-lg border border-red-200">
        <XCircle className="w-12 h-12 text-red-500 mb-4" />
        <h3 className="text-xl font-semibold text-red-700 mb-2">분석 실패</h3>
        <div className="text-center mb-6 max-w-md">
          <p className="text-red-600 mb-2">
            {message || '영상 분석에 실패했습니다.'}
          </p>
          <p className="text-sm text-gray-600">
            가능한 원인:
            <br />• 파일 형식이 지원되지 않음 (MP4, AVI, MOV, MKV, WEBM만 지원)
            <br />• 파일 크기가 너무 큼 (최대 50MB)
            <br />• 파일이 손상되었거나 읽을 수 없음
            <br />• 서버 처리 중 오류 발생
          </p>
        </div>
        <div className="flex gap-3">
          <button
            onClick={onRetry}
            className="px-4 py-2 bg-yellow-500 text-white rounded-lg hover:bg-yellow-600 transition-colors flex items-center"
          >
            <RefreshCw className="w-4 h-4 mr-2" />
            다시 분석
          </button>
          <button
            onClick={onBackToList}
            className="px-4 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600 transition-colors"
          >
            목록으로
          </button>
        </div>
      </div>
    );
  }

  return null;
};

export default AnalysisStatusView;