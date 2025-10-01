import React, { useState, useEffect, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Send, Upload, Loader2, CheckCircle, XCircle, AlertCircle, Clock, FileVideo, RefreshCw } from 'lucide-react';
import { api } from '../utils/api';

const VideoChatPage = () => {
  const { videoId } = useParams();
  const navigate = useNavigate();
  // const user = useSelector(state => state.auth.user);
  
  // 상태 관리
  const [videoList, setVideoList] = useState([]);
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [analysisStatus, setAnalysisStatus] = useState('unknown');
  const [analysisProgress, setAnalysisProgress] = useState(0);
  const [analysisMessage, setAnalysisMessage] = useState('');
  const [showVideoList, setShowVideoList] = useState(!videoId);
  
  const messagesEndRef = useRef(null);
  const loadingText = isLoading ? "분석중…" : "";

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // 비디오 목록 로드
  const loadVideoList = async () => {
    try {
      const response = await api.get('/api/video/list/');
      setVideoList(response.data.videos || []);
    } catch (error) {
      console.error('비디오 목록 로드 실패:', error);
    }
  };

  // 비디오 데이터 로드
  const loadVideoData = async (id) => {
    try {
      const response = await api.get(`/api/video/${id}/analysis/`);
      setSelectedVideo(response.data);
      setAnalysisStatus(response.data.analysis_status);
      
      if (response.data.analysis_status === 'pending') {
        checkAnalysisStatus(id);
      } else if (response.data.analysis_status === 'completed') {
        loadChatHistory(id);
      }
    } catch (error) {
      console.error('비디오 데이터 로드 실패:', error);
    }
  };

  // 분석 상태 확인
  const checkAnalysisStatus = async (id) => {
    const interval = setInterval(async () => {
      try {
        const response = await api.get(`/api/video/${id}/analysis/`);
        setAnalysisStatus(response.data.analysis_status);
        
        // 진행률 정보 업데이트
        if (response.data.progress) {
          setAnalysisProgress(response.data.progress.analysis_progress || 0);
          setAnalysisMessage(response.data.progress.analysis_message || '');
        }
        
        if (response.data.analysis_status === 'completed') {
          clearInterval(interval);
          setAnalysisProgress(100);
          setAnalysisMessage('분석 완료');
          loadChatHistory(id);
          loadVideoList(); // 목록 새로고침
        } else if (response.data.analysis_status === 'failed') {
          clearInterval(interval);
          setAnalysisProgress(0);
          setAnalysisMessage('분석 실패');
        }
      } catch (error) {
        console.error('분석 상태 확인 실패:', error);
        clearInterval(interval);
      }
    }, 1000); // 1초마다 확인 (더 빠른 업데이트)
  };

  // 분석 시작하기
  const startAnalysis = async (videoId) => {
    try {
      setIsLoading(true);
      
      // 진행률 초기화
      setAnalysisProgress(0);
      setAnalysisMessage('분석을 시작합니다...');
      
      // 백엔드에 분석 시작 요청
      const response = await api.post(`/api/video/${videoId}/analysis/`);
      
      if (response.data.status === 'pending') {
        // 분석 상태를 pending으로 변경하고 폴링 시작
        setAnalysisStatus('pending');
        checkAnalysisStatus(videoId);
        loadVideoList(); // 목록 새로고침
        
        // 영상 목록으로 이동
        setShowVideoList(true);
        setSelectedVideo(null);
        navigate('/video-chat');
      }
    } catch (error) {
      console.error('분석 시작 실패:', error);
      alert('분석 시작에 실패했습니다.');
    } finally {
      setIsLoading(false);
    }
  };

  // 채팅 히스토리 로드
  const loadChatHistory = async (id) => {
    try {
      const response = await api.get(`/api/video/${id}/chat/`);
      setMessages(response.data.messages || []);
    } catch (error) {
      console.error('채팅 히스토리 로드 실패:', error);
    }
  };

  // 파일 업로드
  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setUploading(true);
    setUploadProgress(0);

    const formData = new FormData();
    formData.append('video', file);
    formData.append('title', file.name);

    try {
      const response = await api.post('/api/video/upload/', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          setUploadProgress(progress);
        }
      });

      if (response.data.video_id) {
        setShowVideoList(false);
        navigate(`/video-chat/${response.data.video_id}`);
        loadVideoData(response.data.video_id);
        loadVideoList(); // 목록 새로고침
      }
    } catch (error) {
      console.error('영상 업로드 실패:', error);
      alert('영상 업로드에 실패했습니다.');
    } finally {
      setUploading(false);
      setUploadProgress(0);
    }
  };

  // 메시지 전송
  const handleSendMessage = async () => {
    if (!inputMessage.trim() || !selectedVideo) return;

    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: inputMessage,
      timestamp: new Date().toISOString()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      const response = await api.post(`/api/video/${selectedVideo.video_id}/chat/`, {
        message: inputMessage
      });

      if (response.data.ai_responses) {
        const aiMessages = [];
        
        // 개별 AI 응답들
        if (response.data.ai_responses.individual) {
          response.data.ai_responses.individual.forEach(aiResponse => {
            aiMessages.push({
              id: aiResponse.id,
              type: 'ai',
              ai_model: aiResponse.model,
              content: aiResponse.content,
              created_at: aiResponse.created_at,
              relevant_frames: response.data.relevant_frames || []
            });
          });
        }
        
        // 통합 응답
        if (response.data.ai_responses.optimal) {
          aiMessages.push({
            id: `optimal_${Date.now()}`,
            type: 'ai_optimal',
            content: response.data.ai_responses.optimal.content,
            created_at: response.data.ai_responses.optimal.created_at,
            relevant_frames: response.data.relevant_frames || []
          });
        }
        
        setMessages(prev => [...prev, ...aiMessages]);
      }
    } catch (error) {
      console.error('메시지 전송 실패:', error);
      alert('메시지 전송에 실패했습니다.');
    } finally {
      setIsLoading(false);
    }
  };

  // 비디오 선택
  const selectVideo = (video) => {
    setSelectedVideo(video);
    setShowVideoList(false);
    navigate(`/video-chat/${video.id}`);
    loadVideoData(video.id);
  };

  // 비디오 목록으로 돌아가기
  const backToVideoList = () => {
    setShowVideoList(true);
    setSelectedVideo(null);
    setMessages([]);
    navigate('/video-chat');
  };

  // 컴포넌트 마운트 시 초기화
  useEffect(() => {
    loadVideoList();
    if (videoId) {
      loadVideoData(videoId);
    }
  }, [videoId]);

  // 분석 상태에 따른 UI 렌더링
  const renderAnalysisStatus = () => {
    switch (analysisStatus) {
      case 'pending':
        return (
          <div className="flex flex-col items-center justify-center h-64 bg-gray-50 rounded-lg">
            <Loader2 className="w-12 h-12 text-blue-500 animate-spin mb-4" />
            <h3 className="text-xl font-semibold text-gray-700 mb-2">영상 분석 중</h3>
            <p className="text-gray-500 text-center mb-4">
              {analysisMessage || '영상을 분석하고 있습니다...'}
            </p>
            <div className="w-full max-w-sm mb-2">
              <div className="flex justify-between text-sm text-gray-600 mb-1">
                <span>진행률</span>
                <span>{analysisProgress}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-3">
                <div 
                  className="bg-blue-500 h-3 rounded-full transition-all duration-300 ease-out" 
                  style={{ width: `${analysisProgress}%` }}
                ></div>
              </div>
            </div>
            <p className="text-sm text-gray-400">분석이 완료되면 자동으로 채팅이 시작됩니다</p>
          </div>
        );
      case 'failed':
        return (
          <div className="flex flex-col items-center justify-center h-64 bg-red-50 rounded-lg">
            <XCircle className="w-12 h-12 text-red-500 mb-4" />
            <h3 className="text-xl font-semibold text-red-700 mb-2">분석 실패</h3>
            <p className="text-red-500 text-center mb-4">
              영상 분석에 실패했습니다.<br />
              다른 영상을 업로드해주세요.
            </p>
            <button
              onClick={backToVideoList}
              className="px-4 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600 transition-colors"
            >
              새 영상 업로드
            </button>
          </div>
        );
      case 'completed':
        return null; // 채팅 인터페이스 표시
      default:
        return null; // 기본적으로는 아무것도 표시하지 않음
    }
  };

  // 비디오 목록 UI
  const renderVideoList = () => (
    <div className="max-w-4xl mx-auto p-6">
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-gray-800 mb-2">영상 채팅</h1>
        <p className="text-gray-600">영상을 업로드하고 AI와 채팅해보세요</p>
      </div>

      {/* 업로드 영역 */}
      <div className="mb-8">
        <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-blue-400 transition-colors">
          <input
            type="file"
            accept="video/*"
            onChange={handleFileUpload}
            className="hidden"
            id="video-upload"
            disabled={uploading}
          />
          <label
            htmlFor="video-upload"
            className="cursor-pointer flex flex-col items-center"
          >
            {uploading ? (
              <>
                <Loader2 className="w-12 h-12 text-blue-500 animate-spin mb-4" />
                <h3 className="text-lg font-semibold text-gray-700 mb-2">업로드 중...</h3>
                <div className="w-full max-w-xs bg-gray-200 rounded-full h-2 mb-2">
                  <div
                    className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${uploadProgress}%` }}
                  ></div>
                </div>
                <p className="text-gray-500">{uploadProgress}% 완료</p>
              </>
            ) : (
              <>
                <Upload className="w-12 h-12 text-gray-400 mb-4" />
                <h3 className="text-lg font-semibold text-gray-700 mb-2">영상 업로드</h3>
                <p className="text-gray-500 mb-4">클릭하여 영상 파일을 선택하세요</p>
                <p className="text-sm text-gray-400">MP4, AVI, MOV, MKV, WEBM 지원</p>
              </>
            )}
          </label>
        </div>
      </div>

      {/* 비디오 목록 */}
      {videoList.length > 0 && (
        <div>
          <h2 className="text-xl font-semibold text-gray-800 mb-4">업로드된 영상</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {videoList.map((video) => (
              <div
                key={video.id}
                onClick={() => selectVideo(video)}
                className="bg-white rounded-lg shadow-md p-4 hover:shadow-lg transition-shadow cursor-pointer border"
              >
                <div className="flex items-center mb-3">
                  <FileVideo className="w-8 h-8 text-blue-500 mr-3" />
                  <div className="flex-1 min-w-0">
                    <h3 className="font-medium text-gray-800 truncate">
                      {video.original_name}
                    </h3>
                    <p className="text-sm text-gray-500">
                      {new Date(video.uploaded_at).toLocaleDateString()}
                    </p>
                  </div>
                </div>
                
                <div className="flex items-center justify-between text-sm mb-3">
                  <span className="text-gray-500">
                    {(video.file_size / (1024 * 1024)).toFixed(1)}MB
                  </span>
                  <div className="flex items-center">
                    {video.analysis_status === 'completed' && (
                      <CheckCircle className="w-4 h-4 text-green-500 mr-1" />
                    )}
                    {video.analysis_status === 'pending' && (
                      <Clock className="w-4 h-4 text-yellow-500 mr-1" />
                    )}
                    {video.analysis_status === 'failed' && (
                      <XCircle className="w-4 h-4 text-red-500 mr-1" />
                    )}
                    <span className={`text-xs font-medium ${
                      video.analysis_status === 'completed' ? 'text-green-600' :
                      video.analysis_status === 'pending' ? 'text-yellow-600' :
                      'text-red-600'
                    }`}>
                      {video.analysis_status === 'completed' ? '분석 완료' :
                       video.analysis_status === 'pending' ? '분석 중' :
                       '분석 실패'}
                    </span>
                  </div>
                </div>
                
                {/* 액션 버튼들 */}
                <div className="flex gap-2">
                  {video.analysis_status === 'completed' && (
                    <button
                      onClick={() => selectVideo(video)}
                      className="flex-1 px-3 py-2 bg-blue-500 text-white text-sm rounded-lg hover:bg-blue-600 transition-colors"
                    >
                      채팅하기
                    </button>
                  )}
                  {video.analysis_status === 'pending' && (
                    <button
                      disabled
                      className="flex-1 px-3 py-2 bg-gray-300 text-gray-500 text-sm rounded-lg cursor-not-allowed"
                    >
                      분석 중...
                    </button>
                  )}
                  {video.analysis_status === 'failed' && (
                    <button
                      onClick={() => startAnalysis(video.id)}
                      className="flex-1 px-3 py-2 bg-yellow-500 text-white text-sm rounded-lg hover:bg-yellow-600 transition-colors"
                    >
                      다시 분석
                    </button>
                  )}
                  {(!video.analysis_status || video.analysis_status === 'uploaded') && (
                    <button
                      onClick={() => startAnalysis(video.id)}
                      className="flex-1 px-3 py-2 bg-green-500 text-white text-sm rounded-lg hover:bg-green-600 transition-colors"
                    >
                      분석 시작
                    </button>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );

  // Optimal Response Renderer Component (기본 채팅과 동일)
  const OptimalResponseRenderer = ({ content, relevantFrames }) => {
    const parseOptimalResponse = (text) => {
      if (!text || typeof text !== 'string') {
        return {};
      }
      
      const sections = {};
      const lines = text.split('\n');
      let currentSection = '';
      let currentContent = [];
      
      for (const line of lines) {
        if (line.startsWith('## 통합 답변') || line.startsWith('## 🎯 통합 답변')) {
          if (currentSection) sections[currentSection] = currentContent.join('\n').trim();
          currentSection = 'integrated';
          currentContent = [];
        } else if (line.startsWith('## 각 AI 분석') || line.startsWith('## 📊 각 AI 분석')) {
          if (currentSection) sections[currentSection] = currentContent.join('\n').trim();
          currentSection = 'analysis';
          currentContent = [];
        } else if (line.startsWith('## 분석 근거') || line.startsWith('## 🔍 분석 근거')) {
          if (currentSection) sections[currentSection] = currentContent.join('\n').trim();
          currentSection = 'rationale';
          currentContent = [];
        } else if (line.startsWith('## 최종 추천') || line.startsWith('## 🏆 최종 추천')) {
          if (currentSection) sections[currentSection] = currentContent.join('\n').trim();
          currentSection = 'recommendation';
          currentContent = [];
        } else if (line.startsWith('## 추가 인사이트') || line.startsWith('## 💡 추가 인사이트')) {
          if (currentSection) sections[currentSection] = currentContent.join('\n').trim();
          currentSection = 'insights';
          currentContent = [];
        } else if (line.trim() !== '') {
          currentContent.push(line);
        }
      }
      
      if (currentSection) sections[currentSection] = currentContent.join('\n').trim();
      return sections;
    };

    const parseAIAnalysis = (analysisText) => {
      const analyses = {};
      const lines = analysisText.split('\n');
      let currentAI = '';
      let currentAnalysis = { pros: [], cons: [] };
      
      for (const line of lines) {
        if (line.startsWith('### ')) {
          if (currentAI) {
            analyses[currentAI] = currentAnalysis;
          }
          currentAI = line.replace('### ', '').trim();
          currentAnalysis = { pros: [], cons: [] };
        } else if (line.includes('- 장점:')) {
          currentAnalysis.pros.push(line.replace('- 장점:', '').trim());
        } else if (line.includes('- 단점:')) {
          currentAnalysis.cons.push(line.replace('- 단점:', '').trim());
        }
      }
      
      if (currentAI) {
        analyses[currentAI] = currentAnalysis;
      }
      
      return analyses;
    };

    if (!content || typeof content !== 'string') {
      return (
        <div className="optimal-response-container">
          <div className="optimal-section integrated-answer">
            <h3 className="section-title">최적 답변</h3>
            <div className="section-content">최적의 답변을 생성 중입니다...</div>
          </div>
        </div>
      );
    }

    const sections = parseOptimalResponse(content);
    const analysisData = sections.analysis ? parseAIAnalysis(sections.analysis) : {};

    return (
      <div className="optimal-response-container">
        {sections.integrated && (
          <div className="optimal-section integrated-answer">
            <h3 className="section-title">최적 답변</h3>
            <div className="section-content">{sections.integrated}</div>
          </div>
        )}
        
        {sections.analysis && (
          <div className="optimal-section analysis-section">
            <h3 className="section-title">각 AI 분석</h3>
            <div className="analysis-grid">
              {Object.entries(analysisData).map(([aiName, analysis]) => (
                <div key={aiName} className="analysis-item">
                  <h4 className="analysis-ai-name">{aiName}</h4>
                  {analysis.pros.length > 0 && (
                    <div className="analysis-pros">
                      <strong>장점:</strong>
                      <ul>
                        {analysis.pros.map((pro, index) => (
                          <li key={index}>{pro}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                  {analysis.cons.length > 0 && (
                    <div className="analysis-cons">
                      <strong>단점:</strong>
                      <ul>
                        {analysis.cons.map((con, index) => (
                          <li key={index}>{con}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
        
        {sections.rationale && (
          <div className="optimal-section rationale-section">
            <h3 className="section-title">분석 근거</h3>
            <div className="section-content">{sections.rationale}</div>
          </div>
        )}
        
        {sections.recommendation && (
          <div className="optimal-section recommendation-section">
            <h3 className="section-title">최종 추천</h3>
            <div className="section-content">{sections.recommendation}</div>
          </div>
        )}
        
        {sections.insights && (
          <div className="optimal-section insights-section">
            <h3 className="section-title">추가 인사이트</h3>
            <div className="section-content">{sections.insights}</div>
          </div>
        )}

        {/* 관련 프레임 이미지 표시 */}
        {relevantFrames && relevantFrames.length > 0 && (
          <div className="optimal-section frames-section">
            <h3 className="section-title">📸 관련 프레임</h3>
            <div className="frames-grid">
              {relevantFrames.map((frame, index) => (
                <div key={index} className="frame-card">
                  <div className="frame-info">
                    <span className="frame-timestamp">⏰ {frame.timestamp.toFixed(1)}초</span>
                    <span className="frame-score">🎯 {frame.relevance_score}점</span>
                  </div>
                  <img
                    src={`http://localhost:8000${frame.image_url}`}
                    alt={`프레임 ${frame.image_id}`}
                    className="frame-image"
                    onError={(e) => {
                      e.target.style.display = 'none';
                    }}
                  />
                  <div className="frame-tags">
                    {frame.persons && frame.persons.length > 0 && (
                      <span className="frame-tag person-tag">
                        👤 사람 {frame.persons.length}명
                      </span>
                    )}
                    {frame.objects && frame.objects.length > 0 && (
                      <span className="frame-tag object-tag">
                        📦 객체 {frame.objects.length}개
                      </span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    );
  };

  // 채팅 인터페이스 UI (기본 채팅과 동일한 구조)
  const renderChatInterface = () => (
    <div className="h-full w-full flex flex-col" style={{ background: "rgba(245, 242, 234, 0.4)" }}>
      <style jsx>{`
        .chat-header {
          background: rgba(245, 242, 234, 0.4);
          backdrop-filter: blur(10px);
          border-bottom: 1px solid rgba(139, 168, 138, 0.15);
          height: 60px;
        }
        .chat-column {
          background: rgba(255, 255, 255, 0.3);
          backdrop-filter: blur(5px);
        }
        .chat-container {
          height: calc(100% - 180px);
        }
        .aiofai-input-area {
          background: rgba(245, 242, 234, 0.4);
          backdrop-filter: blur(10px);
          border-top: 1px solid rgba(139, 168, 138, 0.15);
          padding: 0.75rem 1.2rem;
          display: flex;
          flex-direction: column;
          justify-content: center;
          gap: 0.3rem;
        }
        .aiofai-user-message {
          background: linear-gradient(135deg, #5d7c5b, #8ba88a);
          color: #ffffff;
          padding: 1.2rem 1.5rem;
          border-radius: 24px 24px 8px 24px;
          max-width: 85%;
          box-shadow: 0 8px 32px rgba(93, 124, 91, 0.3);
          font-weight: 500;
          line-height: 1.5;
          position: relative;
        }
        .aiofai-bot-message {
          background: rgba(255, 255, 255, 0.8);
          backdrop-filter: blur(10px);
          color: #2d3e2c;
          border: 1px solid rgba(139, 168, 138, 0.2);
          padding: 1.2rem 1.5rem;
          border-radius: 24px 24px 24px 8px;
          max-width: 85%;
          box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
          line-height: 1.6;
          position: relative;
        }
        .optimal-response {
          background: rgba(255, 255, 255, 0.95) !important;
          border: 2px solid rgba(139, 168, 138, 0.3) !important;
          padding: 1.5rem !important;
          border-radius: 16px !important;
          max-width: 95% !important;
          box-shadow: 0 8px 32px rgba(139, 168, 138, 0.15) !important;
        }
        .optimal-response-container {
          width: 100%;
        }
        .optimal-section {
          margin-bottom: 1.5rem;
          padding: 1rem;
          border-bottom: 1px solid #e5e7eb;
        }
        .optimal-section:last-child {
          margin-bottom: 0;
          border-bottom: none;
        }
        .section-title {
          margin: 0 0 1rem 0;
          font-size: 1rem;
          font-weight: 600;
          color: #374151;
          text-transform: uppercase;
          letter-spacing: 0.05em;
        }
        .section-content {
          color: #374151;
          line-height: 1.6;
          font-size: 0.95rem;
        }
        .analysis-grid {
          display: flex;
          flex-direction: column;
          gap: 1rem;
        }
        .analysis-item {
          padding: 1rem;
          background: rgba(139, 168, 138, 0.05);
          border-radius: 8px;
          border: 1px solid rgba(139, 168, 138, 0.2);
        }
        .analysis-ai-name {
          font-weight: 600;
          color: #2d3e2c;
          margin-bottom: 0.5rem;
        }
        .analysis-pros, .analysis-cons {
          margin-bottom: 0.5rem;
        }
        .analysis-pros ul, .analysis-cons ul {
          margin: 0.25rem 0 0 1rem;
          padding: 0;
        }
        .analysis-pros li, .analysis-cons li {
          margin-bottom: 0.25rem;
        }
        .frames-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
          gap: 1rem;
        }
        .frame-card {
          border: 1px solid #e5e7eb;
          border-radius: 8px;
          padding: 0.75rem;
          background: rgba(255, 255, 255, 0.8);
        }
        .frame-info {
          display: flex;
          justify-content: space-between;
          margin-bottom: 0.5rem;
        }
        .frame-timestamp, .frame-score {
          font-size: 0.75rem;
          padding: 0.25rem 0.5rem;
          border-radius: 4px;
          background: #f3f4f6;
        }
        .frame-image {
          width: 100%;
          height: 120px;
          object-fit: cover;
          border-radius: 4px;
          margin-bottom: 0.5rem;
        }
        .frame-tags {
          display: flex;
          gap: 0.25rem;
          flex-wrap: wrap;
        }
        .frame-tag {
          font-size: 0.75rem;
          padding: 0.25rem 0.5rem;
          border-radius: 4px;
        }
        .person-tag {
          background: #dcfce7;
          color: #166534;
        }
        .object-tag {
          background: #dbeafe;
          color: #1e40af;
        }
      `}</style>

      {/* 헤더 */}
      <div className="chat-header flex items-center justify-between px-6">
        <div className="flex items-center">
          <button
            onClick={backToVideoList}
            className="mr-4 p-2 hover:bg-white/20 rounded-lg transition-colors"
          >
            <RefreshCw className="w-5 h-5 text-gray-600" />
          </button>
          <div>
            <h1 className="text-lg font-semibold text-gray-800">
              {selectedVideo?.original_name || '영상 채팅'}
            </h1>
            <p className="text-sm text-gray-600">
              {selectedVideo && `${(selectedVideo.file_size / (1024 * 1024)).toFixed(1)}MB`}
            </p>
          </div>
        </div>
        <div className="flex items-center text-sm text-green-600">
          <CheckCircle className="w-4 h-4 mr-1" />
          분석 완료
        </div>
      </div>

      {/* 메시지 영역 - 여러 컬럼 구조 (기본 채팅과 동일) */}
      <div className="chat-container flex overflow-hidden">
        {['gpt', 'claude', 'mixtral', 'optimal'].map((modelId) => (
          <div key={modelId} className="border-r flex-1 overflow-y-auto chat-column">
            <div className="h-full px-4 py-3">
              <div className="text-center text-sm font-medium text-gray-600 mb-4 pb-2 border-b">
                {modelId === 'optimal' ? '🤖 통합 응답' : `🤖 ${modelId.toUpperCase()}`}
              </div>
              
              {messages.map((message, index) => {
                const isUser = message.type === 'user';
                const isOptimal = modelId === 'optimal' && message.type === 'ai_optimal';
                const isModelMessage = modelId !== 'optimal' && message.ai_model === modelId;
                
                if (!isUser && !isOptimal && !isModelMessage) return null;
                
                return (
                  <div key={`${modelId}-${index}`} className={`flex ${isUser ? "justify-end" : "justify-start"} mb-4`}>
                    <div className={`${isUser ? "aiofai-user-message" : "aiofai-bot-message"} ${isOptimal ? "optimal-response" : ""}`}>
                      {isOptimal ? (
                        <OptimalResponseRenderer 
                          content={message.content} 
                          relevantFrames={message.relevant_frames}
                        />
                      ) : (
                        <div>
                          <div className="whitespace-pre-wrap">{message.content}</div>
                          
                          {/* 관련 프레임 이미지 표시 (개별 AI 응답에도 표시) */}
                          {message.relevant_frames && message.relevant_frames.length > 0 && (
                            <div className="mt-3 space-y-2">
                              <div className="text-xs font-medium text-gray-600">📸 관련 프레임:</div>
                              <div className="grid grid-cols-1 gap-2">
                                {message.relevant_frames.map((frame, frameIndex) => (
                                  <div key={frameIndex} className="border rounded-lg p-2 bg-gray-50">
                                    <div className="flex items-center space-x-2 mb-2">
                                      <span className="text-xs text-gray-500">
                                        ⏰ {frame.timestamp.toFixed(1)}초
                                      </span>
                                      <span className="text-xs text-blue-600">
                                        🎯 {frame.relevance_score}점
                                      </span>
                                    </div>
                                    <img
                                      src={`http://localhost:8000${frame.image_url}`}
                                      alt={`프레임 ${frame.image_id}`}
                                      className="w-full h-24 object-cover rounded"
                                      onError={(e) => {
                                        e.target.style.display = 'none';
                                      }}
                                    />
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}
                          
                          <div className="text-xs opacity-60 mt-2">
                            {new Date(message.created_at).toLocaleTimeString()}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}

              {isLoading && (
                <div className="flex justify-start mb-4">
                  <div className="bg-gray-100 text-gray-800 p-4 rounded-2xl">
                    {loadingText || "입력 중..."}
                  </div>
                </div>
              )}

              <div className="h-3" />
              <div ref={messagesEndRef} />
            </div>
          </div>
        ))}
      </div>

      {/* 입력 영역 */}
      <div className="aiofai-input-area">
        <div className="flex space-x-3">
          <input
            type="text"
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
            placeholder="영상에 대해 질문해보세요..."
            className="flex-1 px-4 py-3 bg-white/80 backdrop-blur-sm border border-gray-200 rounded-2xl focus:outline-none focus:ring-2 focus:ring-green-400 focus:border-transparent transition-all duration-200"
            disabled={isLoading}
          />
          <button
            onClick={handleSendMessage}
            disabled={!inputMessage.trim() || isLoading}
            className="px-6 py-3 bg-gradient-to-r from-green-500 to-green-600 text-white rounded-2xl hover:from-green-600 hover:to-green-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 shadow-lg hover:shadow-xl"
          >
            <Send className="w-5 h-5" />
          </button>
        </div>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gray-50">
      {showVideoList ? (
        renderVideoList()
      ) : analysisStatus === 'completed' ? (
        renderChatInterface()
      ) : (
        <div className="max-w-2xl mx-auto p-6">
          {renderAnalysisStatus()}
        </div>
      )}
    </div>
  );
};

export default VideoChatPage;