import React, { useState, useEffect, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Send, Loader2, CheckCircle, XCircle, RefreshCw, Clock } from 'lucide-react';
import { api } from '../utils/api';

const VideoChatDetailPage = () => {
  const { videoId } = useParams();
  const navigate = useNavigate();
  
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [analysisStatus, setAnalysisStatus] = useState('unknown');
  const [analysisProgress, setAnalysisProgress] = useState(0);
  const [analysisMessage, setAnalysisMessage] = useState('');
  
  const [selectedFrame, setSelectedFrame] = useState(null);
  const [isFrameModalOpen, setIsFrameModalOpen] = useState(false);
  const [showBboxOverlay, setShowBboxOverlay] = useState(true);
  const canvasRef = useRef(null);
  
  const scrollRefs = useRef({
    gpt: null,
    claude: null,
    mixtral: null,
    optimal: null
  });

  const loadingText = isLoading ? "분석중…" : "";

  useEffect(() => {
    if (videoId) {
      loadVideoData(videoId);
    }
  }, [videoId]);

  useEffect(() => {
    ['gpt', 'claude', 'mixtral', 'optimal'].forEach(modelId => {
      scrollToBottomForModel(modelId);
    });
  }, [messages]);

  useEffect(() => {
    if (showBboxOverlay && selectedFrame) {
      const img = document.getElementById('modal-frame-image');
      if (img && img.complete) {
        drawBboxOnCanvas(img, selectedFrame);
      }
    }
  }, [showBboxOverlay, selectedFrame]);

  const scrollToBottomForModel = (modelId) => {
    const scrollRef = scrollRefs.current[modelId];
    if (scrollRef) {
      scrollRef.scrollIntoView({ behavior: 'smooth' });
    }
  };

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

  const checkAnalysisStatus = async (id) => {
    const interval = setInterval(async () => {
      try {
        const response = await api.get(`/api/video/${id}/analysis/`);
        setAnalysisStatus(response.data.analysis_status);
        
        if (response.data.progress) {
          setAnalysisProgress(response.data.progress.analysis_progress || 0);
          setAnalysisMessage(response.data.progress.analysis_message || '');
        }
        
        if (response.data.analysis_status === 'completed') {
          clearInterval(interval);
          setAnalysisProgress(100);
          setAnalysisMessage('분석 완료');
          loadChatHistory(id);
        } else if (response.data.analysis_status === 'failed') {
          clearInterval(interval);
          setAnalysisProgress(0);
          setAnalysisMessage(response.data.progress?.analysis_message || '분석 실패');
        }
      } catch (error) {
        console.error('분석 상태 확인 실패:', error);
        clearInterval(interval);
        setAnalysisMessage('분석 상태 확인 실패');
      }
    }, 2000);
  };

  const startAnalysis = async (videoId) => {
    try {
      setIsLoading(true);
      setAnalysisProgress(0);
      setAnalysisMessage('분석을 시작합니다...');
      
      const response = await api.post(`/api/video/${videoId}/analysis/`);
      
      if (response.data.status === 'pending') {
        setAnalysisStatus('pending');
        checkAnalysisStatus(videoId);
      }
    } catch (error) {
      console.error('분석 시작 실패:', error);
      alert('분석 시작에 실패했습니다.');
    } finally {
      setIsLoading(false);
    }
  };

  const loadChatHistory = async (id) => {
    try {
      const response = await api.get(`/api/video/${id}/chat/`);
      setMessages(response.data.messages || []);
    } catch (error) {
      console.error('채팅 히스토리 로드 실패:', error);
    }
  };

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

  const backToVideoList = () => {
    navigate('/video-chat');
  };

  const drawBboxOnCanvas = (imageElement, frame) => {
    if (!canvasRef.current || !imageElement) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    canvas.width = imageElement.naturalWidth;
    canvas.height = imageElement.naturalHeight;
    
    ctx.drawImage(imageElement, 0, 0);
    
    if (frame.persons && frame.persons.length > 0) {
      frame.persons.forEach((person, index) => {
        const bbox = person.bbox || [];
        if (bbox.length === 4) {
          const [x1, y1, x2, y2] = bbox;
          const x = x1 * canvas.width;
          const y = y1 * canvas.height;
          const width = (x2 - x1) * canvas.width;
          const height = (y2 - y1) * canvas.height;
          
          ctx.strokeStyle = '#8B4513';
          ctx.lineWidth = 3;
          ctx.strokeRect(x, y, width, height);
          
          const label = `사람 ${index + 1} (${(person.confidence * 100).toFixed(1)}%)`;
          ctx.font = '16px Arial';
          const textWidth = ctx.measureText(label).width;
          ctx.fillStyle = '#8B4513';
          ctx.fillRect(x, y - 25, textWidth + 10, 25);
          
          ctx.fillStyle = '#FFFFFF';
          ctx.fillText(label, x + 5, y - 7);
        }
      });
    }
    
    if (frame.objects && frame.objects.length > 0) {
      frame.objects.forEach((obj) => {
        const bbox = obj.bbox || [];
        if (bbox.length === 4) {
          const [x1, y1, x2, y2] = bbox;
          const x = x1 * canvas.width;
          const y = y1 * canvas.height;
          const width = (x2 - x1) * canvas.width;
          const height = (y2 - y1) * canvas.height;
          
          ctx.strokeStyle = '#FF8C00';
          ctx.lineWidth = 3;
          ctx.strokeRect(x, y, width, height);
          
          const label = `${obj.class} (${(obj.confidence * 100).toFixed(1)}%)`;
          ctx.font = '16px Arial';
          const textWidth = ctx.measureText(label).width;
          ctx.fillStyle = '#FF8C00';
          ctx.fillRect(x, y - 25, textWidth + 10, 25);
          
          ctx.fillStyle = '#FFFFFF';
          ctx.fillText(label, x + 5, y - 7);
        }
      });
    }
  };

  const OptimalResponseRenderer = ({ content, relevantFrames }) => {
    const parseOptimalResponse = (text) => {
      if (!text || typeof text !== 'string') return {};
      
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
        } else if (line.startsWith('## 분석 근거') || line.startsWith('## 📝 분석 근거')) {
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
          if (currentAI) analyses[currentAI] = currentAnalysis;
          currentAI = line.replace('### ', '').trim();
          currentAnalysis = { pros: [], cons: [] };
        } else if (line.includes('- 장점:')) {
          currentAnalysis.pros.push(line.replace('- 장점:', '').trim());
        } else if (line.includes('- 단점:')) {
          currentAnalysis.cons.push(line.replace('- 단점:', '').trim());
        }
      }
      
      if (currentAI) analyses[currentAI] = currentAnalysis;
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

        {relevantFrames && relevantFrames.length > 0 && (
          <div className="optimal-section frames-section">
            <h3 className="section-title">관련 프레임</h3>
            <div className="frames-grid">
              {relevantFrames.map((frame, index) => (
                <div key={index} className="frame-card">
                  <div className="frame-info">
                    <span className="frame-timestamp">{frame.timestamp.toFixed(1)}초</span>
                    <span className="frame-score">{frame.relevance_score}점</span>
                  </div>
                  <img
                    src={`${api.defaults.baseURL}${frame.image_url}`}
                    alt={`프레임 ${frame.image_id}`}
                    className="frame-image"
                    onError={(e) => {
                      console.error(`프레임 이미지 로드 실패: ${frame.image_url}`);
                      e.target.style.display = 'none';
                    }}
                  />
                  <div className="frame-tags">
                    {frame.persons && frame.persons.length > 0 && (
                      <span className="frame-tag person-tag">
                        사람 {frame.persons.length}명
                      </span>
                    )}
                    {frame.objects && frame.objects.length > 0 && (
                      <span className="frame-tag object-tag">
                        객체 {frame.objects.length}개
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

  const renderAnalysisStatus = () => {
    switch (analysisStatus) {
      case 'pending':
        return (
          <div className="flex flex-col items-center justify-center h-64 bg-gradient-to-br from-blue-50 to-indigo-50 rounded-lg border border-blue-200">
            <Loader2 className="w-12 h-12 text-blue-500 animate-spin mb-4" />
            <h3 className="text-xl font-semibold text-gray-700 mb-2">영상 분석 중</h3>
            <p className="text-gray-600 text-center mb-4 max-w-md">
              {analysisMessage || '영상을 분석하고 있습니다...'}
            </p>
            <div className="w-full max-w-sm mb-4">
              <div className="flex justify-between text-sm text-gray-600 mb-2">
                <span className="font-medium">진행률</span>
                <span className="font-semibold text-blue-600">{analysisProgress}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-3 shadow-inner">
                <div 
                  className="bg-gradient-to-r from-blue-500 to-blue-600 h-3 rounded-full transition-all duration-500 ease-out shadow-sm" 
                  style={{ width: `${analysisProgress}%` }}
                ></div>
              </div>
            </div>
            <div className="flex items-center text-sm text-gray-500">
              <Clock className="w-4 h-4 mr-1" />
              <span>분석이 완료되면 자동으로 채팅이 시작됩니다</span>
            </div>
          </div>
        );
      case 'failed':
        return (
          <div className="flex flex-col items-center justify-center h-64 bg-gradient-to-br from-red-50 to-pink-50 rounded-lg border border-red-200">
            <XCircle className="w-12 h-12 text-red-500 mb-4" />
            <h3 className="text-xl font-semibold text-red-700 mb-2">분석 실패</h3>
            <div className="text-center mb-6 max-w-md">
              <p className="text-red-600 mb-2">
                {analysisMessage || '영상 분석에 실패했습니다.'}
              </p>
              <p className="text-sm text-gray-600">
                가능한 원인: 파일 형식 미지원, 파일 크기 초과, 파일 손상, 서버 오류
              </p>
            </div>
            <div className="flex gap-3">
              <button
                onClick={() => startAnalysis(selectedVideo?.id)}
                className="px-4 py-2 bg-yellow-500 text-white rounded-lg hover:bg-yellow-600 transition-colors flex items-center"
              >
                <RefreshCw className="w-4 h-4 mr-2" />
                다시 분석
              </button>
              <button
                onClick={backToVideoList}
                className="px-4 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600 transition-colors"
              >
                영상 목록으로
              </button>
            </div>
          </div>
        );
      default:
        return null;
    }
  };

  const renderChatInterface = () => (
    <div className="min-h-screen bg-gray-50">
      <style>{`
        .chat-header {
          background: rgba(245, 242, 234, 0.4);
          backdrop-filter: blur(10px);
          border-bottom: 1px solid rgba(139, 168, 138, 0.15);
          height: 60px;
        }
        .chat-column {
          background: rgba(255, 255, 255, 0.3);
          backdrop-filter: blur(5px);
          min-height: 0;
        }
        .chat-container {
          height: calc(100vh - 200px);
          min-height: 0;
        }
        .chat-column .overflow-y-auto::-webkit-scrollbar {
          width: 6px;
        }
        .chat-column .overflow-y-auto::-webkit-scrollbar-thumb {
          background: rgba(139, 168, 138, 0.3);
          border-radius: 3px;
        }
        .aiofai-input-area {
          background: rgba(245, 242, 234, 0.4);
          backdrop-filter: blur(10px);
          border-top: 1px solid rgba(139, 168, 138, 0.15);
          padding: 0.75rem 1.2rem;
          position: sticky;
          bottom: 0;
          z-index: 20;
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
        }
        .optimal-response {
          background: rgba(255, 255, 255, 0.95) !important;
          border: 2px solid rgba(139, 168, 138, 0.3) !important;
          padding: 1.5rem !important;
          border-radius: 16px !important;
          max-width: 95% !important;
          box-shadow: 0 8px 32px rgba(139, 168, 138, 0.15) !important;
        }
        .optimal-section {
          margin-bottom: 1.5rem;
          padding: 1rem;
          border-bottom: 1px solid #e5e7eb;
        }
        .section-title {
          margin: 0 0 1rem 0;
          font-size: 1rem;
          font-weight: 600;
          color: #374151;
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

      <div className="chat-container flex overflow-hidden h-full">
        {['gpt', 'claude', 'mixtral', 'optimal'].map((modelId) => (
          <div key={modelId} className="border-r flex-1 flex flex-col chat-column">
            <div className="flex-shrink-0 px-4 py-3 border-b bg-gray-50 flex items-center justify-between">
              <div className="text-center text-sm font-medium text-gray-600 flex-1">
                {modelId === 'optimal' ? '통합 응답' : modelId.toUpperCase()}
              </div>
              <button
                onClick={() => scrollToBottomForModel(modelId)}
                className="ml-2 p-1 text-gray-400 hover:text-gray-600 transition-colors"
                title="맨 아래로 스크롤"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
                </svg>
              </button>
            </div>
            
            <div className="flex-1 overflow-y-auto px-4 py-3" style={{ height: 'calc(100vh - 260px)' }}>
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
                          
                          {message.relevant_frames && message.relevant_frames.length > 0 && (
                            <div className="mt-4">
                              <div className="text-sm font-semibold text-gray-700 mb-3">
                                관련 프레임 ({message.relevant_frames.length}개)
                              </div>
                              <div className="grid grid-cols-1 gap-3">
                                {message.relevant_frames.map((frame, frameIndex) => (
                                  <div 
                                    key={frameIndex} 
                                    className="group relative bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden hover:shadow-lg hover:border-blue-300 transition-all duration-300 cursor-pointer"
                                    onClick={() => {
                                      setSelectedFrame(frame);
                                      setIsFrameModalOpen(true);
                                    }}
                                  >
                                    <div className="relative">
                                      <img
                                        src={`${api.defaults.baseURL}${frame.image_url}`}
                                        alt={`프레임 ${frame.image_id}`}
                                        className="w-full h-32 object-cover group-hover:scale-105 transition-transform duration-300"
                                        onError={(e) => {
                                          console.error(`프레임 이미지 로드 실패: ${frame.image_url}`);
                                          e.target.style.display = 'none';
                                        }}
                                      />
                                      <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-20 transition-all duration-300 flex items-center justify-center">
                                        <div className="opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                                          <div className="bg-white rounded-full p-2 shadow-lg">
                                            <svg className="w-6 h-6 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM10 7v3m0 0v3m0-3h3m-3 0H7" />
                                            </svg>
                                          </div>
                                        </div>
                                      </div>
                                    </div>
                                    
                                    <div className="p-3">
                                      <div className="flex items-center justify-between mb-2">
                                        <div className="flex items-center space-x-2">
                                          <div className="flex items-center bg-blue-50 text-blue-700 px-2 py-1 rounded-full text-xs font-medium">
                                            {frame.timestamp.toFixed(1)}초
                                          </div>
                                          <div className="flex items-center bg-green-50 text-green-700 px-2 py-1 rounded-full text-xs font-medium">
                                            {frame.relevance_score}점
                                          </div>
                                        </div>
                                        <div className="text-xs text-gray-500">
                                          #{frame.image_id}
                                        </div>
                                      </div>
                                      
                                      {frame.persons && frame.persons.length > 0 && (
                                        <div className="flex items-center space-x-2">
                                          <div className="flex items-center bg-purple-50 text-purple-700 px-2 py-1 rounded-full text-xs font-medium">
                                            사람 {frame.persons.length}명
                                          </div>
                                          {frame.objects && frame.objects.length > 0 && (
                                            <div className="flex items-center bg-orange-50 text-orange-700 px-2 py-1 rounded-full text-xs font-medium">
                                              객체 {frame.objects.length}개
                                            </div>
                                          )}
                                        </div>
                                      )}
                                    </div>
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
              <div ref={(el) => { scrollRefs.current[modelId] = el; }} />
            </div>
          </div>
        ))}
      </div>

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
      {analysisStatus === 'completed' ? (
        renderChatInterface()
      ) : (
        <div className="max-w-2xl mx-auto p-6">
          {renderAnalysisStatus()}
        </div>
      )}
      
      {isFrameModalOpen && selectedFrame && (
        <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-2xl max-w-4xl max-h-[90vh] overflow-auto shadow-2xl">
            <div className="flex items-center justify-between p-4 border-b bg-gray-50 sticky top-0 z-10">
              <div className="flex items-center space-x-3">
                <div className="flex items-center bg-blue-50 text-blue-700 px-3 py-1 rounded-full text-sm font-medium">
                  {selectedFrame.timestamp.toFixed(1)}초
                </div>
                <div className="flex items-center bg-green-50 text-green-700 px-3 py-1 rounded-full text-sm font-medium">
                  {selectedFrame.relevance_score}점
                </div>
                <div className="text-sm text-gray-600">
                  프레임 #{selectedFrame.image_id}
                </div>
              </div>
              <button
                onClick={() => setIsFrameModalOpen(false)}
                className="p-2 hover:bg-gray-200 rounded-full transition-colors"
              >
                <svg className="w-6 h-6 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            
            <div className="p-6">
              <div className="flex flex-col lg:flex-row gap-6">
                <div className="flex-1">
                  <div className="relative">
                    <div className="absolute top-2 right-2 z-10">
                      <button
                        onClick={() => setShowBboxOverlay(!showBboxOverlay)}
                        className={`px-3 py-1 rounded-full text-sm font-medium transition-colors ${
                          showBboxOverlay
                            ? 'bg-blue-500 text-white'
                            : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                        }`}
                      >
                        {showBboxOverlay ? 'bbox ON' : 'bbox OFF'}
                      </button>
                    </div>
                    
                    {showBboxOverlay ? (
                      <canvas
                        ref={canvasRef}
                        className="w-full h-auto max-h-[60vh] object-contain rounded-lg shadow-lg"
                      />
                    ) : (
                      <img
                        src={`${api.defaults.baseURL}${selectedFrame.image_url}`}
                        alt={`프레임 ${selectedFrame.image_id}`}
                        className="w-full h-auto max-h-[60vh] object-contain rounded-lg shadow-lg"
                        onError={(e) => {
                          console.error(`프레임 이미지 로드 실패: ${selectedFrame.image_url}`);
                          e.target.style.display = 'none';
                        }}
                      />
                    )}
                    
                    <img
                      id="modal-frame-image"
                      src={`${api.defaults.baseURL}${selectedFrame.image_url}`}
                      alt={`프레임 ${selectedFrame.image_id}`}
                      style={{ display: 'none' }}
                      onLoad={(e) => {
                        if (showBboxOverlay) {
                          drawBboxOnCanvas(e.target, selectedFrame);
                        }
                      }}
                      onError={(e) => {
                        console.error(`프레임 이미지 로드 실패: ${selectedFrame.image_url}`);
                      }}
                    />
                  </div>
                </div>
                
                <div className="lg:w-80 space-y-4">
                  <div>
                    <h3 className="text-lg font-semibold text-gray-800 mb-3">프레임 정보</h3>
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <span className="text-gray-600">타임스탬프:</span>
                        <span className="font-medium">{selectedFrame.timestamp.toFixed(1)}초</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">관련도 점수:</span>
                        <span className="font-medium text-green-600">{selectedFrame.relevance_score}점</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">프레임 ID:</span>
                        <span className="font-medium">#{selectedFrame.image_id}</span>
                      </div>
                    </div>
                  </div>
                  
                  {selectedFrame.persons && selectedFrame.persons.length > 0 && (
                    <div>
                      <h4 className="text-md font-semibold text-gray-800 mb-2">
                        감지된 사람 ({selectedFrame.persons.length}명)
                      </h4>
                      <div className="space-y-2">
                        {selectedFrame.persons.map((person, index) => (
                          <div key={index} className="bg-purple-50 rounded-lg p-3">
                            <div className="flex justify-between items-center mb-2">
                              <span className="text-sm font-medium text-purple-800">사람 {index + 1}</span>
                              <span className="text-xs bg-purple-200 text-purple-800 px-2 py-1 rounded-full">
                                신뢰도 {(person.confidence * 100).toFixed(1)}%
                              </span>
                            </div>
                            {person.attributes && (
                              <div className="text-xs text-gray-600 space-y-1">
                                {person.attributes.gender && (
                                  <div>성별: {person.attributes.gender.value}</div>
                                )}
                                {person.attributes.age && (
                                  <div>나이: {person.attributes.age.value}</div>
                                )}
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                  
                  {selectedFrame.objects && selectedFrame.objects.length > 0 && (
                    <div>
                      <h4 className="text-md font-semibold text-gray-800 mb-2">
                        감지된 객체 ({selectedFrame.objects.length}개)
                      </h4>
                      <div className="space-y-2">
                        {selectedFrame.objects.map((obj, index) => (
                          <div key={index} className="bg-orange-50 rounded-lg p-3">
                            <div className="flex justify-between items-center">
                              <span className="text-sm font-medium text-orange-800">{obj.class}</span>
                              <span className="text-xs bg-orange-200 text-orange-800 px-2 py-1 rounded-full">
                                신뢰도 {(obj.confidence * 100).toFixed(1)}%
                              </span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default VideoChatDetailPage;