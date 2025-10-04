import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Upload, Loader2, CheckCircle, XCircle, Clock, FileVideo } from 'lucide-react';
import { api } from '../utils/api';

const BRAND = '#5d7c5b';
const BRAND_BASE = '#8ba88a'; 
const BRAND_HOVER = '#5d7c5b'; 
const BRAND_TINT_BG = 'rgba(139, 168, 138, 0.05)';  
const BRAND_TINT_BORDER = 'rgba(139, 168, 138, 0.4)';
const NEUTRAL_BORDER = '#e5e7eb';

const VideoListPage = () => {
  const navigate = useNavigate();
  
  const [videoList, setVideoList] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isDragOver, setIsDragOver] = useState(false);
  const [isHover, setIsHover] = useState(false); 
  const [restartingId, setRestartingId] = useState(null);

  useEffect(() => { loadVideoList(); }, []);

  const loadVideoList = async () => {
    try {
      const response = await api.get('/api/video/list/');
      setVideoList(response.data.videos || []);
    } catch (error) {
      console.error('비디오 목록 로드 실패:', error);
    }
  };

  const validateFile = (file) => {
    if (file.size > 50 * 1024 * 1024) {
      alert(`파일 크기가 너무 큽니다. 최대 50MB까지 업로드 가능합니다. (현재: ${(file.size / (1024*1024)).toFixed(1)}MB)`);
      return false;
    }
    if (file.name.length > 200) {
      alert('파일명이 너무 깁니다. 200자 이하로 제한됩니다.');
      return false;
    }
    const allowedTypes = ['video/mp4', 'video/avi', 'video/mov', 'video/mkv', 'video/webm'];
    if (!allowedTypes.includes(file.type)) {
      alert('지원하지 않는 파일 형식입니다. MP4, AVI, MOV, MKV, WEBM 파일을 업로드해주세요.');
      return false;
    }
    return true;
  };

  const processFileUpload = async (file) => {
    if (!validateFile(file)) return;

    setUploading(true);
    setUploadProgress(0);

    const formData = new FormData();
    formData.append('video', file);
    formData.append('title', file.name);

    try {
      const response = await api.post('/api/video/upload/', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        onUploadProgress: (e) => {
          const progress = Math.round((e.loaded * 100) / e.total);
          setUploadProgress(progress);
        }
      });

      if (response.data.video_id) {
        alert('영상이 성공적으로 업로드되었습니다! 분석이 시작됩니다.');
        loadVideoList();
      }
    } catch (error) {
      console.error('영상 업로드 실패:', error);
      let errorMessage = '영상 업로드에 실패했습니다.';
      if (error.response?.data?.error) errorMessage = error.response.data.error;
      else if (error.message) errorMessage = `업로드 오류: ${error.message}`;
      alert(errorMessage);
    } finally {
      setUploading(false);
      setUploadProgress(0);
    }
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    await processFileUpload(file);
  };

  const handleDragOver = (e) => { e.preventDefault(); e.stopPropagation(); setIsDragOver(true); };
  const handleDragLeave = (e) => { e.preventDefault(); e.stopPropagation(); setIsDragOver(false); };
  const handleDrop = async (e) => {
    e.preventDefault(); e.stopPropagation(); setIsDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file) await processFileUpload(file);
  };

  const startAnalysis = async (videoId) => {
    setRestartingId(videoId);
    try {
      const tryForce = async () =>
        api.post(`/api/video/${videoId}/analysis/`, { force: true }).then(r => r.data);
  
      const tryBasic = async () =>
        api.post(`/api/video/${videoId}/analysis/`).then(r => r.data);
  
      const tryRestart = async () => {
        const candidates = [
          `/api/video/${videoId}/analysis/restart/`,
          `/api/video/${videoId}/analysis/reanalyze/`,
          `/api/video/${videoId}/analysis/restart`,
          `/api/video/${videoId}/reanalyze/`,
          `/api/video/${videoId}/reset/`,
        ];
        for (const url of candidates) {
          try {
            const { data } = await api.post(url);
            return data;
          } catch (e) {
          }
        }
        throw new Error('재분석 엔드포인트 없음');
      };
  
      let data;
      try {
        data = await tryForce();
      } catch {
        try {
          data = await tryBasic();
        } catch {
          data = await tryRestart();
        }
      }
  
      const status = (data?.status || data?.analysis_status || '').toLowerCase();
      const okStatuses = ['pending', 'started', 'processing', 'queued', 'queue', 'ok', 'success'];
      const doneStatuses = ['completed', 'done', 'finished', 'ready'];
  
      if (okStatuses.includes(status)) {
        alert('분석이 시작되었습니다.');
      } else if (doneStatuses.includes(status)) {
        alert('이미 완료된 분석이 있어 최신 상태로 갱신합니다.');
      } else {
        alert(data?.message || '분석 요청을 전송했습니다.');
      }
  
      await loadVideoList();
    } catch (error) {
      console.error('분석 시작 실패:', error);
      alert('분석 시작에 실패했습니다.');
    } finally {
      setRestartingId(null);
    }
  };

  const selectVideo = (video) => { navigate(`/video-chat/${video.id}`); };

  // 드롭존 스타일 (파란색 → 연그린 계열)
  const dropzoneStyle = (() => {
    if (uploading) return { borderColor: '#d1d5db', backgroundColor: '#f9fafb' };
    if (isHover || isDragOver) return { borderColor: BRAND_TINT_BORDER, backgroundColor: BRAND_TINT_BG };
    return { borderColor: NEUTRAL_BORDER, backgroundColor: 'white' };
  })();
  const dropzoneTextColor = (isHover || isDragOver) ? BRAND : '#374151';

  return (
    <div className="min-h-screen" style={{ background: "rgba(245, 242, 234, 0.4)" }}>
      <div className="max-w-4xl mx-auto p-6">
        <div className="mb-6">
          {/* 제목 색상 → #5d7c5b */}
          <h1 className="text-3xl font-bold mb-2" style={{ color: BRAND }}>영상 채팅</h1>
          <p className="text-gray-600">영상을 업로드하고 AI와 채팅해보세요</p>
        </div>

        {/* 업로드 영역 (연그린 hover) */}
        <div className="mb-8">
          <div
            className="border-2 border-dashed rounded-lg p-8 text-center transition-all duration-300"
            style={{ 
              ...dropzoneStyle,
              transform: (isHover || isDragOver) ? 'scale(1.02)' : 'scale(1.0)'
            }}
            onMouseEnter={() => setIsHover(true)}
            onMouseLeave={() => setIsHover(false)}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <input
              type="file"
              accept="video/*"
              onChange={handleFileUpload}
              className="hidden"
              id="video-upload"
              disabled={uploading}
            />
            <label htmlFor="video-upload" className={`flex flex-col items-center ${uploading ? 'cursor-not-allowed' : 'cursor-pointer'}`}>
              {uploading ? (
                <>
                  <Loader2 className="w-12 h-12 text-gray-500 animate-spin mb-4" />
                  <h3 className="text-lg font-semibold text-gray-700 mb-2">업로드 중...</h3>
                  <div className="w-full max-w-xs bg-gray-200 rounded-full h-2 mb-2">
                    <div className="h-2 rounded-full transition-all duration-300" style={{ width: `${uploadProgress}%`, backgroundColor: BRAND }} />
                  </div>
                  <p className="text-gray-500">{uploadProgress}% 완료</p>
                </>
              ) : (
                <>
                  <Upload className="w-12 h-12 mb-4" style={{ color: dropzoneTextColor }} />
                  <h3 className="text-lg font-semibold mb-2" style={{ color: dropzoneTextColor }}>
                    {isDragOver ? '영상을 여기에 놓으세요' : '영상 업로드'}
                  </h3>
                  <p className="mb-4" style={{ color: (isHover || isDragOver) ? BRAND : '#6b7280' }}>
                    {isDragOver ? '마우스를 놓으면 업로드됩니다' : '클릭하거나 드래그하여 영상 파일을 선택하세요'}
                  </p>
                  <p className="text-sm text-gray-400">MP4, AVI, MOV, MKV, WEBM 지원 (최대 50MB)</p>
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
                <div key={video.id} className="bg-white rounded-lg shadow-md p-4 hover:shadow-lg transition-shadow border">
                  <div className="flex items-center mb-3">
                    {/* FileVideo 아이콘 → #5d7c5b */}
                    <FileVideo className="w-8 h-8 mr-3" style={{ color: BRAND }} />
                    <div className="flex-1 min-w-0">
                      <h3 className="font-medium text-gray-800 truncate">{video.original_name}</h3>
                      <p className="text-sm text-gray-500">{new Date(video.uploaded_at).toLocaleDateString()}</p>
                    </div>
                  </div>

                  <div className="flex items-center justify-between text-sm mb-3">
                    <span className="text-gray-500">{(video.file_size / (1024 * 1024)).toFixed(1)}MB</span>
                    <div className="flex items-center">
                      {video.analysis_status === 'completed' && <CheckCircle className="w-4 h-4 text-green-500 mr-1" />}
                      {video.analysis_status === 'pending' && <Clock className="w-4 h-4 text-yellow-500 mr-1" />}
                      {video.analysis_status === 'failed' && <XCircle className="w-4 h-4 text-red-500 mr-1" />}
                      <span className={`text-xs font-medium ${
                        video.analysis_status === 'completed' ? 'text-green-600' :
                        video.analysis_status === 'pending' ? 'text-yellow-600' : 'text-red-600'
                      }`}>
                        {video.analysis_status === 'completed' ? '분석 완료' :
                         video.analysis_status === 'pending' ? '분석 중' : '분석 실패'}
                      </span>
                    </div>
                  </div>

                  {/* 액션 버튼들 */}
                  <div className="flex gap-2">
                    {video.analysis_status === 'completed' && (
                      <button
                        className="flex-1 px-6 py-3 rounded-lg transition-colors font-bold text-white"
                        style={{ backgroundColor: BRAND_BASE }}
                        onMouseEnter={(e) => { e.currentTarget.style.backgroundColor = BRAND_HOVER; }}
                        onMouseLeave={(e) => { e.currentTarget.style.backgroundColor = BRAND_BASE; }}
                        onClick={() => selectVideo(video)}
                      >
                        채팅하기
                      </button>
                    )}

                    {video.analysis_status === 'pending' && (
                      <div className="flex-1">
                        <button
                          disabled
                          className="w-full px-3 py-2 bg-gray-300 text-gray-500 text-sm rounded-lg cursor-not-allowed"
                        >
                          분석 중... ({video.analysis_progress || 0}%)
                        </button>
                        {video.analysis_message && (
                          <p className="text-xs text-gray-500 mt-1 text-center">{video.analysis_message}</p>
                        )}
                      </div>
                    )}

                    {video.analysis_status === 'failed' && (
                    <div className="flex-1">
                        <button
                        onClick={() => startAnalysis(video.id)}
                        className="w-full px-6 py-3 rounded-lg font-bold bg-gray-300 text-gray-600 hover:bg-gray-400 transition-colors flex items-center justify-center gap-2"
                        disabled={restartingId === video.id}
                        >
                        {restartingId === video.id && (
                            <Loader2 className="w-4 h-4 animate-spin" />
                        )}
                        다시 분석
                        </button>
                        {video.analysis_message && (
                        <p className="text-xs text-red-500 mt-1 text-center">{video.analysis_message}</p>
                        )}
                    </div>
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
    </div>
  );
};

export default VideoListPage;