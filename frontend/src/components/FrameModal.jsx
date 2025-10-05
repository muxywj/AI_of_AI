import React, { useRef, useEffect } from 'react';
import { api } from '../utils/api';

const FrameModal = ({ frame, isOpen, onClose, showBboxOverlay, setShowBboxOverlay }) => {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (showBboxOverlay && frame && isOpen) {
      const img = document.getElementById('modal-frame-image');
      if (img && img.complete) {
        drawBboxOnCanvas(img, frame);
      }
    }
  }, [showBboxOverlay, frame, isOpen]);

  const drawBboxOnCanvas = (imageElement, frameData) => {
    if (!canvasRef.current || !imageElement) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    canvas.width = imageElement.naturalWidth;
    canvas.height = imageElement.naturalHeight;
    
    ctx.drawImage(imageElement, 0, 0);
    
    // 사람 bbox 그리기
    if (frameData.persons && frameData.persons.length > 0) {
      frameData.persons.forEach((person, index) => {
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
    
    // 객체 bbox 그리기
    if (frameData.objects && frameData.objects.length > 0) {
      frameData.objects.forEach((obj) => {
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

  if (!isOpen || !frame) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-2xl max-w-4xl max-h-[90vh] overflow-auto shadow-2xl">
        <div className="flex items-center justify-between p-4 border-b bg-gray-50 sticky top-0 z-10">
          <div className="flex items-center space-x-3">
            <div className="flex items-center bg-blue-50 text-blue-700 px-3 py-1 rounded-full text-sm font-medium">
              <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              {frame.timestamp.toFixed(1)}초
            </div>
            <div className="flex items-center bg-green-50 text-green-700 px-3 py-1 rounded-full text-sm font-medium">
              <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              {frame.relevance_score}점
            </div>
            <div className="text-sm text-gray-600">
              프레임 #{frame.image_id}
            </div>
          </div>
          <button
            onClick={onClose}
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
                    {showBboxOverlay ? '🎯 bbox ON' : '🎯 bbox OFF'}
                  </button>
                </div>
                
                {showBboxOverlay ? (
                  <canvas
                    ref={canvasRef}
                    className="w-full h-auto max-h-[60vh] object-contain rounded-lg shadow-lg"
                  />
                ) : (
                  <img
                    src={`${api.defaults.baseURL}${frame.image_url}`}
                    alt={`프레임 ${frame.image_id}`}
                    className="w-full h-auto max-h-[60vh] object-contain rounded-lg shadow-lg"
                    onError={(e) => {
                      console.error(`프레임 이미지 로드 실패: ${frame.image_url}`);
                      e.target.style.display = 'none';
                    }}
                  />
                )}
                
                <img
                  id="modal-frame-image"
                  src={`${api.defaults.baseURL}${frame.image_url}`}
                  alt={`프레임 ${frame.image_id}`}
                  style={{ display: 'none' }}
                  onLoad={(e) => {
                    if (showBboxOverlay) {
                      drawBboxOnCanvas(e.target, frame);
                    }
                  }}
                  onError={(e) => {
                    console.error(`프레임 이미지 로드 실패: ${frame.image_url}`);
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
                    <span className="font-medium">{frame.timestamp.toFixed(1)}초</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">관련도 점수:</span>
                    <span className="font-medium text-green-600">{frame.relevance_score}점</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">프레임 ID:</span>
                    <span className="font-medium">#{frame.image_id}</span>
                  </div>
                </div>
              </div>
              
              {frame.persons && frame.persons.length > 0 && (
                <div>
                  <h4 className="text-md font-semibold text-gray-800 mb-2 flex items-center">
                    <svg className="w-4 h-4 mr-2 text-purple-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                    </svg>
                    감지된 사람 ({frame.persons.length}명)
                  </h4>
                  <div className="space-y-2">
                    {frame.persons.map((person, index) => (
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
              
              {frame.objects && frame.objects.length > 0 && (
                <div>
                  <h4 className="text-md font-semibold text-gray-800 mb-2 flex items-center">
                    <svg className="w-4 h-4 mr-2 text-orange-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
                    </svg>
                    감지된 객체 ({frame.objects.length}개)
                  </h4>
                  <div className="space-y-2">
                    {frame.objects.map((obj, index) => (
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
  );
};

export default FrameModal;