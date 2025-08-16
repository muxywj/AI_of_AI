import React from 'react';
import { X } from 'lucide-react';

const Loginbar = ({ onClose }) => {
  return (
    <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50 z-40">
      <div className="bg-white p-6 rounded-lg shadow-lg w-96 relative">
        <X
          className="absolute top-3 right-3 w-6 h-6 cursor-pointer text-gray-500 hover:text-gray-700"
          onClick={onClose}
        />
        <h3 className="text-xl font-bold mb-2 text-left" style={{ color: '#2d3e2c' }}>로그인</h3>
        <p className="text-sm text-gray-600 mb-4 text-left">소셜 계정으로 간편하게 로그인하세요.</p>
        <hr className="w-full border-gray-300 mb-4" />

        {/* 소셜 로그인 버튼들 */}
        <div className="w-full flex flex-col space-y-3">
          <button 
            className="w-full p-3 border border-gray-200 rounded-lg bg-white transition-colors font-medium"
            style={{ color: '#2d3e2c' }}
            onMouseEnter={(e) => {
              e.target.style.backgroundColor = 'rgba(139, 168, 138, 0.05)';
              e.target.style.borderColor = 'rgba(139, 168, 138, 0.4)';
            }}
            onMouseLeave={(e) => {
              e.target.style.backgroundColor = 'white';
              e.target.style.borderColor = '#d1d5db';
            }}
          >
            Google로 로그인
          </button>
          <button 
            className="w-full p-3 border border-gray-200 rounded-lg bg-white transition-colors font-medium"
            style={{ color: '#2d3e2c' }}
            onMouseEnter={(e) => {
              e.target.style.backgroundColor = 'rgba(139, 168, 138, 0.05)';
              e.target.style.borderColor = 'rgba(139, 168, 138, 0.4)';
            }}
            onMouseLeave={(e) => {
              e.target.style.backgroundColor = 'white';
              e.target.style.borderColor = '#d1d5db';
            }}
          >
            Kakao로 로그인
          </button>
          <button 
            className="w-full p-3 border border-gray-200 rounded-lg bg-white transition-colors font-medium"
            style={{ color: '#2d3e2c' }}
            onMouseEnter={(e) => {
              e.target.style.backgroundColor = 'rgba(139, 168, 138, 0.05)';
              e.target.style.borderColor = 'rgba(139, 168, 138, 0.4)';
            }}
            onMouseLeave={(e) => {
              e.target.style.backgroundColor = 'white';
              e.target.style.borderColor = '#d1d5db';
            }}
          >
            Naver로 로그인
          </button>
        </div>
      </div>
    </div>
  );
};

export default Loginbar;