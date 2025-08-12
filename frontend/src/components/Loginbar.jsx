import React from 'react';
import { X } from 'lucide-react';

const Loginbar = ({ onClose }) => {
  return (
    <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50 z-40">
      <div className="bg-white p-6 rounded-lg shadow-lg w-96 relative">
        <X
          className="absolute top-3 right-3 w-6 h-6 cursor-pointer"
          onClick={onClose}
        />
        <h3 className="text-xl font-bold mb-2 text-left">로그인</h3>
        <p className="text-sm text-gray-600 mb-4 text-left">소셜 계정으로 간편하게 로그인하세요.</p>
        <hr className="w-full border-gray-300 mb-4" />

        {/* 소셜 로그인 버튼들 */}
        <div className="w-full flex flex-col space-y-3">
          <button className="w-full p-3 border rounded-lg bg-white hover:bg-blue-50 transition-colors font-medium">
            Google로 로그인
          </button>
          <button className="w-full p-3 border rounded-lg bg-white hover:bg-yellow-50 transition-colors font-medium">
            Kakao로 로그인
          </button>
          <button className="w-full p-3 border rounded-lg bg-white hover:bg-green-50 transition-colors font-medium">
            Naver로 로그인
          </button>
        </div>
      </div>
    </div>
  );
};

export default Loginbar;