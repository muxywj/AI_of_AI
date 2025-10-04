import React, { useState } from 'react';
import { Search, Clock, Users, Palette, MapPin, Zap, Brain, Sparkles, BarChart3, Eye } from 'lucide-react';
import { api } from '../utils/api';

const EnhancedSearchPanel = ({ onSearch, videoList, isLoading, selectedVideo }) => {
  const [searchMode, setSearchMode] = useState('natural'); // natural, advanced, semantic
  const [query, setQuery] = useState('');
  const [searchType, setSearchType] = useState('auto'); // auto, cross_video, intra_video, time_analysis
  const [filters, setFilters] = useState({
    colors: [],
    gender: '',
    timeRange: { start: '', end: '' },
    weather: '',
    location: ''
  });
  const [searchInsights, setSearchInsights] = useState(null);
  const [showInsights, setShowInsights] = useState(false);

  const colorOptions = [
    { name: '빨간색', value: 'red' },
    { name: '주황색', value: 'orange' },
    { name: '노란색', value: 'yellow' },
    { name: '초록색', value: 'green' },
    { name: '파란색', value: 'blue' },
    { name: '보라색', value: 'purple' },
    { name: '분홍색', value: 'pink' },
    { name: '검은색', value: 'black' },
    { name: '흰색', value: 'white' },
    { name: '회색', value: 'gray' }
  ];

  const genderOptions = [
    { name: '전체', value: '' },
    { name: '남성', value: 'male' },
    { name: '여성', value: 'female' }
  ];

  const weatherOptions = [
    { name: '맑음', value: 'sunny' },
    { name: '흐림', value: 'cloudy' },
    { name: '비', value: 'rainy' },
    { name: '눈', value: 'snowy' },
    { name: '밤', value: 'night' }
  ];

  const searchModeOptions = [
    { value: 'natural', label: '자연어 검색', icon: Brain, description: 'LLM이 이해하는 자연어로 검색' },
    { value: 'semantic', label: '의미적 검색', icon: Sparkles, description: '의미적 유사도 기반 검색' },
    { value: 'advanced', label: '고급 필터', icon: BarChart3, description: '세부 조건으로 정확한 검색' }
  ];

  const handleColorToggle = (color) => {
    setFilters(prev => ({
      ...prev,
      colors: prev.colors.includes(color)
        ? prev.colors.filter(c => c !== color)
        : [...prev.colors, color]
    }));
  };

  const handleNaturalLanguageSearch = async () => {
    if (!query.trim()) return;

    try {
      // 자연어 쿼리 처리 API 호출
      const response = await api.post('/api/natural-language-query/', {
        query: query.trim(),
        video_id: selectedVideo?.id
      });

      const searchData = {
        query: query.trim(),
        search_type: 'natural',
        parsed_query: response.data.parsed_query,
        search_results: response.data.search_results,
        query_analysis: response.data.query_analysis
      };

      onSearch(searchData);
      
      // 검색 인사이트 생성
      if (response.data.search_results?.results) {
        await generateSearchInsights(query.trim(), response.data.search_results.results);
      }
    } catch (error) {
      console.error('자연어 검색 실패:', error);
      // 폴백: 기존 검색 방식 사용
      handleAdvancedSearch();
    }
  };

  const handleSemanticSearch = async () => {
    if (!query.trim()) return;

    try {
      const response = await api.post('/api/semantic-search/', {
        query: query.trim(),
        video_id: selectedVideo?.id,
        search_type: 'semantic'
      });

      const searchData = {
        query: query.trim(),
        search_type: 'semantic',
        results: response.data.results,
        search_metadata: response.data.search_metadata
      };

      onSearch(searchData);
      
      // 검색 인사이트 생성
      if (response.data.results) {
        await generateSearchInsights(query.trim(), response.data.results);
      }
    } catch (error) {
      console.error('의미적 검색 실패:', error);
      // 폴백: 기존 검색 방식 사용
      handleAdvancedSearch();
    }
  };

  const handleAdvancedSearch = () => {
    if (!query.trim()) return;

    const searchData = {
      query: query.trim(),
      search_type: searchType,
      ...filters
    };

    onSearch(searchData);
  };

  const generateSearchInsights = async (query, results) => {
    try {
      const response = await api.post('/api/search-insights/', {
        query: query,
        search_results: results
      });

      setSearchInsights(response.data.insights);
      setShowInsights(true);
    } catch (error) {
      console.error('검색 인사이트 생성 실패:', error);
    }
  };

  const handleSearch = () => {
    switch (searchMode) {
      case 'natural':
        handleNaturalLanguageSearch();
        break;
      case 'semantic':
        handleSemanticSearch();
        break;
      case 'advanced':
        handleAdvancedSearch();
        break;
      default:
        handleAdvancedSearch();
    }
  };

  const getSearchTypeIcon = (type) => {
    switch (type) {
      case 'cross_video': return <MapPin className="w-4 h-4" />;
      case 'intra_video': return <Users className="w-4 h-4" />;
      case 'time_analysis': return <Clock className="w-4 h-4" />;
      default: return <Zap className="w-4 h-4" />;
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
      <div className="flex items-center gap-2 mb-4">
        <Search className="w-5 h-5 text-blue-600" />
        <h3 className="text-lg font-semibold text-gray-800">LLM 기반 비디오 검색</h3>
      </div>

      {/* 검색 모드 선택 */}
      <div className="mb-6">
        <label className="block text-sm font-medium text-gray-700 mb-3">검색 모드</label>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {searchModeOptions.map(mode => {
            const IconComponent = mode.icon;
            return (
              <button
                key={mode.value}
                onClick={() => setSearchMode(mode.value)}
                className={`p-4 rounded-lg border-2 transition-all ${
                  searchMode === mode.value
                    ? 'border-blue-500 bg-blue-50 text-blue-700'
                    : 'border-gray-200 bg-white text-gray-600 hover:border-gray-300'
                }`}
              >
                <div className="flex items-center gap-2 mb-2">
                  <IconComponent className="w-5 h-5" />
                  <span className="font-medium">{mode.label}</span>
                </div>
                <p className="text-xs text-gray-500">{mode.description}</p>
              </button>
            );
          })}
        </div>
      </div>

      {/* 검색어 입력 */}
      <div className="mb-4">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          {searchMode === 'natural' ? '자연어 검색어' : '검색어'}
        </label>
        <div className="relative">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder={
              searchMode === 'natural' 
                ? "예: 비가오는 밤에 촬영된 영상을 찾아줘, 이 영상에서 주황색 상의를 입은 남성이 지나간 장면을 추적해줘"
                : "예: 비가오는 밤에 촬영된 영상, 주황색 상의를 입은 남성, 3:00~5:00분 사이 성비 분포"
            }
            className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent pr-12"
          />
          {searchMode === 'natural' && (
            <div className="absolute right-3 top-1/2 transform -translate-y-1/2">
              <Brain className="w-5 h-5 text-blue-500" />
            </div>
          )}
        </div>
      </div>

      {/* 고급 필터 (advanced 모드에서만 표시) */}
      {searchMode === 'advanced' && (
        <div className="space-y-4 mb-4">
          {/* 검색 타입 선택 */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">검색 타입</label>
            <div className="flex gap-2 flex-wrap">
              {[
                { value: 'auto', label: '자동 감지' },
                { value: 'cross_video', label: '영상 간 검색' },
                { value: 'intra_video', label: '영상 내 검색' },
                { value: 'time_analysis', label: '시간대 분석' }
              ].map(type => (
                <button
                  key={type.value}
                  onClick={() => setSearchType(type.value)}
                  className={`flex items-center gap-1 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                    searchType === type.value
                      ? 'bg-blue-100 text-blue-700 border border-blue-300'
                      : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                  }`}
                >
                  {getSearchTypeIcon(type.value)}
                  {type.label}
                </button>
              ))}
            </div>
          </div>

          {/* 색상 필터 */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              <Palette className="w-4 h-4 inline mr-1" />
              색상 필터
            </label>
            <div className="flex flex-wrap gap-2">
              {colorOptions.map(color => (
                <button
                  key={color.value}
                  onClick={() => handleColorToggle(color.value)}
                  className={`px-3 py-1 rounded-full text-sm font-medium transition-colors ${
                    filters.colors.includes(color.value)
                      ? 'bg-blue-100 text-blue-700 border border-blue-300'
                      : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                  }`}
                >
                  {color.name}
                </button>
              ))}
            </div>
          </div>

          {/* 성별 필터 */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              <Users className="w-4 h-4 inline mr-1" />
              성별 필터
            </label>
            <div className="flex gap-2">
              {genderOptions.map(gender => (
                <button
                  key={gender.value}
                  onClick={() => setFilters(prev => ({ ...prev, gender: gender.value }))}
                  className={`px-3 py-1 rounded-lg text-sm font-medium transition-colors ${
                    filters.gender === gender.value
                      ? 'bg-blue-100 text-blue-700 border border-blue-300'
                      : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                  }`}
                >
                  {gender.name}
                </button>
              ))}
            </div>
          </div>

          {/* 시간 범위 필터 */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              <Clock className="w-4 h-4 inline mr-1" />
              시간 범위 (분:초)
            </label>
            <div className="flex gap-2">
              <input
                type="text"
                value={filters.timeRange.start}
                onChange={(e) => setFilters(prev => ({
                  ...prev,
                  timeRange: { ...prev.timeRange, start: e.target.value }
                }))}
                placeholder="시작 (예: 3:00)"
                className="flex-1 px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
              <span className="self-center text-gray-500">~</span>
              <input
                type="text"
                value={filters.timeRange.end}
                onChange={(e) => setFilters(prev => ({
                  ...prev,
                  timeRange: { ...prev.timeRange, end: e.target.value }
                }))}
                placeholder="끝 (예: 5:00)"
                className="flex-1 px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>
          </div>

          {/* 날씨 필터 */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">날씨 조건</label>
            <div className="flex flex-wrap gap-2">
              {weatherOptions.map(weather => (
                <button
                  key={weather.value}
                  onClick={() => setFilters(prev => ({ ...prev, weather: weather.value }))}
                  className={`px-3 py-1 rounded-lg text-sm font-medium transition-colors ${
                    filters.weather === weather.value
                      ? 'bg-blue-100 text-blue-700 border border-blue-300'
                      : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                  }`}
                >
                  {weather.name}
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* 검색 버튼 */}
      <div className="mt-6">
        <button
          onClick={handleSearch}
          disabled={!query.trim() || isLoading}
          className="w-full bg-blue-600 text-white py-3 px-4 rounded-lg font-medium hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-2"
        >
          {isLoading ? (
            <>
              <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
              검색 중...
            </>
          ) : (
            <>
              {searchMode === 'natural' && <Brain className="w-4 h-4" />}
              {searchMode === 'semantic' && <Sparkles className="w-4 h-4" />}
              {searchMode === 'advanced' && <Search className="w-4 h-4" />}
              {searchMode === 'natural' ? '자연어 검색' : 
               searchMode === 'semantic' ? '의미적 검색' : '검색 실행'}
            </>
          )}
        </button>
      </div>

      {/* 검색 인사이트 */}
      {searchInsights && showInsights && (
        <div className="mt-4 p-4 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg border border-blue-200">
          <div className="flex items-center justify-between mb-3">
            <h4 className="text-sm font-medium text-blue-800 flex items-center gap-2">
              <Eye className="w-4 h-4" />
              검색 인사이트
            </h4>
            <button
              onClick={() => setShowInsights(false)}
              className="text-blue-600 hover:text-blue-800"
            >
              ✕
            </button>
          </div>
          <div className="text-sm text-blue-700 space-y-2">
            <p className="font-medium">{searchInsights.summary}</p>
            {searchInsights.key_findings && searchInsights.key_findings.length > 0 && (
              <div>
                <p className="font-medium mb-1">주요 발견사항:</p>
                <ul className="list-disc list-inside space-y-1">
                  {searchInsights.key_findings.map((finding, index) => (
                    <li key={index}>{finding}</li>
                  ))}
                </ul>
              </div>
            )}
            {searchInsights.recommendations && searchInsights.recommendations.length > 0 && (
              <div>
                <p className="font-medium mb-1">추천사항:</p>
                <ul className="list-disc list-inside space-y-1">
                  {searchInsights.recommendations.map((rec, index) => (
                    <li key={index}>{rec}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>
      )}

      {/* 검색 힌트 */}
      <div className="mt-4 p-3 bg-blue-50 rounded-lg">
        <h4 className="text-sm font-medium text-blue-800 mb-2">검색 예시:</h4>
        <ul className="text-xs text-blue-700 space-y-1">
          <li>• <strong>자연어:</strong> "비가오는 밤에 촬영된 영상을 찾아줘"</li>
          <li>• <strong>의미적:</strong> "어두운 밤 거리에서 사람들이 걷는 장면"</li>
          <li>• <strong>고급:</strong> 색상, 시간대, 성별 등 세부 조건으로 검색</li>
        </ul>
      </div>
    </div>
  );
};

export default EnhancedSearchPanel;
