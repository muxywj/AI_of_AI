import React from "react";
import { AlertTriangle, Check, Globe, Code, BookOpen, List, Link as LinkIcon, BarChart4 } from "lucide-react";

const SimilarityDetailModal = ({ isOpen, onClose, similarityData }) => {
  if (!isOpen) return null;

  const {
    messageId,
    noDataAvailable,
    debugInfo,
    availableDataKeys,
    similarGroups,
    mainGroup,
    semanticTags = {},
    responseFeatures = {},
    similarityMatrix,
    detectedLanguages
  } = similarityData || {};

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 overflow-y-auto">
      <div className="bg-white rounded-lg p-6 w-full max-w-4xl max-h-[90vh] overflow-y-auto">
        {/* 헤더 */}
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-bold">유사도 분석 상세 결과</h2>
          <button onClick={onClose} className="text-gray-500 hover:text-gray-800">✕</button>
        </div>

        {/* 데이터 없음 표시 */}
        {noDataAvailable ? (
          <div className="p-4 bg-yellow-50 text-yellow-800 rounded-lg">
            <p>아직 이 메시지에 대한 유사도 분석 데이터가 준비되지 않았습니다.</p>
            <p>잠시 후 다시 시도해주세요.</p>
            <p className="mt-4 text-sm font-semibold">디버그 정보:</p>
            <pre className="bg-gray-100 p-2 rounded text-xs overflow-auto mt-2">
              {JSON.stringify({ messageId, debugInfo, availableDataKeys }, null, 2)}
            </pre>
          </div>
        ) : (
          <div className="space-y-6">
            {/* 다국어 모델 정보 */}
            <div className="p-4 bg-indigo-50 rounded-lg">
              <h3 className="font-bold mb-2 flex items-center">
                <Globe className="mr-2 text-indigo-600" size={20} />
                다국어 유사도 모델
              </h3>
              <div className="bg-white p-3 rounded-lg shadow-sm">
                <p className="font-medium">paraphrase-multilingual-MiniLM-L12-v2 모델 사용 중</p>
                {detectedLanguages && (
                  <div className="mt-2 pt-2 border-t">
                    <p className="font-medium">감지된 언어:</p>
                    <div className="flex flex-wrap gap-2 mt-1">
                      {Object.entries(detectedLanguages).map(([model, lang]) => (
                        <span key={model} className="px-2 py-1 bg-indigo-100 text-indigo-800 rounded-full text-xs">
                          {model}: {lang}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* 유사 그룹 */}
            <div className="p-4 bg-gray-50 rounded-lg">
              <h3 className="font-bold mb-2 flex items-center">
                <BarChart4 className="mr-2 text-blue-600" size={20} />
                유사 그룹
              </h3>
              {(similarGroups || mainGroup) ? (
                (similarGroups || [mainGroup]).map((group, idx) => (
                  <div
                    key={idx}
                    className={`mb-3 p-3 rounded-lg ${
                      idx === 0 ? "bg-blue-50" : idx === 1 ? "bg-green-50" : "bg-yellow-50"
                    }`}
                  >
                    <p className="font-medium">
                      {idx === 0 ? "주요 그룹" : idx === 1 ? "부 그룹" : `그룹 ${idx + 1}`}
                    </p>
                    <p>{Array.isArray(group) ? group.join(", ") : group}</p>
                  </div>
                ))
              ) : (
                <p>유사 그룹 데이터가 없습니다.</p>
              )}
            </div>

            {/* 의미적 태그 */}
            <div className="p-4 bg-gray-50 rounded-lg">
              <h3 className="font-bold mb-2 flex items-center">
                <BookOpen className="mr-2 text-purple-600" size={20} />
                의미적 태그
              </h3>
              {Object.keys(semanticTags).length > 0 ? (
                <div className="flex flex-wrap gap-2">
                  {Object.entries(semanticTags).map(([model, tag]) => (
                    <span
                      key={model}
                      className="px-2 py-1 bg-purple-100 text-purple-800 rounded-full text-xs"
                    >
                      {model}: {tag}
                    </span>
                  ))}
                </div>
              ) : (
                <p>의미적 태그 데이터가 없습니다.</p>
              )}
            </div>

            { /* 응답 특성 */ }
<div className="p-4 bg-gray-50 rounded-lg">
  <h3 className="font-bold mb-2 flex items-center">
    <Check className="mr-2 text-green-600" size={20} />
    응답 특성
  </h3>
  {Object.keys(responseFeatures).length > 0 ? (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {Object.entries(responseFeatures).map(([model, f]) => (
        <div key={model} className="p-3 bg-white rounded-lg shadow-sm">
          <p className="font-medium border-b pb-2 mb-2">{model}</p>
          {f.detectedLang && (
            <p className="text-sm mb-1">감지 언어: {f.detectedLang}</p>
          )}
          {f.codeBlockCount > 0 && (
            <p className="text-sm mb-1">코드 블록: {f.codeBlockCount}개</p>
          )}
          {f.listItemCount > 0 && (
            <p className="text-sm mb-1">목록 항목: {f.listItemCount}개</p>
          )}
          {f.linkCount > 0 && (
            <p className="text-sm mb-1">링크: {f.linkCount}개</p>
          )}
          { /* 문자열이든 숫자든 값이 있으면 렌더링 */ }
          {f.length != null && (
            <p className="text-sm">글자 수: {parseInt(f.length, 10)}자</p>
          )}
        </div>
      ))}
    </div>
  ) : (
    <p>응답 특성 데이터가 없습니다.</p>
  )}
</div>



            {/* 유사도 행렬 */}
            <div className="p-4 bg-gray-50 rounded-lg">
              <h3 className="font-bold mb-2 flex items-center">
                <BarChart4 className="mr-2 text-blue-600" size={20} />
                유사도 행렬
              </h3>
              {similarityMatrix ? (
                <div className="overflow-x-auto bg-white p-3 rounded-lg shadow-sm">
                  <table className="min-w-full border">
                    <thead>
                      <tr className="bg-gray-50">
                        <th className="border p-2">모델</th>
                        {Object.keys(similarityMatrix).map(m => (
                          <th key={m} className="border p-2">{m}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(similarityMatrix).map(([m, sims]) => (
                        <tr key={m}>
                          <td className="border p-2 font-medium bg-gray-50">{m}</td>
                          {Object.values(sims).map((val, i) => {
                            const pct = parseFloat(val) * 100;
                            let bg = "bg-white";
                            if (pct >= 90) bg = "bg-green-200";
                            else if (pct >= 70) bg = "bg-green-100";
                            else if (pct >= 50) bg = "bg-yellow-100";
                            else if (pct >= 30) bg = "bg-orange-100";
                            else bg = "bg-red-100";
                            return (
                              <td key={i} className={`border p-2 text-center ${bg}`}>
                                {pct.toFixed(1)}%
                              </td>
                            );
                          })}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <p>유사도 행렬 데이터가 없습니다.</p>
              )}
            </div>

            {/* 개발자 모드: 원본 데이터 */}
            <div className="p-4 bg-gray-50 rounded-lg">
              <h3 className="font-bold mb-2 flex items-center">
                <Code className="mr-2 text-gray-600" size={20} />
                개발자 정보
              </h3>
              <pre className="bg-gray-900 text-green-400 p-3 rounded-lg overflow-auto max-h-48 text-xs">
                {JSON.stringify(similarityData, null, 2)}
              </pre>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default SimilarityDetailModal;
