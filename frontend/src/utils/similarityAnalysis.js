// utils/similarityAnalysis.js
import _ from 'lodash';

/**
 * 텍스트 유사도를 계산하는 함수
 * @param {string} text1 - 첫 번째 텍스트
 * @param {string} text2 - 두 번째 텍스트
 * @returns {number} 유사도 점수 (0~1)
 */
export const calculateTextSimilarity = (text1, text2) => {
  // 텍스트 정규화 (소문자 변환, 특수문자 제거 등)
  const normalizeText = (text) => {
    return text
      .toLowerCase()
      .replace(/[^\w\s]/g, '')
      .trim();
  };

  const normalized1 = normalizeText(text1);
  const normalized2 = normalizeText(text2);

  // 단어 집합 생성
  const words1 = normalized1.split(/\s+/);
  const words2 = normalized2.split(/\s+/);

  // 자카드 유사도 계산 (교집합 / 합집합)
  const intersection = _.intersection(words1, words2).length;
  const union = _.union(words1, words2).length;
  
  const jaccardSimilarity = union === 0 ? 0 : intersection / union;

  // 코사인 유사도를 위한 텍스트 벡터화
  const createWordVector = (text, wordSet) => {
    const vector = {};
    const words = text.split(/\s+/);
    
    // 각 단어의 빈도 계산
    words.forEach(word => {
      if (vector[word]) {
        vector[word]++;
      } else {
        vector[word] = 1;
      }
    });
    
    return vector;
  };

  // 두 텍스트의 모든 고유 단어 집합
  const wordSet = _.union(words1, words2);
  
  // 각 텍스트의 단어 벡터 생성
  const vector1 = createWordVector(normalized1, wordSet);
  const vector2 = createWordVector(normalized2, wordSet);

  // 코사인 유사도 계산을 위한 내적 및 벡터 크기
  let dotProduct = 0;
  let magnitude1 = 0;
  let magnitude2 = 0;

  // 모든 고유 단어에 대해 계산
  wordSet.forEach(word => {
    const val1 = vector1[word] || 0;
    const val2 = vector2[word] || 0;
    
    dotProduct += val1 * val2;
    magnitude1 += val1 * val1;
    magnitude2 += val2 * val2;
  });

  magnitude1 = Math.sqrt(magnitude1);
  magnitude2 = Math.sqrt(magnitude2);
  
  // 0으로 나누기 방지
  const cosineSimilarity = (magnitude1 * magnitude2 === 0) ? 0 : dotProduct / (magnitude1 * magnitude2);

  // 최종 유사도 점수 (자카드와 코사인의 가중 평균)
  return (jaccardSimilarity * 0.4) + (cosineSimilarity * 0.6);
};

/**
 * 여러 AI 모델의 응답을 유사도에 따라 군집화하는 함수
 * @param {Object} responses - 모델별 응답 객체 (key: 모델명, value: 응답 텍스트)
 * @param {number} threshold - 유사 응답으로 분류할 임계값 (0~1)
 * @returns {Object} 군집화된 응답 그룹
 */
export const clusterResponses = (responses, threshold = 0.85) => {
  const modelNames = Object.keys(responses);
  if (modelNames.length <= 1) {
    return { similarGroups: [modelNames], outliers: [] };
  }

  // 유사도 행렬 계산
  const similarityMatrix = {};
  modelNames.forEach(model1 => {
    similarityMatrix[model1] = {};
    modelNames.forEach(model2 => {
      if (model1 === model2) {
        similarityMatrix[model1][model2] = 1; // 자기 자신과의 유사도는 1
      } else if (similarityMatrix[model2] && similarityMatrix[model2][model1] !== undefined) {
        similarityMatrix[model1][model2] = similarityMatrix[model2][model1]; // 대칭성 활용
      } else {
        similarityMatrix[model1][model2] = calculateTextSimilarity(
          responses[model1], 
          responses[model2]
        );
      }
    });
  });

  // 군집화 (계층적 클러스터링 - 단순화된 구현)
  let clusters = modelNames.map(model => [model]); // 초기 클러스터: 각 모델이 별도 클러스터

  let mergeHappened = true;
  while (mergeHappened && clusters.length > 1) {
    mergeHappened = false;
    
    // 가장 유사한 두 클러스터 찾기
    let maxSimilarity = -1;
    let mergeIndices = [-1, -1];
    
    for (let i = 0; i < clusters.length; i++) {
      for (let j = i + 1; j < clusters.length; j++) {
        // 두 클러스터 간 평균 유사도 계산
        let clusterSimilarity = 0;
        let pairCount = 0;
        
        clusters[i].forEach(model1 => {
          clusters[j].forEach(model2 => {
            clusterSimilarity += similarityMatrix[model1][model2];
            pairCount++;
          });
        });
        
        const avgSimilarity = clusterSimilarity / pairCount;
        
        if (avgSimilarity > maxSimilarity) {
          maxSimilarity = avgSimilarity;
          mergeIndices = [i, j];
        }
      }
    }
    
    // 임계값보다 유사도가 높으면 클러스터 병합
    if (maxSimilarity >= threshold) {
      const [i, j] = mergeIndices;
      clusters[i] = [...clusters[i], ...clusters[j]];
      clusters.splice(j, 1);
      mergeHappened = true;
    } else {
      // 더 이상 병합할 클러스터가 없음
      break;
    }
  }

  // 결과 형식화: 유사 그룹과 이상치(outlier) 구분
  // 가장 큰 클러스터를 유사 그룹으로, 나머지를 이상치로 분류
  const sortedClusters = _.sortBy(clusters, cluster => -cluster.length);
  
  // 유사 응답 그룹 (첫 번째 클러스터)
  const similarGroup = sortedClusters[0] || [];
  
  // 이상치 (첫 번째 클러스터를 제외한 나머지)
  const outliers = sortedClusters.slice(1).flat();
  
  // 각 클러스터의 대표 응답 선정 (가장 긴 응답)
  const representativeResponses = sortedClusters.map(cluster => {
    const representative = _.maxBy(cluster, model => responses[model].length);
    return {
      models: cluster,
      representative,
      response: responses[representative]
    };
  });

  return {
    similarGroups: sortedClusters,
    outliers,
    similarGroup,
    representativeResponses
  };
};

// 응답 아티팩트를 분석하여 중요 특성 추출
export const extractResponseFeatures = (text) => {
  // 응답의 길이
  const length = text.length;
  
  // 코드 블록 포함 여부 및 개수
  const codeBlockCount = (text.match(/```[\s\S]*?```/g) || []).length;
  
  // 링크 포함 여부 및 개수
  const linkCount = (text.match(/\[.*?\]\(.*?\)/g) || []).length;
  
  // 목록 항목 개수
  const listItemCount = (text.match(/^[-*+] |^\d+\. /gm) || []).length;
  
  // 텍스트 복잡성 지표 (평균 문장 길이)
  const sentences = text.split(/[.!?]+/).filter(s => s.trim());
  const avgSentenceLength = sentences.length > 0 
    ? sentences.reduce((sum, s) => sum + s.length, 0) / sentences.length 
    : 0;
  
  return {
    length,
    codeBlockCount,
    linkCount,
    listItemCount,
    avgSentenceLength,
    hasCode: codeBlockCount > 0
  };
};