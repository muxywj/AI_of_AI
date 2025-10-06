"""
LLM 캐시 관리 시스템
새로고침 시 캐시 초기화하되, 세션 내에서는 대화 기억 유지
"""

import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from django.core.cache import cache
from django.conf import settings

logger = logging.getLogger(__name__)

class LLMCacheManager:
    """LLM 응답 캐시를 관리하는 클래스"""
    
    def __init__(self, cache_timeout: int = 1800):  # 30분
        self.cache_timeout = cache_timeout
        self.session_cache_timeout = 3600  # 1시간 (세션 유지)
    
    def get_session_key(self, session_id: str) -> str:
        """세션별 캐시 키 생성"""
        return f"llm_session_{session_id}"
    
    def get_cache_key(self, session_id: str, query: str) -> str:
        """쿼리별 캐시 키 생성"""
        import hashlib
        query_hash = hashlib.md5(query.encode()).hexdigest()
        return f"llm_cache_{session_id}_{query_hash}"
    
    def store_llm_response(self, session_id: str, query: str, ai_name: str, response: str) -> None:
        """LLM 응답을 캐시에 저장"""
        try:
            cache_key = self.get_cache_key(session_id, query)
            
            # 기존 캐시 가져오기
            cached_data = cache.get(cache_key, {})
            
            # 새로운 응답 추가
            cached_data[ai_name] = {
                'response': response,
                'timestamp': datetime.now().isoformat(),
                'query': query
            }
            
            # 캐시에 저장 (짧은 시간만 유지)
            cache.set(cache_key, cached_data, self.cache_timeout)
            
            # 세션 정보도 업데이트
            self._update_session_info(session_id, query, ai_name)
            
            logger.info(f"✅ LLM 응답 캐시 저장: {ai_name} - {query[:50]}...")
            
        except Exception as e:
            logger.error(f"❌ LLM 응답 캐시 저장 실패: {e}")
    
    def get_llm_response(self, session_id: str, query: str, ai_name: str) -> Optional[str]:
        """캐시에서 LLM 응답 가져오기"""
        try:
            cache_key = self.get_cache_key(session_id, query)
            cached_data = cache.get(cache_key, {})
            
            if ai_name in cached_data:
                logger.info(f"✅ LLM 응답 캐시 히트: {ai_name} - {query[:50]}...")
                return cached_data[ai_name]['response']
            
            return None
            
        except Exception as e:
            logger.error(f"❌ LLM 응답 캐시 조회 실패: {e}")
            return None
    
    def get_all_llm_responses(self, session_id: str, query: str) -> Dict[str, str]:
        """특정 쿼리에 대한 모든 LLM 응답 가져오기"""
        try:
            cache_key = self.get_cache_key(session_id, query)
            cached_data = cache.get(cache_key, {})
            
            responses = {}
            for ai_name, data in cached_data.items():
                responses[ai_name] = data['response']
            
            return responses
            
        except Exception as e:
            logger.error(f"❌ 모든 LLM 응답 조회 실패: {e}")
            return {}
    
    def clear_session_cache(self, session_id: str) -> None:
        """특정 세션의 모든 LLM 캐시 초기화"""
        try:
            session_key = self.get_session_key(session_id)
            session_info = cache.get(session_key, {})
            
            # 세션에 저장된 모든 캐시 키 삭제
            for cache_key in session_info.get('cache_keys', []):
                cache.delete(cache_key)
            
            # 세션 정보도 삭제
            cache.delete(session_key)
            
            logger.info(f"✅ 세션 LLM 캐시 초기화: {session_id}")
            
        except Exception as e:
            logger.error(f"❌ 세션 LLM 캐시 초기화 실패: {e}")
    
    def clear_all_cache(self) -> None:
        """모든 LLM 캐시 초기화"""
        try:
            # 패턴으로 모든 LLM 관련 캐시 삭제
            from django.core.cache.utils import make_template_fragment_key
            
            # 모든 세션 캐시 삭제
            cache.delete_many(cache.keys("llm_session_*"))
            cache.delete_many(cache.keys("llm_cache_*"))
            
            logger.info("✅ 모든 LLM 캐시 초기화 완료")
            
        except Exception as e:
            logger.error(f"❌ 모든 LLM 캐시 초기화 실패: {e}")
    
    def is_cache_valid(self, session_id: str, query: str) -> bool:
        """캐시가 유효한지 확인"""
        try:
            cache_key = self.get_cache_key(session_id, query)
            cached_data = cache.get(cache_key)
            
            if not cached_data:
                return False
            
            # 최소 3개 AI 응답이 있어야 유효
            return len(cached_data) >= 3
            
        except Exception as e:
            logger.error(f"❌ 캐시 유효성 확인 실패: {e}")
            return False
    
    def get_session_statistics(self, session_id: str) -> Dict[str, Any]:
        """세션 통계 정보 가져오기"""
        try:
            session_key = self.get_session_key(session_id)
            session_info = cache.get(session_key, {})
            
            stats = {
                'total_queries': len(session_info.get('queries', [])),
                'cache_hits': session_info.get('cache_hits', 0),
                'cache_misses': session_info.get('cache_misses', 0),
                'last_activity': session_info.get('last_activity'),
                'session_start': session_info.get('session_start')
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ 세션 통계 조회 실패: {e}")
            return {}
    
    def _update_session_info(self, session_id: str, query: str, ai_name: str) -> None:
        """세션 정보 업데이트"""
        try:
            session_key = self.get_session_key(session_id)
            session_info = cache.get(session_key, {
                'session_start': datetime.now().isoformat(),
                'queries': [],
                'cache_keys': [],
                'cache_hits': 0,
                'cache_misses': 0
            })
            
            # 쿼리 추가 (중복 제거)
            if query not in session_info['queries']:
                session_info['queries'].append(query)
            
            # 캐시 키 추가
            cache_key = self.get_cache_key(session_id, query)
            if cache_key not in session_info['cache_keys']:
                session_info['cache_keys'].append(cache_key)
            
            # 마지막 활동 시간 업데이트
            session_info['last_activity'] = datetime.now().isoformat()
            
            # 세션 정보 저장
            cache.set(session_key, session_info, self.session_cache_timeout)
            
        except Exception as e:
            logger.error(f"❌ 세션 정보 업데이트 실패: {e}")

class ConversationContextManager:
    """대화 맥락 관리자 (새로고침 시 초기화되지 않음)"""
    
    def __init__(self, max_context_length: int = 10, cache_timeout: int = 7200):  # 2시간
        self.max_context_length = max_context_length
        self.cache_timeout = cache_timeout
    
    def get_context_key(self, session_id: str) -> str:
        """맥락 캐시 키 생성"""
        return f"conversation_context_{session_id}"
    
    def add_conversation(self, session_id: str, user_message: str, 
                        ai_responses: Dict[str, str], optimal_response: str = "") -> None:
        """대화 추가 (새로고침 후에도 유지)"""
        try:
            context_key = self.get_context_key(session_id)
            context = cache.get(context_key, {
                'session_id': session_id,
                'created_at': datetime.now().isoformat(),
                'conversations': []
            })
            
            # 새로운 대화 추가
            conversation_entry = {
                'timestamp': datetime.now().isoformat(),
                'user_message': user_message,
                'ai_responses': ai_responses,
                'optimal_response': optimal_response,
                'intent': self._extract_intent(user_message)
            }
            
            context['conversations'].append(conversation_entry)
            
            # 최대 길이 제한
            if len(context['conversations']) > self.max_context_length:
                context['conversations'] = context['conversations'][-self.max_context_length:]
            
            # 캐시에 저장 (긴 시간 유지)
            cache.set(context_key, context, self.cache_timeout)
            
            logger.info(f"✅ 대화 맥락 추가: {session_id}, 총 {len(context['conversations'])}개 대화")
            
        except Exception as e:
            logger.error(f"❌ 대화 맥락 추가 실패: {e}")
    
    def get_recent_context(self, session_id: str, limit: int = 3) -> List[Dict[str, Any]]:
        """최근 대화 맥락 가져오기"""
        try:
            context_key = self.get_context_key(session_id)
            context = cache.get(context_key, {'conversations': []})
            
            return context['conversations'][-limit:]
            
        except Exception as e:
            logger.error(f"❌ 대화 맥락 조회 실패: {e}")
            return []
    
    def generate_context_prompt(self, session_id: str, current_message: str) -> str:
        """맥락을 포함한 프롬프트 생성"""
        try:
            recent_conversations = self.get_recent_context(session_id, 3)
            
            if not recent_conversations:
                return ""
            
            prompt_parts = ["📝 최근 대화 맥락:"]
            
            for conv in recent_conversations:
                prompt_parts.append(f"- 사용자: {conv['user_message']}")
                if conv.get('optimal_response'):
                    prompt_parts.append(f"- 최적 답변: {conv['optimal_response'][:100]}...")
                else:
                    # 최적 답변이 없으면 가장 좋은 AI 응답 사용
                    best_response = self._select_best_response(conv.get('ai_responses', {}))
                    if best_response:
                        prompt_parts.append(f"- AI: {best_response[:100]}...")
            
            context_prompt = "\n".join(prompt_parts)
            return f"다음은 이전 대화 맥락입니다:\n\n{context_prompt}\n\n위 맥락을 고려하여 자연스럽게 응답해주세요."
            
        except Exception as e:
            logger.error(f"❌ 맥락 프롬프트 생성 실패: {e}")
            return ""
    
    def clear_context(self, session_id: str) -> None:
        """대화 맥락 초기화"""
        try:
            context_key = self.get_context_key(session_id)
            cache.delete(context_key)
            logger.info(f"✅ 대화 맥락 초기화: {session_id}")
            
        except Exception as e:
            logger.error(f"❌ 대화 맥락 초기화 실패: {e}")
    
    def _extract_intent(self, message: str) -> str:
        """메시지에서 의도 추출"""
        try:
            message_lower = message.lower()
            
            intent_keywords = {
                'greeting': ['안녕', 'hello', 'hi', '좋은', '하루'],
                'question': ['뭐', '어떻게', '왜', '언제', '어디', '누구'],
                'request': ['해줘', '알려줘', '설명', '도와줘'],
                'thanks': ['고마워', '감사', 'thank', 'thanks']
            }
            
            for intent, keywords in intent_keywords.items():
                if any(keyword in message_lower for keyword in keywords):
                    return intent
            
            return 'general'
            
        except Exception as e:
            logger.error(f"❌ 의도 추출 실패: {e}")
            return 'general'
    
    def _select_best_response(self, ai_responses: Dict[str, str]) -> str:
        """가장 좋은 AI 응답 선택"""
        try:
            if not ai_responses:
                return ""
            
            # 간단한 선택 로직
            best_response = ""
            best_score = 0
            
            for ai_name, response in ai_responses.items():
                score = len(response)  # 길이 기준
                
                # AI별 가중치
                if ai_name == 'gpt':
                    score *= 1.2
                elif ai_name == 'claude':
                    score *= 1.1
                
                if score > best_score:
                    best_score = score
                    best_response = response
            
            return best_response
            
        except Exception as e:
            logger.error(f"❌ 최적 응답 선택 실패: {e}")
            return list(ai_responses.values())[0] if ai_responses else ""

# 전역 인스턴스 생성
llm_cache_manager = LLMCacheManager()
conversation_context_manager = ConversationContextManager()
