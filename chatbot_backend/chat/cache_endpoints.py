"""
캐시 관리 엔드포인트
새로고침 시 LLM 캐시 초기화하되, 세션 내에서는 대화 기억 유지
"""

from rest_framework.decorators import api_view, authentication_classes, permission_classes
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny
from .llm_cache_manager import llm_cache_manager, conversation_context_manager
import logging

logger = logging.getLogger(__name__)

@api_view(['POST'])
@authentication_classes([])
@permission_classes([AllowAny])
def clear_cache(request):
    """새로고침 시 LLM 캐시 초기화 (대화 맥락은 유지)"""
    try:
        session_id = request.data.get('user_id', 'default_user')
        
        # LLM 캐시만 초기화 (대화 맥락은 유지)
        llm_cache_manager.clear_session_cache(session_id)
        
        return Response({
            'success': True,
            'message': 'LLM 캐시가 초기화되었습니다.',
            'session_id': session_id
        })
        
    except Exception as e:
        logger.error(f"❌ 캐시 초기화 실패: {e}")
        return Response({
            'success': False,
            'error': f'캐시 초기화 실패: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
@authentication_classes([])
@permission_classes([AllowAny])
def get_conversation_context(request):
    """대화 맥락 조회 (새로고침 후에도 유지됨)"""
    try:
        session_id = request.GET.get('user_id', 'default_user')
        limit = int(request.GET.get('limit', 3))
        
        # 최근 대화 맥락 가져오기
        recent_context = conversation_context_manager.get_recent_context(session_id, limit)
        
        return Response({
            'success': True,
            'context': recent_context,
            'session_id': session_id
        })
        
    except Exception as e:
        logger.error(f"❌ 대화 맥락 조회 실패: {e}")
        return Response({
            'success': False,
            'error': f'대화 맥락 조회 실패: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
@authentication_classes([])
@permission_classes([AllowAny])
def clear_conversation_context(request):
    """대화 맥락 초기화"""
    try:
        session_id = request.data.get('user_id', 'default_user')
        
        # 대화 맥락 초기화
        conversation_context_manager.clear_context(session_id)
        
        return Response({
            'success': True,
            'message': '대화 맥락이 초기화되었습니다.',
            'session_id': session_id
        })
        
    except Exception as e:
        logger.error(f"❌ 대화 맥락 초기화 실패: {e}")
        return Response({
            'success': False,
            'error': f'대화 맥락 초기화 실패: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
@authentication_classes([])
@permission_classes([AllowAny])
def get_cache_statistics(request):
    """캐시 통계 정보 조회"""
    try:
        session_id = request.GET.get('user_id', 'default_user')
        
        # 세션 통계 정보 가져오기
        stats = llm_cache_manager.get_session_statistics(session_id)
        
        return Response({
            'success': True,
            'statistics': stats,
            'session_id': session_id
        })
        
    except Exception as e:
        logger.error(f"❌ 캐시 통계 조회 실패: {e}")
        return Response({
            'success': False,
            'error': f'캐시 통계 조회 실패: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
@authentication_classes([])
@permission_classes([AllowAny])
def get_verification_models(request):
    """사용 가능한 검증 모델 목록 조회"""
    try:
        from .factual_verification_system import factual_verification_system
        
        models = factual_verification_system.get_available_models()
        current_model = factual_verification_system.get_current_model()
        
        return Response({
            'success': True,
            'models': models,
            'current_model': current_model
        })
        
    except Exception as e:
        logger.error(f"❌ 검증 모델 목록 조회 실패: {e}")
        return Response({
            'success': False,
            'error': f'검증 모델 목록 조회 실패: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
@authentication_classes([])
@permission_classes([AllowAny])
def set_verification_model(request):
    """검증 모델 설정"""
    try:
        from .factual_verification_system import factual_verification_system
        
        model_name = request.data.get('model_name')
        
        if not model_name:
            return Response({
                'success': False,
                'error': 'model_name이 필요합니다.'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        success = factual_verification_system.set_verification_model(model_name)
        
        if success:
            return Response({
                'success': True,
                'message': f'검증 모델이 {model_name}로 변경되었습니다.',
                'current_model': factual_verification_system.get_current_model()
            })
        else:
            return Response({
                'success': False,
                'error': f'지원하지 않는 모델: {model_name}'
            }, status=status.HTTP_400_BAD_REQUEST)
        
    except Exception as e:
        logger.error(f"❌ 검증 모델 설정 실패: {e}")
        return Response({
            'success': False,
            'error': f'검증 모델 설정 실패: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
