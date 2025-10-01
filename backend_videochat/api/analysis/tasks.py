from celery import shared_task
from django.utils import timezone
from api.models import Video, AnalysisResult
from api.video_analyzer import get_video_analyzer
from api.db_builder import get_enhanced_video_rag_system
import json
import os

@shared_task
def analyze_video_task(video_id):
    """비디오 분석 Celery 작업"""
    try:
        video = Video.objects.get(id=video_id)
        video.status = 'processing'
        video.save()
        
        # 비디오 분석기 실행
        analyzer = get_video_analyzer()
        
        def progress_callback(progress, message):
            print(f"분석 진행률: {progress:.1f}% - {message}")
        
        # 고도화된 분석 실행
        result = analyzer.analyze_video_comprehensive_advanced(
            video, 
            analysis_type='comprehensive',
            progress_callback=progress_callback
        )
        
        if result['success']:
            # 분석 결과 저장
            analysis_result = AnalysisResult.objects.create(
                video=video,
                frame_results=result['frame_results'],
                person_database=result['person_database'],
                quality_metrics=result['quality_metrics'],
                analysis_config=result['analysis_config']
            )
            
            # JSON 파일로 저장
            json_path = f"media/analysis/{video_id}_analysis.json"
            os.makedirs(os.path.dirname(json_path), exist_ok=True)
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            # RAG 시스템에 데이터 로드
            rag_system = get_enhanced_video_rag_system()
            rag_system.process_video_analysis_json_advanced(json_path, video_id)
            
            # 비디오 상태 업데이트
            video.status = 'completed'
            video.is_analyzed = True
            video.analysis_result_path = json_path
            video.analysis_summary = result['video_summary']
            video.analysis_completed_at = timezone.now()
            video.save()
            
            return {'status': 'success', 'message': '분석 완료'}
            
        else:
            video.status = 'failed'
            video.save()
            return {'status': 'error', 'message': result.get('error', '분석 실패')}
            
    except Exception as e:
        if 'video' in locals():
            video.status = 'failed'
            video.save()
        return {'status': 'error', 'message': str(e)}

@shared_task
def analyze_video_enhanced(video_id, analysis_config):
    """향상된 비디오 분석 Celery 작업"""
    try:
        video = Video.objects.get(id=video_id)
        video.analysis_status = 'processing'
        video.save()
        
        print(f"🚀 비디오 {video_id}번 분석 시작: {video.original_name}")
        
        # 비디오 분석기 실행
        from api.video_analyzer import EnhancedVideoAnalyzer
        analyzer = EnhancedVideoAnalyzer()
        
        def progress_callback(progress, message):
            print(f"📊 비디오 {video_id}번 분석 진행률: {progress:.1f}% - {message}")
        
        # 향상된 분석 실행
        result = analyzer.analyze_video_enhanced(
            video, 
            analysis_config=analysis_config,
            progress_callback=progress_callback
        )
        
        if result['success']:
            # 비디오 상태 업데이트
            video.analysis_status = 'completed'
            video.is_analyzed = True
            video.processing_time = result.get('processing_time', 0)
            video.save()
            
            print(f"✅ 비디오 {video_id}번 분석 완료!")
            return {'status': 'success', 'message': '분석 완료', 'result': result}
            
        else:
            video.analysis_status = 'failed'
            video.save()
            print(f"❌ 비디오 {video_id}번 분석 실패: {result.get('error', '알 수 없는 오류')}")
            return {'status': 'error', 'message': result.get('error', '분석 실패')}
            
    except Exception as e:
        if 'video' in locals():
            video.analysis_status = 'failed'
            video.save()
        print(f"❌ 비디오 {video_id}번 분석 중 오류: {str(e)}")
        return {'status': 'error', 'message': str(e)}