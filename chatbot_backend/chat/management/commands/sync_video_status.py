from django.core.management.base import BaseCommand
from chat.models import Video
from chat.services.video_analysis_service import VideoAnalysisService
import os
from django.conf import settings

class Command(BaseCommand):
    help = '모든 영상의 데이터베이스 상태와 실제 파일 상태를 동기화합니다'

    def add_arguments(self, parser):
        parser.add_argument(
            '--video-id',
            type=int,
            help='특정 영상 ID만 동기화 (지정하지 않으면 모든 영상 동기화)',
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='실제 변경 없이 상태만 확인',
        )

    def handle(self, *args, **options):
        video_analysis_service = VideoAnalysisService()
        
        if options['video_id']:
            videos = Video.objects.filter(id=options['video_id'])
            if not videos.exists():
                self.stdout.write(
                    self.style.ERROR(f'영상 ID {options["video_id"]}를 찾을 수 없습니다.')
                )
                return
        else:
            videos = Video.objects.all()
        
        self.stdout.write(f'총 {videos.count()}개의 영상을 동기화합니다...')
        
        synced_count = 0
        error_count = 0
        
        for video in videos:
            try:
                self.stdout.write(f'영상 {video.id} 동기화 중...')
                
                # 현재 상태 확인
                old_status = video.analysis_status
                old_progress = video.analysis_progress
                old_message = video.analysis_message
                
                if options['dry_run']:
                    # Dry run: 실제 파일 확인만
                    analysis_file_exists = False
                    if video.analysis_json_path:
                        full_path = os.path.join(settings.MEDIA_ROOT, video.analysis_json_path)
                        analysis_file_exists = os.path.exists(full_path)
                    
                    frame_files_exist = False
                    if video.frame_images_path:
                        frame_paths = video.frame_images_path.split(',')
                        frame_files_exist = all(
                            os.path.exists(os.path.join(settings.MEDIA_ROOT, path.strip()))
                            for path in frame_paths
                        )
                    
                    if analysis_file_exists and frame_files_exist:
                        if old_status != 'completed':
                            self.stdout.write(
                                self.style.WARNING(f'  -> 상태 변경 필요: {old_status} -> completed')
                            )
                        else:
                            self.stdout.write(
                                self.style.SUCCESS(f'  -> 상태 정상: {old_status}')
                            )
                    else:
                        if old_status == 'completed':
                            self.stdout.write(
                                self.style.WARNING(f'  -> 상태 변경 필요: {old_status} -> failed (파일 없음)')
                            )
                        else:
                            self.stdout.write(
                                self.style.SUCCESS(f'  -> 상태 정상: {old_status}')
                            )
                else:
                    # 실제 동기화 수행
                    success = video_analysis_service.sync_video_status_with_files(video.id)
                    
                    if success:
                        video.refresh_from_db()
                        new_status = video.analysis_status
                        new_progress = video.analysis_progress
                        new_message = video.analysis_message
                        
                        if old_status != new_status:
                            self.stdout.write(
                                self.style.SUCCESS(f'  -> 상태 변경: {old_status} -> {new_status}')
                            )
                            synced_count += 1
                        else:
                            self.stdout.write(
                                self.style.SUCCESS(f'  -> 상태 유지: {new_status}')
                            )
                    else:
                        self.stdout.write(
                            self.style.ERROR(f'  -> 동기화 실패')
                        )
                        error_count += 1
                        
            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f'영상 {video.id} 처리 중 오류: {e}')
                )
                error_count += 1
        
        if options['dry_run']:
            self.stdout.write(
                self.style.SUCCESS(f'\nDry run 완료! 실제 변경사항은 없습니다.')
            )
        else:
            self.stdout.write(
                self.style.SUCCESS(f'\n동기화 완료!')
            )
            self.stdout.write(f'변경된 영상: {synced_count}개')
            self.stdout.write(f'오류 발생: {error_count}개')
