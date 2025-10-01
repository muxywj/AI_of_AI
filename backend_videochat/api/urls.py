from django.urls import path
from . import views
from . import views_basic_analysis
from . import clip_views


urlpatterns = [
    path('videos/', views.VideoListView.as_view()),
    path('videos/upload/', views.VideoUploadView.as_view()),
    path('videos/<int:pk>/', views.get_video_details),
    path('videos/<int:pk>/start-analysis/', views.start_analysis),
    path('videos/<int:pk>/analysis/start/', views.start_analysis),
    path('videos/<int:pk>/basic-analysis/', views_basic_analysis.BasicAnalyzeVideoView.as_view()),
    path('videos/<int:pk>/analysis/status/', views.get_analysis_status),
    path('videos/<int:pk>/chat/', views.chat_with_video),
    path('videos/<int:pk>/delete/', views.delete_video),
    path('videos/batch-delete/', views.batch_delete_videos),
    path('videos/cleanup/', views.cleanup_storage),
    path('videos/<int:pk>/tracks/', views.get_tracks),
    path('videos/<int:pk>/frames/<int:frame_number>/', views.get_frame_image), # 프레임 이미지
    path('videos/<int:pk>/frames/<int:frame_number>/bbox/', views.get_frame_image_bbox), # 프레임 이미지 (bbox)
    path('videos/<int:pk>/frames/<int:frame_number>/file/', views.get_frame_image_file), # 프레임 이미지 파일
    path('videos/<int:pk>/search/', views.search_frames), # 프레임 검색
    path('videos/<int:pk>/object-search/', views.ObjectSearchView.as_view()), # 객체 탐지 검색
    # path('videos/<int:pk>/scene/<int:frame_index>/', views.ScenePreviewView.as_view()), # 장면 미리보기
    path('videos/<int:video_id>/summary/', views.VideoSummaryView.as_view()), # 비디오 요약 및 하이라이트
    path('clip/<int:video_id>/<str:timestamp>/', clip_views.ClipPreviewView.as_view(), name='clip_preview'), # 클립 미리보기
    path('test/', views.APIStatusView.as_view()),
]

## path("api/", include("api.urls")),