from django.urls import path, include
from .views import ChatView, google_callback, kakao_callback, naver_callback, VideoUploadView, VideoChatView, VideoAnalysisView, VideoListView, FrameImageView, VideoSummaryView, VideoHighlightView
from .video_search_view import VideoSearchView
from .advanced_search_view import InterVideoSearchView, IntraVideoSearchView, TemporalAnalysisView

urlpatterns = [
    path('chat/<str:bot_name>/', ChatView.as_view(), name='chat'),
    path('api/auth/google/callback/', google_callback, name='google_callback'),
    path('api/auth/kakao/callback/', kakao_callback, name='kakao_callback'),
    path('api/auth/naver/callback/', naver_callback, name='naver_callback'),
    path('api/auth/', include('dj_rest_auth.urls')),
    path('api/auth/registration/', include('dj_rest_auth.registration.urls')),
    
    # 영상 관련 API
    path('api/video/upload/', VideoUploadView.as_view(), name='video_upload'),
    path('api/video/list/', VideoListView.as_view(), name='video_list'),
    path('api/video/<int:video_id>/analysis/', VideoAnalysisView.as_view(), name='video_analysis'),
    path('api/video/<int:video_id>/chat/', VideoChatView.as_view(), name='video_chat'),
    path('api/video/chat/sessions/', VideoChatView.as_view(), name='video_chat_sessions'),
    path('api/video/<int:video_id>/frame/<int:frame_number>/', FrameImageView.as_view(), name='frame_image'),
    
    # 영상 요약 및 하이라이트 API
    path('api/video/summary/', VideoSummaryView.as_view(), name='video_summary'),
    path('api/video/highlights/', VideoHighlightView.as_view(), name='video_highlights'),
    
    # 영상 검색 API
    path('api/video/search/', VideoSearchView.as_view(), name='video_search'),
    
    # 고급 검색 API
    path('api/video/search/inter/', InterVideoSearchView.as_view(), name='inter_video_search'),
    path('api/video/search/intra/', IntraVideoSearchView.as_view(), name='intra_video_search'),
    path('api/video/analysis/temporal/', TemporalAnalysisView.as_view(), name='temporal_analysis'),
]