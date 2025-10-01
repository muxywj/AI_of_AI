from django.db import models
from django.utils import timezone


class Video(models.Model):
    title = models.CharField(max_length=200, blank=True, null=True)
    filename = models.CharField(max_length=255, default='')
    original_name = models.CharField(max_length=255, default='')
    file_path = models.CharField(max_length=500, default='')
    file_size = models.BigIntegerField(default=0)
    file = models.FileField(upload_to='uploads/', blank=True, null=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    analysis_status = models.CharField(max_length=50, default='pending')
    is_analyzed = models.BooleanField(default=False)
    duration = models.FloatField(default=0.0)
    enhanced_analysis = models.BooleanField(default=False)
    success_rate = models.FloatField(default=0.0)
    processing_time = models.IntegerField(default=0)
    analysis_type = models.CharField(max_length=50, default='basic')
    advanced_features_used = models.JSONField(default=dict, blank=True)
    scene_types = models.JSONField(default=list, blank=True)
    unique_objects = models.IntegerField(default=0)
    analysis_json_path = models.CharField(max_length=500, blank=True, null=True)


    def __str__(self):
        return self.title


class TrackPoint(models.Model):
    video = models.ForeignKey(Video, on_delete=models.CASCADE, related_name='tracks')
    track_id = models.IntegerField()
    frame_number = models.IntegerField()
    x1 = models.FloatField()
    y1 = models.FloatField()
    x2 = models.FloatField()
    y2 = models.FloatField()
    class_id = models.IntegerField()
    score = models.FloatField()


    class Meta:
        indexes = [models.Index(fields=['video','track_id','frame_number'])]


class Frame(models.Model):
    video = models.ForeignKey(Video, on_delete=models.CASCADE, related_name='frames')
    image_id = models.IntegerField()
    timestamp = models.FloatField(default=0.0)
    detected_objects = models.JSONField(default=list, blank=True)
    caption = models.TextField(blank=True)
    enhanced_caption = models.TextField(blank=True, null=True)
    final_caption = models.TextField(blank=True, null=True)
    comprehensive_features = models.JSONField(default=dict, blank=True, null=True)
    image = models.ImageField(upload_to='images/', blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        indexes = [models.Index(fields=['video', 'image_id'])]
        unique_together = ['video', 'image_id']


class Scene(models.Model):
    video = models.ForeignKey(Video, on_delete=models.CASCADE, related_name='scenes')
    start_time = models.FloatField()
    end_time = models.FloatField()
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        indexes = [models.Index(fields=['video', 'start_time'])]


class AnalysisResult(models.Model):
    """분석 결과 모델"""
    video = models.ForeignKey(Video, on_delete=models.CASCADE, related_name='analysis_results')
    frame_id = models.IntegerField()
    timestamp = models.FloatField(default=0.0)
    persons_detected = models.JSONField(default=list, blank=True)
    scene_analysis = models.JSONField(default=dict, blank=True)
    quality_metrics = models.JSONField(default=dict, blank=True)
    image_path = models.CharField(max_length=500, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        indexes = [models.Index(fields=['video', 'frame_id'])]
        unique_together = ['video', 'frame_id']


class PersonDetection(models.Model):
    """사람 탐지 결과를 저장하는 모델 - video_analyzer.py와 호환"""
    video = models.ForeignKey(Video, on_delete=models.CASCADE, related_name='person_detections')
    frame = models.ForeignKey(Frame, on_delete=models.CASCADE, related_name='person_detections', null=True, blank=True)
    person_id = models.IntegerField(default=0)
    track_id = models.IntegerField(null=True, blank=True)
    bbox_x1 = models.FloatField(default=0.0)
    bbox_y1 = models.FloatField(default=0.0)
    bbox_x2 = models.FloatField(default=0.0)
    bbox_y2 = models.FloatField(default=0.0)
    confidence = models.FloatField(default=0.0)
    gender_estimation = models.CharField(max_length=50, default='unknown')
    gender_confidence = models.FloatField(default=0.0)
    age_group = models.CharField(max_length=50, default='unknown')
    age_confidence = models.FloatField(default=0.0)
    upper_body_color = models.CharField(max_length=50, default='unknown')
    upper_color_confidence = models.FloatField(default=0.0)
    lower_body_color = models.CharField(max_length=50, default='unknown')
    lower_color_confidence = models.FloatField(default=0.0)
    posture = models.CharField(max_length=50, default='unknown')
    posture_confidence = models.FloatField(default=0.0)
    detailed_attributes = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"PersonDetection - Video {self.video.id}, Person {self.person_id}"
    
    class Meta:
        indexes = [models.Index(fields=['video', 'person_id'])]


class YOLOObjectDetection(models.Model):
    """YOLO 객체 감지 결과를 저장하는 모델"""
    video = models.ForeignKey(Video, on_delete=models.CASCADE, related_name='yolo_detections')
    frame = models.ForeignKey(Frame, on_delete=models.CASCADE, related_name='yolo_detections', null=True, blank=True)
    frame_number = models.IntegerField()
    timestamp = models.FloatField(default=0.0)
    class_name = models.CharField(max_length=50)
    confidence = models.FloatField()
    bbox_x1 = models.FloatField()
    bbox_y1 = models.FloatField()
    bbox_x2 = models.FloatField()
    bbox_y2 = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"YOLOObjectDetection - Video {self.video.id}, {self.class_name} (confidence: {self.confidence:.2f})"
    
    class Meta:
        indexes = [models.Index(fields=['video', 'frame_number', 'class_name'])]


class PersonSearchIndex(models.Model):
    """사람 검색을 위한 인덱스 모델"""
    person_detection = models.ForeignKey(PersonDetection, on_delete=models.CASCADE)
    clothing_color = models.CharField(max_length=50, blank=True, null=True)
    clothing_type = models.CharField(max_length=100, blank=True, null=True)
    posture = models.CharField(max_length=50, blank=True, null=True)
    gender = models.CharField(max_length=20, blank=True, null=True)
    age_group = models.CharField(max_length=20, blank=True, null=True)
    
    def __str__(self):
        return f"PersonSearchIndex - {self.person_detection}"
    
    class Meta:
        indexes = [
            models.Index(fields=['clothing_color']),
            models.Index(fields=['clothing_type']),
            models.Index(fields=['posture']),
            models.Index(fields=['gender']),
        ]