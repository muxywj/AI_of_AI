from rest_framework import serializers
from .models import Video, TrackPoint


class VideoSerializer(serializers.ModelSerializer):
    class Meta:
        model = Video
        fields = ('id','title','file','uploaded_at')


class TrackPointSerializer(serializers.ModelSerializer):
    class Meta:
        model = TrackPoint
        fields = '__all__'