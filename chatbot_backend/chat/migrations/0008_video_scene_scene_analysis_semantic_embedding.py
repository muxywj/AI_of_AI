# Generated manually for LLM-based video search system

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('chat', '0007_video_enhanced_metadata_video_frame_captions_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='VideoScene',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('scene_id', models.IntegerField(help_text='장면 번호')),
                ('start_timestamp', models.FloatField(help_text='장면 시작 시간 (초)')),
                ('end_timestamp', models.FloatField(help_text='장면 종료 시간 (초)')),
                ('duration', models.FloatField(help_text='장면 지속 시간 (초)')),
                ('scene_description', models.TextField(blank=True, help_text='LLM이 생성한 장면 설명')),
                ('scene_type', models.CharField(blank=True, help_text='장면 유형 (indoor/outdoor/street 등)', max_length=50)),
                ('activity_context', models.TextField(blank=True, help_text='활동 맥락 설명')),
                ('dominant_objects', models.JSONField(blank=True, default=list, help_text='주요 객체들')),
                ('dominant_colors', models.JSONField(blank=True, default=list, help_text='주요 색상들')),
                ('weather_condition', models.CharField(blank=True, help_text='날씨 조건', max_length=20)),
                ('time_of_day', models.CharField(blank=True, help_text='시간대', max_length=20)),
                ('lighting_condition', models.CharField(blank=True, help_text='조명 조건', max_length=20)),
                ('semantic_embedding', models.JSONField(blank=True, default=list, help_text='의미적 벡터 임베딩')),
                ('search_keywords', models.JSONField(blank=True, default=list, help_text='검색 키워드')),
                ('semantic_tags', models.JSONField(blank=True, default=list, help_text='의미적 태그')),
                ('quality_score', models.FloatField(default=0.0, help_text='장면 품질 점수')),
                ('confidence_score', models.FloatField(default=0.0, help_text='분석 신뢰도')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('video', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='scenes', to='chat.video')),
            ],
            options={
                'verbose_name': '비디오 장면',
                'verbose_name_plural': '비디오 장면들',
                'ordering': ['video', 'scene_id'],
            },
        ),
        migrations.CreateModel(
            name='SceneAnalysis',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('detected_persons', models.JSONField(blank=True, default=list, help_text='감지된 사람들')),
                ('detected_objects', models.JSONField(blank=True, default=list, help_text='감지된 객체들')),
                ('person_count', models.IntegerField(default=0, help_text='사람 수')),
                ('object_count', models.IntegerField(default=0, help_text='객체 수')),
                ('activity_type', models.CharField(blank=True, help_text='활동 유형', max_length=50)),
                ('activity_intensity', models.CharField(blank=True, help_text='활동 강도', max_length=20)),
                ('movement_patterns', models.JSONField(blank=True, default=list, help_text='움직임 패턴')),
                ('emotional_tone', models.CharField(blank=True, help_text='감정적 톤', max_length=30)),
                ('atmosphere', models.CharField(blank=True, help_text='분위기', max_length=30)),
                ('brightness_level', models.FloatField(default=0.0, help_text='밝기 수준')),
                ('contrast_level', models.FloatField(default=0.0, help_text='대비 수준')),
                ('sharpness_level', models.FloatField(default=0.0, help_text='선명도 수준')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('scene', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, related_name='analysis', to='chat.videoscene')),
            ],
            options={
                'verbose_name': '장면 분석',
                'verbose_name_plural': '장면 분석들',
            },
        ),
        migrations.CreateModel(
            name='SemanticEmbedding',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('embedding_type', models.CharField(choices=[('scene', '장면'), ('object', '객체'), ('activity', '활동'), ('query', '쿼리')], max_length=20)),
                ('content_id', models.IntegerField(help_text='관련 콘텐츠 ID (장면 ID, 객체 ID 등)')),
                ('content_type', models.CharField(help_text='콘텐츠 타입', max_length=50)),
                ('embedding_vector', models.JSONField(help_text='벡터 임베딩 데이터')),
                ('embedding_dimension', models.IntegerField(help_text='임베딩 차원')),
                ('embedding_model', models.CharField(help_text='사용된 임베딩 모델', max_length=100)),
                ('original_text', models.TextField(blank=True, help_text='원본 텍스트')),
                ('language', models.CharField(default='ko', help_text='언어', max_length=10)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
            ],
            options={
                'verbose_name': '의미적 임베딩',
                'verbose_name_plural': '의미적 임베딩들',
            },
        ),
        migrations.AddIndex(
            model_name='semanticembedding',
            index=models.Index(fields=['embedding_type', 'content_type'], name='chat_semant_embeddi_6b8b8a_idx'),
        ),
        migrations.AlterUniqueTogether(
            name='videoscene',
            unique_together={('video', 'scene_id')},
        ),
        migrations.AlterUniqueTogether(
            name='semanticembedding',
            unique_together={('embedding_type', 'content_id', 'content_type')},
        ),
    ]
