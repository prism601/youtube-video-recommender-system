"""
Video data models for the recommendation system.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime
from enum import Enum


class VideoCategory(Enum):
    """Video categories."""
    MUSIC = "music"
    GAMING = "gaming"
    EDUCATION = "education"
    ENTERTAINMENT = "entertainment"
    NEWS = "news"
    SPORTS = "sports"
    TECHNOLOGY = "technology"
    COMEDY = "comedy"
    FILM = "film"
    SCIENCE = "science"
    COOKING = "cooking"
    TRAVEL = "travel"
    FASHION = "fashion"
    FITNESS = "fitness"
    HOWTO = "howto"
    OTHER = "other"


class VideoLength(Enum):
    """Video length categories."""
    SHORT = "short"  # < 5 minutes
    MEDIUM = "medium"  # 5-20 minutes
    LONG = "long"  # > 20 minutes


@dataclass
class VideoStatistics:
    """Video engagement statistics."""
    view_count: int = 0
    like_count: int = 0
    dislike_count: int = 0
    comment_count: int = 0
    share_count: int = 0
    average_watch_time: float = 0.0  # seconds
    click_through_rate: float = 0.0

    @property
    def like_ratio(self) -> float:
        """Calculate like/dislike ratio."""
        total = self.like_count + self.dislike_count
        if total == 0:
            return 0.0
        return self.like_count / total

    @property
    def engagement_score(self) -> float:
        """Calculate overall engagement score."""
        # Weighted combination of metrics
        if self.view_count == 0:
            return 0.0

        like_score = self.like_count / max(self.view_count, 1)
        comment_score = self.comment_count / max(self.view_count, 1)
        share_score = self.share_count / max(self.view_count, 1)

        # Normalize and combine
        engagement = (
            like_score * 0.5 +
            comment_score * 0.3 +
            share_score * 0.2
        )
        return min(engagement * 100, 100.0)  # Scale to 0-100


@dataclass
class VideoMetadata:
    """Video metadata and content information."""
    video_id: str
    title: str
    description: str
    channel_id: str
    channel_name: str
    category: VideoCategory
    tags: List[str] = field(default_factory=list)
    duration: int = 0  # seconds
    upload_date: Optional[datetime] = None
    language: str = "en"
    thumbnail_url: Optional[str] = None
    video_url: Optional[str] = None

    # Quality indicators
    is_hd: bool = False
    is_4k: bool = False
    has_captions: bool = False

    @property
    def video_length_category(self) -> VideoLength:
        """Categorize video by length."""
        if self.duration < 300:  # 5 minutes
            return VideoLength.SHORT
        elif self.duration < 1200:  # 20 minutes
            return VideoLength.MEDIUM
        else:
            return VideoLength.LONG

    @property
    def age_days(self) -> Optional[int]:
        """Get video age in days."""
        if not self.upload_date:
            return None
        delta = datetime.now() - self.upload_date
        return delta.days


@dataclass
class VideoFeatures:
    """Extracted features for recommendation algorithms."""
    video_id: str

    # Content features
    content_embedding: Optional[List[float]] = None  # From title/description
    category_vector: Optional[List[float]] = None
    tag_vector: Optional[List[float]] = None

    # Popularity features
    popularity_score: float = 0.0
    trending_score: float = 0.0
    recency_score: float = 0.0

    # Quality features
    quality_score: float = 0.0

    # Collaborative features
    cf_embedding: Optional[List[float]] = None  # From matrix factorization

    def calculate_popularity_score(self, stats: VideoStatistics, max_views: int = 1000000) -> float:
        """Calculate normalized popularity score."""
        view_score = min(stats.view_count / max_views, 1.0)
        engagement_score = stats.engagement_score / 100.0

        # Weighted combination
        self.popularity_score = view_score * 0.6 + engagement_score * 0.4
        return self.popularity_score

    def calculate_recency_score(self, metadata: VideoMetadata, decay_days: int = 30) -> float:
        """Calculate recency score with exponential decay."""
        if not metadata.age_days:
            return 0.0

        # Exponential decay: score = e^(-age/decay_days)
        import math
        self.recency_score = math.exp(-metadata.age_days / decay_days)
        return self.recency_score

    def calculate_trending_score(self,
                                 stats: VideoStatistics,
                                 metadata: VideoMetadata,
                                 view_velocity: float = 0.0) -> float:
        """
        Calculate trending score based on recent growth.
        view_velocity: views per hour in last 24h
        """
        if not metadata.age_days or metadata.age_days == 0:
            age_factor = 1.0
        else:
            # Newer videos get higher trending potential
            age_factor = max(0.1, 1.0 - (metadata.age_days / 7.0))

        # Normalize view velocity (assuming 10k views/hour is very high)
        velocity_score = min(view_velocity / 10000.0, 1.0)

        engagement = stats.engagement_score / 100.0

        self.trending_score = (
            age_factor * 0.4 +
            velocity_score * 0.4 +
            engagement * 0.2
        )
        return self.trending_score


@dataclass
class Video:
    """Complete video object combining all information."""
    metadata: VideoMetadata
    statistics: VideoStatistics
    features: Optional[VideoFeatures] = None

    def __post_init__(self):
        """Initialize features if not provided."""
        if self.features is None:
            self.features = VideoFeatures(video_id=self.metadata.video_id)
            self.features.calculate_popularity_score(self.statistics)
            self.features.calculate_recency_score(self.metadata)

    def to_dict(self) -> Dict:
        """Convert to dictionary for API responses."""
        return {
            'video_id': self.metadata.video_id,
            'title': self.metadata.title,
            'channel_name': self.metadata.channel_name,
            'thumbnail_url': self.metadata.thumbnail_url,
            'duration': self.metadata.duration,
            'view_count': self.statistics.view_count,
            'upload_date': self.metadata.upload_date.isoformat() if self.metadata.upload_date else None,
            'category': self.metadata.category.value,
        }
