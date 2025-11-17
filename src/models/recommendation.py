"""
Recommendation result models.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime


@dataclass
class RecommendationScore:
    """Score breakdown for a single recommendation."""
    video_id: str
    final_score: float

    # Individual algorithm scores
    collaborative_score: float = 0.0
    content_based_score: float = 0.0
    deep_learning_score: float = 0.0
    trending_score: float = 0.0
    popularity_score: float = 0.0

    # Adjustment factors
    diversity_boost: float = 0.0
    recency_boost: float = 0.0
    exploration_boost: float = 0.0

    # Metadata
    algorithm_source: str = "hybrid"  # Which algorithm contributed most
    confidence: float = 0.0

    def get_score_breakdown(self) -> Dict[str, float]:
        """Get detailed score breakdown."""
        return {
            'collaborative': self.collaborative_score,
            'content_based': self.content_based_score,
            'deep_learning': self.deep_learning_score,
            'trending': self.trending_score,
            'popularity': self.popularity_score,
            'diversity_boost': self.diversity_boost,
            'recency_boost': self.recency_boost,
            'exploration_boost': self.exploration_boost,
        }


@dataclass
class RecommendationItem:
    """A single recommended video with metadata."""
    video_id: str
    title: str
    channel_name: str
    thumbnail_url: Optional[str]
    duration: int
    score: RecommendationScore
    rank: int
    reason: Optional[str] = None  # Explanation for recommendation

    def to_dict(self) -> Dict:
        """Convert to dictionary for API response."""
        return {
            'video_id': self.video_id,
            'title': self.title,
            'channel_name': self.channel_name,
            'thumbnail_url': self.thumbnail_url,
            'duration': self.duration,
            'rank': self.rank,
            'score': self.score.final_score,
            'reason': self.reason,
        }


@dataclass
class RecommendationResult:
    """Complete recommendation result for a user."""
    user_id: str
    recommendations: List[RecommendationItem]
    generated_at: datetime
    algorithm_version: str = "1.0"

    # Metadata
    total_candidates: int = 0
    generation_time_ms: float = 0.0
    context: Optional[Dict] = None

    # A/B testing
    experiment_id: Optional[str] = None
    variant: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for API response."""
        return {
            'user_id': self.user_id,
            'recommendations': [item.to_dict() for item in self.recommendations],
            'generated_at': self.generated_at.isoformat(),
            'total_candidates': self.total_candidates,
            'generation_time_ms': self.generation_time_ms,
            'algorithm_version': self.algorithm_version,
        }

    def get_top_k(self, k: int = 20) -> List[RecommendationItem]:
        """Get top K recommendations."""
        return self.recommendations[:k]


@dataclass
class CandidateSet:
    """Set of candidate videos from a single algorithm."""
    algorithm_name: str
    candidates: List[tuple[str, float]]  # (video_id, score)
    generated_at: datetime
    metadata: Optional[Dict] = None

    def get_top_k(self, k: int = 100) -> List[tuple[str, float]]:
        """Get top K candidates sorted by score."""
        sorted_candidates = sorted(self.candidates, key=lambda x: x[1], reverse=True)
        return sorted_candidates[:k]


@dataclass
class RecommendationRequest:
    """Request for generating recommendations."""
    user_id: str
    num_recommendations: int = 20
    context: str = "homepage"  # homepage, search, watch_page, etc.
    exclude_videos: List[str] = field(default_factory=list)
    device_type: Optional[str] = None

    # Algorithm selection
    algorithms: List[str] = field(default_factory=lambda: ["collaborative", "content_based", "trending"])

    # Personalization settings
    enable_diversity: bool = True
    enable_exploration: bool = True
    exploration_rate: float = 0.15

    # Filters
    max_video_length: Optional[int] = None  # seconds
    preferred_categories: Optional[List[str]] = None
    preferred_languages: Optional[List[str]] = None
