"""
Data models for YouTube recommendation system.
"""
from .user import (
    UserProfile,
    UserDemographics,
    UserPreferences,
    WatchHistoryItem,
    UserInteraction,
    UserSession,
    DeviceType,
)
from .video import (
    Video,
    VideoMetadata,
    VideoStatistics,
    VideoFeatures,
    VideoCategory,
    VideoLength,
)
from .recommendation import (
    RecommendationResult,
    RecommendationItem,
    RecommendationScore,
    RecommendationRequest,
    CandidateSet,
)

__all__ = [
    # User models
    'UserProfile',
    'UserDemographics',
    'UserPreferences',
    'WatchHistoryItem',
    'UserInteraction',
    'UserSession',
    'DeviceType',
    # Video models
    'Video',
    'VideoMetadata',
    'VideoStatistics',
    'VideoFeatures',
    'VideoCategory',
    'VideoLength',
    # Recommendation models
    'RecommendationResult',
    'RecommendationItem',
    'RecommendationScore',
    'RecommendationRequest',
    'CandidateSet',
]
