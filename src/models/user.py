"""
User data models for the recommendation system.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime
from enum import Enum


class DeviceType(Enum):
    MOBILE = "mobile"
    DESKTOP = "desktop"
    TABLET = "tablet"
    TV = "tv"


@dataclass
class UserDemographics:
    """User demographic information."""
    age: Optional[int] = None
    gender: Optional[str] = None
    location: Optional[str] = None
    country: Optional[str] = None
    language: str = "en"
    timezone: Optional[str] = None


@dataclass
class UserPreferences:
    """User preference settings."""
    preferred_categories: List[str] = field(default_factory=list)
    preferred_languages: List[str] = field(default_factory=list)
    preferred_video_length: str = "medium"  # short, medium, long
    autoplay_enabled: bool = True
    hd_preferred: bool = True


@dataclass
class WatchHistoryItem:
    """Individual watch history entry."""
    video_id: str
    watched_at: datetime
    watch_duration: int  # seconds
    total_duration: int  # seconds
    watch_percentage: float
    completed: bool
    device_type: DeviceType

    @property
    def engagement_score(self) -> float:
        """Calculate engagement score based on watch percentage."""
        if self.watch_percentage >= 0.9:
            return 1.0
        elif self.watch_percentage >= 0.7:
            return 0.8
        elif self.watch_percentage >= 0.5:
            return 0.6
        elif self.watch_percentage >= 0.3:
            return 0.4
        else:
            return 0.2


@dataclass
class UserInteraction:
    """User interaction with a video."""
    user_id: str
    video_id: str
    interaction_type: str  # view, like, dislike, share, comment, subscribe
    timestamp: datetime
    context: Optional[Dict] = None


@dataclass
class UserProfile:
    """Complete user profile for recommendation system."""
    user_id: str
    username: str
    created_at: datetime
    demographics: UserDemographics
    preferences: UserPreferences
    subscriptions: List[str] = field(default_factory=list)  # channel_ids
    watch_history: List[WatchHistoryItem] = field(default_factory=list)
    liked_videos: List[str] = field(default_factory=list)
    disliked_videos: List[str] = field(default_factory=list)
    saved_playlists: List[str] = field(default_factory=list)
    blocked_channels: List[str] = field(default_factory=list)

    # Computed features
    total_watch_time: int = 0  # total seconds watched
    avg_watch_percentage: float = 0.0
    engagement_rate: float = 0.0
    active_hours: List[int] = field(default_factory=list)  # hours of day when active

    def update_computed_features(self):
        """Update computed features based on watch history."""
        if not self.watch_history:
            return

        # Total watch time
        self.total_watch_time = sum(item.watch_duration for item in self.watch_history)

        # Average watch percentage
        self.avg_watch_percentage = sum(
            item.watch_percentage for item in self.watch_history
        ) / len(self.watch_history)

        # Engagement rate (likes / total views)
        if self.watch_history:
            self.engagement_rate = len(self.liked_videos) / len(self.watch_history)

        # Active hours
        hour_counts = {}
        for item in self.watch_history:
            hour = item.watched_at.hour
            hour_counts[hour] = hour_counts.get(hour, 0) + 1

        # Top 3 active hours
        self.active_hours = sorted(hour_counts.keys(),
                                   key=lambda h: hour_counts[h],
                                   reverse=True)[:3]

    def get_category_preferences(self) -> Dict[str, float]:
        """Extract category preferences from watch history."""
        # This would be implemented with actual video category data
        # Placeholder implementation
        category_scores = {}
        for category in self.preferences.preferred_categories:
            category_scores[category] = 1.0
        return category_scores

    def get_recent_watch_history(self, limit: int = 50) -> List[WatchHistoryItem]:
        """Get most recent watch history."""
        sorted_history = sorted(self.watch_history,
                               key=lambda x: x.watched_at,
                               reverse=True)
        return sorted_history[:limit]


@dataclass
class UserSession:
    """Current user session context."""
    user_id: str
    session_id: str
    started_at: datetime
    device_type: DeviceType
    current_context: str  # homepage, search, watch_page, etc.
    videos_viewed_in_session: List[str] = field(default_factory=list)
    last_activity: Optional[datetime] = None
