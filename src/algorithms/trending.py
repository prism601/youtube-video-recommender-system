"""
Trending and popularity-based recommendations.
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class TrendingRecommender:
    """
    Recommends trending and popular videos.
    Uses time-weighted view counts and engagement metrics.
    """

    def __init__(self, time_decay_hours: float = 24.0, min_views: int = 1000):
        """
        Initialize trending recommender.

        Args:
            time_decay_hours: Hours for time decay calculation
            min_views: Minimum views to be considered trending
        """
        self.time_decay_hours = time_decay_hours
        self.min_views = min_views
        self.video_stats: Dict[str, Dict] = {}
        self.category_trending: Dict[str, List[Tuple[str, float]]] = defaultdict(list)

    def update_video_stats(self, video_id: str, stats: Dict):
        """
        Update statistics for a video.

        Args:
            video_id: Video identifier
            stats: Dictionary containing view_count, like_count, upload_date, etc.
        """
        self.video_stats[video_id] = stats

    def calculate_trending_score(self, video_id: str, current_time: Optional[datetime] = None) -> float:
        """
        Calculate trending score for a video.

        Score components:
        - View velocity (views per hour)
        - Engagement rate (likes, comments, shares)
        - Recency boost
        - Acceleration (increasing view rate)

        Args:
            video_id: Video identifier
            current_time: Reference time (defaults to now)

        Returns:
            Trending score (0-100)
        """
        if video_id not in self.video_stats:
            return 0.0

        stats = self.video_stats[video_id]
        current_time = current_time or datetime.now()

        view_count = stats.get('view_count', 0)
        if view_count < self.min_views:
            return 0.0

        # Calculate video age in hours
        upload_date = stats.get('upload_date')
        if not upload_date:
            return 0.0

        age_hours = (current_time - upload_date).total_seconds() / 3600
        if age_hours <= 0:
            age_hours = 0.1

        # View velocity (views per hour)
        view_velocity = view_count / age_hours

        # Engagement metrics
        like_count = stats.get('like_count', 0)
        comment_count = stats.get('comment_count', 0)
        share_count = stats.get('share_count', 0)

        engagement_rate = 0.0
        if view_count > 0:
            engagement_rate = (
                (like_count * 1.0 + comment_count * 2.0 + share_count * 3.0) /
                view_count
            ) * 100

        # Recency boost (exponential decay)
        recency_boost = np.exp(-age_hours / self.time_decay_hours)

        # Normalize view velocity (log scale)
        # Assuming 1000 views/hour is very high
        velocity_score = min(np.log10(view_velocity + 1) / np.log10(1000), 1.0)

        # Normalize engagement (cap at 10% engagement)
        engagement_score = min(engagement_rate / 10.0, 1.0)

        # Combined trending score
        trending_score = (
            velocity_score * 0.4 +
            engagement_score * 0.3 +
            recency_boost * 0.3
        ) * 100

        return trending_score

    def get_trending_videos(self, n_videos: int = 50,
                           category: Optional[str] = None,
                           time_window_hours: float = 48.0) -> List[Tuple[str, float]]:
        """
        Get trending videos.

        Args:
            n_videos: Number of videos to return
            category: Filter by category (optional)
            time_window_hours: Consider videos from last N hours

        Returns:
            List of (video_id, trending_score) tuples
        """
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(hours=time_window_hours)

        trending_videos = []

        for video_id, stats in self.video_stats.items():
            # Filter by category if specified
            if category and stats.get('category') != category:
                continue

            # Filter by time window
            upload_date = stats.get('upload_date')
            if upload_date and upload_date < cutoff_time:
                continue

            score = self.calculate_trending_score(video_id, current_time)
            if score > 0:
                trending_videos.append((video_id, score))

        # Sort by score and return top N
        trending_videos.sort(key=lambda x: x[1], reverse=True)
        return trending_videos[:n_videos]

    def get_popular_videos(self, n_videos: int = 50,
                          category: Optional[str] = None) -> List[Tuple[str, float]]:
        """
        Get popular videos based on overall statistics.

        Args:
            n_videos: Number of videos to return
            category: Filter by category

        Returns:
            List of (video_id, popularity_score) tuples
        """
        popular_videos = []

        for video_id, stats in self.video_stats.items():
            if category and stats.get('category') != category:
                continue

            view_count = stats.get('view_count', 0)
            like_count = stats.get('like_count', 0)
            comment_count = stats.get('comment_count', 0)

            # Popularity score combining views and engagement
            # Normalize assuming 10M views is maximum
            view_score = min(view_count / 10_000_000, 1.0)

            engagement = 0.0
            if view_count > 0:
                engagement = (like_count + comment_count * 2) / view_count
            engagement_score = min(engagement * 100, 1.0)

            popularity_score = (view_score * 0.6 + engagement_score * 0.4) * 100

            popular_videos.append((video_id, popularity_score))

        popular_videos.sort(key=lambda x: x[1], reverse=True)
        return popular_videos[:n_videos]

    def get_category_trending(self, categories: List[str],
                             n_per_category: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        """
        Get trending videos for each category.

        Args:
            categories: List of categories
            n_per_category: Number of videos per category

        Returns:
            Dictionary mapping category to list of (video_id, score) tuples
        """
        category_trending = {}

        for category in categories:
            trending = self.get_trending_videos(n_videos=n_per_category, category=category)
            category_trending[category] = trending

        return category_trending

    def recommend(self, user_id: str, n_recommendations: int = 50,
                  user_preferences: Optional[Dict] = None,
                  exclude_videos: Optional[List[str]] = None) -> List[Tuple[str, float]]:
        """
        Generate trending recommendations personalized for user preferences.

        Args:
            user_id: User identifier (for logging)
            n_recommendations: Number of recommendations
            user_preferences: User category preferences
            exclude_videos: Videos to exclude

        Returns:
            List of (video_id, score) tuples
        """
        exclude_videos = exclude_videos or []
        exclude_set = set(exclude_videos)

        if user_preferences and 'preferred_categories' in user_preferences:
            # Mix trending from preferred categories
            preferred_categories = user_preferences['preferred_categories']
            videos_per_category = max(1, n_recommendations // len(preferred_categories))

            all_trending = []
            for category in preferred_categories:
                category_trending = self.get_trending_videos(
                    n_videos=videos_per_category,
                    category=category
                )
                all_trending.extend(category_trending)

            # Add general trending to fill up
            general_trending = self.get_trending_videos(n_videos=n_recommendations)
            all_trending.extend(general_trending)

            # Deduplicate and filter
            seen = set()
            unique_trending = []
            for video_id, score in all_trending:
                if video_id not in seen and video_id not in exclude_set:
                    seen.add(video_id)
                    unique_trending.append((video_id, score))

            # Sort and return top N
            unique_trending.sort(key=lambda x: x[1], reverse=True)
            return unique_trending[:n_recommendations]
        else:
            # Return general trending
            trending = self.get_trending_videos(n_videos=n_recommendations * 2)
            return [(vid, score) for vid, score in trending if vid not in exclude_set][:n_recommendations]


class NewVideosRecommender:
    """Recommends newly uploaded videos from subscribed channels."""

    def __init__(self):
        """Initialize new videos recommender."""
        self.channel_subscribers: Dict[str, List[str]] = defaultdict(list)
        self.recent_uploads: Dict[str, List[Tuple[str, datetime]]] = defaultdict(list)

    def add_subscription(self, user_id: str, channel_id: str):
        """Add user subscription to channel."""
        if user_id not in self.channel_subscribers[channel_id]:
            self.channel_subscribers[channel_id].append(user_id)

    def add_upload(self, channel_id: str, video_id: str, upload_time: datetime):
        """Register new video upload."""
        self.recent_uploads[channel_id].append((video_id, upload_time))

    def recommend(self, user_id: str, subscriptions: List[str],
                  n_recommendations: int = 20,
                  hours_lookback: float = 168.0) -> List[Tuple[str, float]]:
        """
        Recommend new uploads from subscribed channels.

        Args:
            user_id: User identifier
            subscriptions: List of subscribed channel IDs
            n_recommendations: Number of recommendations
            hours_lookback: Look back N hours for new uploads

        Returns:
            List of (video_id, recency_score) tuples
        """
        cutoff_time = datetime.now() - timedelta(hours=hours_lookback)
        new_videos = []

        for channel_id in subscriptions:
            if channel_id in self.recent_uploads:
                for video_id, upload_time in self.recent_uploads[channel_id]:
                    if upload_time > cutoff_time:
                        # Recency score (newer = higher)
                        hours_ago = (datetime.now() - upload_time).total_seconds() / 3600
                        recency_score = max(0, 100 - hours_ago)
                        new_videos.append((video_id, recency_score))

        # Sort by recency and return top N
        new_videos.sort(key=lambda x: x[1], reverse=True)
        return new_videos[:n_recommendations]
