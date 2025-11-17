"""
Recommendation service API implementation.
"""
from typing import List, Dict, Optional
from datetime import datetime
import logging
import time

from ..models import (
    UserProfile,
    Video,
    RecommendationResult,
    RecommendationItem,
    RecommendationScore,
    RecommendationRequest,
)
from ..algorithms import HybridRecommender

logger = logging.getLogger(__name__)


class RecommendationService:
    """
    Main service for generating video recommendations.
    """

    def __init__(self, recommender: Optional[HybridRecommender] = None):
        """
        Initialize recommendation service.

        Args:
            recommender: Hybrid recommender instance
        """
        self.recommender = recommender or HybridRecommender()
        self.cache: Dict[str, RecommendationResult] = {}
        self.cache_ttl = 300  # 5 minutes

    def get_recommendations(self,
                           request: RecommendationRequest,
                           user_profile: Optional[UserProfile] = None,
                           video_metadata: Optional[Dict[str, Dict]] = None) -> RecommendationResult:
        """
        Generate recommendations for a user.

        Args:
            request: Recommendation request
            user_profile: User profile (optional)
            video_metadata: Video metadata dictionary

        Returns:
            RecommendationResult with ranked recommendations
        """
        start_time = time.time()

        logger.info(f"Generating recommendations for user {request.user_id}")

        # Check cache
        cache_key = f"{request.user_id}:{request.context}"
        if cache_key in self.cache:
            cached_result = self.cache[cache_key]
            cache_age = (datetime.now() - cached_result.generated_at).total_seconds()
            if cache_age < self.cache_ttl:
                logger.info(f"Returning cached recommendations (age: {cache_age:.1f}s)")
                return cached_result

        # Build user profile dictionary for recommender
        user_profile_dict = {}
        if user_profile:
            user_profile_dict['preferred_categories'] = user_profile.preferences.preferred_categories
            user_profile_dict['subscriptions'] = user_profile.subscriptions

        if video_metadata:
            user_profile_dict['video_metadata'] = video_metadata

        # Generate recommendations
        raw_recommendations = self.recommender.recommend(
            user_id=request.user_id,
            n_recommendations=request.num_recommendations,
            user_profile=user_profile_dict,
            exclude_videos=request.exclude_videos,
            enable_diversity=request.enable_diversity,
            enable_exploration=request.enable_exploration,
            exploration_rate=request.exploration_rate,
        )

        # Convert to RecommendationItem objects
        recommendation_items = []
        for rank, (video_id, score) in enumerate(raw_recommendations, start=1):
            # Get video metadata
            video_meta = video_metadata.get(video_id, {}) if video_metadata else {}

            # Create score object
            rec_score = RecommendationScore(
                video_id=video_id,
                final_score=score,
                algorithm_source="hybrid"
            )

            # Create recommendation item
            item = RecommendationItem(
                video_id=video_id,
                title=video_meta.get('title', f'Video {video_id}'),
                channel_name=video_meta.get('channel_name', 'Unknown'),
                thumbnail_url=video_meta.get('thumbnail_url'),
                duration=video_meta.get('duration', 0),
                score=rec_score,
                rank=rank,
                reason=self._generate_reason(request, user_profile, video_meta)
            )
            recommendation_items.append(item)

        # Create result
        generation_time_ms = (time.time() - start_time) * 1000
        result = RecommendationResult(
            user_id=request.user_id,
            recommendations=recommendation_items,
            generated_at=datetime.now(),
            total_candidates=len(raw_recommendations),
            generation_time_ms=generation_time_ms,
            context={'request_context': request.context}
        )

        # Cache result
        self.cache[cache_key] = result

        logger.info(f"Generated {len(recommendation_items)} recommendations in {generation_time_ms:.1f}ms")
        return result

    def get_trending_recommendations(self,
                                    n_recommendations: int = 50,
                                    category: Optional[str] = None) -> List[Dict]:
        """
        Get trending videos.

        Args:
            n_recommendations: Number of trending videos
            category: Filter by category

        Returns:
            List of trending video dictionaries
        """
        trending_videos = self.recommender.trending_recommender.get_trending_videos(
            n_videos=n_recommendations,
            category=category
        )

        return [
            {'video_id': video_id, 'trending_score': score}
            for video_id, score in trending_videos
        ]

    def get_similar_videos(self, video_id: str, n_similar: int = 20) -> List[Dict]:
        """
        Get videos similar to a given video.

        Args:
            video_id: Reference video ID
            n_similar: Number of similar videos

        Returns:
            List of similar video dictionaries
        """
        similar_videos = self.recommender.get_similar_videos(video_id, n_similar)

        return [
            {'video_id': vid, 'similarity_score': score}
            for vid, score in similar_videos
        ]

    def log_interaction(self, user_id: str, video_id: str, interaction_type: str):
        """
        Log user interaction with a video.

        Args:
            user_id: User identifier
            video_id: Video identifier
            interaction_type: Type of interaction (view, click, like, etc.)
        """
        logger.info(f"User {user_id} {interaction_type} video {video_id}")

        # Invalidate cache for this user
        cache_keys_to_remove = [key for key in self.cache.keys() if key.startswith(f"{user_id}:")]
        for key in cache_keys_to_remove:
            del self.cache[key]

        # In a real system, this would:
        # 1. Store interaction in database
        # 2. Update real-time user profile
        # 3. Trigger online learning update
        # 4. Send to analytics pipeline

    def train_models(self,
                    interactions: List[tuple],
                    videos: List[Dict],
                    video_stats: Optional[Dict] = None):
        """
        Train recommendation models.

        Args:
            interactions: User-video interactions
            videos: Video metadata
            video_stats: Video statistics
        """
        logger.info("Training recommendation models...")
        self.recommender.train(interactions, videos, video_stats)
        self.cache.clear()  # Clear cache after training
        logger.info("Model training complete")

    def _generate_reason(self,
                        request: RecommendationRequest,
                        user_profile: Optional[UserProfile],
                        video_meta: Dict) -> Optional[str]:
        """
        Generate explanation for why video was recommended.

        Args:
            request: Recommendation request
            user_profile: User profile
            video_meta: Video metadata

        Returns:
            Human-readable explanation
        """
        # Simple rule-based explanations
        if not user_profile:
            return "Popular video"

        category = video_meta.get('category', '')
        if category in user_profile.preferences.preferred_categories:
            return f"Because you watch {category} videos"

        channel = video_meta.get('channel_id', '')
        if channel in user_profile.subscriptions:
            return f"New from {video_meta.get('channel_name', 'subscribed channel')}"

        return "Recommended for you"

    def clear_cache(self):
        """Clear recommendation cache."""
        self.cache.clear()
        logger.info("Recommendation cache cleared")

    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            'cache_size': len(self.cache),
            'cache_ttl_seconds': self.cache_ttl,
        }
