"""
Hybrid recommender combining multiple recommendation strategies.
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import random
import logging

from .collaborative_filtering import MatrixFactorization, ItemBasedCF
from .content_based import ContentBasedRecommender
from .trending import TrendingRecommender, NewVideosRecommender

logger = logging.getLogger(__name__)


class HybridRecommender:
    """
    Hybrid recommendation system combining multiple algorithms.

    Combines:
    - Collaborative Filtering (Matrix Factorization + Item-based)
    - Content-Based Filtering
    - Trending/Popular videos
    - New uploads from subscriptions
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize hybrid recommender.

        Args:
            weights: Algorithm weights for score fusion
        """
        # Default weights
        self.weights = weights or {
            'collaborative_mf': 0.30,
            'collaborative_item': 0.20,
            'content_based': 0.25,
            'trending': 0.15,
            'new_uploads': 0.10,
        }

        # Initialize individual recommenders
        self.mf_recommender = MatrixFactorization(n_factors=50, n_iterations=20)
        self.item_cf_recommender = ItemBasedCF(k_neighbors=50)
        self.content_recommender = ContentBasedRecommender()
        self.trending_recommender = TrendingRecommender()
        self.new_videos_recommender = NewVideosRecommender()

        self.is_trained = False

    def train(self, interactions: List[Tuple[str, str, float]],
              videos: List[Dict],
              video_stats: Optional[Dict[str, Dict]] = None):
        """
        Train all recommendation models.

        Args:
            interactions: List of (user_id, video_id, rating) tuples
            videos: List of video metadata dictionaries
            video_stats: Video statistics for trending calculation
        """
        logger.info("Training hybrid recommender system")

        # Train collaborative filtering
        logger.info("Training matrix factorization...")
        self.mf_recommender.fit(interactions)

        logger.info("Training item-based CF...")
        self.item_cf_recommender.fit(interactions)

        # Train content-based
        logger.info("Training content-based recommender...")
        self.content_recommender.fit(videos, interactions)

        # Update trending statistics
        if video_stats:
            logger.info("Updating trending statistics...")
            for video_id, stats in video_stats.items():
                self.trending_recommender.update_video_stats(video_id, stats)

        self.is_trained = True
        logger.info("Hybrid recommender training complete")

    def _merge_candidates(self, candidate_sets: Dict[str, List[Tuple[str, float]]]) -> Dict[str, float]:
        """
        Merge candidate sets from multiple algorithms.

        Args:
            candidate_sets: Dictionary mapping algorithm name to list of (video_id, score) tuples

        Returns:
            Dictionary mapping video_id to weighted combined score
        """
        # Normalize scores for each algorithm
        normalized_scores: Dict[str, Dict[str, float]] = {}

        for algo_name, candidates in candidate_sets.items():
            if not candidates:
                normalized_scores[algo_name] = {}
                continue

            scores = [score for _, score in candidates]
            if len(scores) > 1:
                min_score = min(scores)
                max_score = max(scores)
                score_range = max_score - min_score

                if score_range > 0:
                    normalized = {
                        video_id: (score - min_score) / score_range
                        for video_id, score in candidates
                    }
                else:
                    normalized = {video_id: 1.0 for video_id, _ in candidates}
            else:
                normalized = {candidates[0][0]: 1.0}

            normalized_scores[algo_name] = normalized

        # Weighted combination
        combined_scores: Dict[str, float] = defaultdict(float)

        for algo_name, video_scores in normalized_scores.items():
            weight = self.weights.get(algo_name, 0.0)
            for video_id, score in video_scores.items():
                combined_scores[video_id] += weight * score

        return combined_scores

    def _apply_diversity(self, recommendations: List[Tuple[str, float]],
                        video_metadata: Dict[str, Dict],
                        diversity_factor: float = 0.3) -> List[Tuple[str, float]]:
        """
        Apply diversity to recommendations to avoid over-concentration.

        Args:
            recommendations: List of (video_id, score) tuples
            video_metadata: Video metadata for diversity calculation
            diversity_factor: Strength of diversity boost (0-1)

        Returns:
            Re-ranked recommendations with diversity
        """
        if diversity_factor == 0:
            return recommendations

        diverse_recs = []
        seen_categories = set()
        seen_channels = set()

        category_count = defaultdict(int)
        channel_count = defaultdict(int)

        for video_id, score in recommendations:
            metadata = video_metadata.get(video_id, {})
            category = metadata.get('category', 'other')
            channel = metadata.get('channel_id', '')

            # Apply diversity penalty
            category_penalty = category_count[category] * 0.1
            channel_penalty = channel_count[channel] * 0.15

            diversity_penalty = (category_penalty + channel_penalty) * diversity_factor
            adjusted_score = score * (1 - diversity_penalty)

            diverse_recs.append((video_id, adjusted_score))

            category_count[category] += 1
            channel_count[channel] += 1

        # Re-sort by adjusted scores
        diverse_recs.sort(key=lambda x: x[1], reverse=True)
        return diverse_recs

    def _apply_exploration(self, recommendations: List[Tuple[str, float]],
                          all_videos: List[str],
                          exploration_rate: float = 0.15) -> List[Tuple[str, float]]:
        """
        Add random exploration to recommendations.

        Args:
            recommendations: Current recommendations
            all_videos: All available videos
            exploration_rate: Percentage of recommendations to replace with random

        Returns:
            Recommendations with exploration
        """
        if exploration_rate == 0 or not all_videos:
            return recommendations

        n_explore = int(len(recommendations) * exploration_rate)
        n_keep = len(recommendations) - n_explore

        # Keep top recommendations
        final_recs = recommendations[:n_keep]

        # Add random videos
        recommended_ids = {video_id for video_id, _ in recommendations}
        available_for_exploration = [vid for vid in all_videos if vid not in recommended_ids]

        if available_for_exploration:
            random_videos = random.sample(
                available_for_exploration,
                min(n_explore, len(available_for_exploration))
            )
            # Assign decreasing scores to random videos
            for i, video_id in enumerate(random_videos):
                score = final_recs[-1][1] * (1 - (i / len(random_videos)) * 0.5) if final_recs else 0.5
                final_recs.append((video_id, score))

        return final_recs

    def recommend(self,
                  user_id: str,
                  n_recommendations: int = 20,
                  user_profile: Optional[Dict] = None,
                  exclude_videos: Optional[List[str]] = None,
                  enable_diversity: bool = True,
                  enable_exploration: bool = True,
                  exploration_rate: float = 0.15) -> List[Tuple[str, float]]:
        """
        Generate hybrid recommendations for a user.

        Args:
            user_id: User identifier
            n_recommendations: Number of recommendations to generate
            user_profile: User profile with preferences and subscriptions
            exclude_videos: Videos to exclude
            enable_diversity: Apply diversity boosting
            enable_exploration: Include random exploration
            exploration_rate: Rate of exploration (0-1)

        Returns:
            List of (video_id, score) tuples
        """
        if not self.is_trained:
            logger.warning("Recommender not trained, returning empty recommendations")
            return []

        exclude_videos = exclude_videos or []
        user_profile = user_profile or {}

        # Number of candidates to fetch from each algorithm
        n_candidates = n_recommendations * 5

        # Collect candidates from each algorithm
        candidate_sets = {}

        # Collaborative Filtering - Matrix Factorization
        try:
            mf_candidates = self.mf_recommender.recommend(
                user_id, n_candidates, exclude_videos
            )
            candidate_sets['collaborative_mf'] = mf_candidates
            logger.debug(f"MF candidates: {len(mf_candidates)}")
        except Exception as e:
            logger.warning(f"MF recommendation failed: {e}")
            candidate_sets['collaborative_mf'] = []

        # Collaborative Filtering - Item-based
        try:
            item_candidates = self.item_cf_recommender.recommend(
                user_id, n_candidates, exclude_videos
            )
            candidate_sets['collaborative_item'] = item_candidates
            logger.debug(f"Item-CF candidates: {len(item_candidates)}")
        except Exception as e:
            logger.warning(f"Item-CF recommendation failed: {e}")
            candidate_sets['collaborative_item'] = []

        # Content-Based
        try:
            content_candidates = self.content_recommender.recommend(
                user_id, n_candidates, exclude_videos
            )
            candidate_sets['content_based'] = content_candidates
            logger.debug(f"Content-based candidates: {len(content_candidates)}")
        except Exception as e:
            logger.warning(f"Content-based recommendation failed: {e}")
            candidate_sets['content_based'] = []

        # Trending
        try:
            trending_candidates = self.trending_recommender.recommend(
                user_id, n_candidates // 2, user_profile, exclude_videos
            )
            candidate_sets['trending'] = trending_candidates
            logger.debug(f"Trending candidates: {len(trending_candidates)}")
        except Exception as e:
            logger.warning(f"Trending recommendation failed: {e}")
            candidate_sets['trending'] = []

        # New uploads from subscriptions
        try:
            if 'subscriptions' in user_profile:
                new_upload_candidates = self.new_videos_recommender.recommend(
                    user_id, user_profile['subscriptions'], n_candidates // 4
                )
                candidate_sets['new_uploads'] = new_upload_candidates
                logger.debug(f"New uploads candidates: {len(new_upload_candidates)}")
        except Exception as e:
            logger.warning(f"New uploads recommendation failed: {e}")
            candidate_sets['new_uploads'] = []

        # Merge candidates
        combined_scores = self._merge_candidates(candidate_sets)

        # Convert to sorted list
        recommendations = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

        # Apply diversity
        if enable_diversity and 'video_metadata' in user_profile:
            recommendations = self._apply_diversity(
                recommendations,
                user_profile['video_metadata'],
                diversity_factor=0.3
            )

        # Take top N before exploration
        recommendations = recommendations[:n_recommendations]

        # Apply exploration
        if enable_exploration and 'all_videos' in user_profile:
            recommendations = self._apply_exploration(
                recommendations,
                user_profile['all_videos'],
                exploration_rate
            )

        logger.info(f"Generated {len(recommendations)} recommendations for user {user_id}")
        return recommendations[:n_recommendations]

    def get_similar_videos(self, video_id: str, n_similar: int = 20) -> List[Tuple[str, float]]:
        """
        Find videos similar to a given video.

        Args:
            video_id: Reference video ID
            n_similar: Number of similar videos

        Returns:
            List of (video_id, similarity_score) tuples
        """
        return self.content_recommender.get_similar_videos(video_id, n_similar)

    def update_weights(self, new_weights: Dict[str, float]):
        """Update algorithm weights for A/B testing."""
        self.weights.update(new_weights)
        logger.info(f"Updated weights: {self.weights}")
