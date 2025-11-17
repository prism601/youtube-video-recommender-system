"""
Content-Based Filtering for video recommendations.
"""
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import Counter, defaultdict
import re
import logging

logger = logging.getLogger(__name__)


class TFIDFVectorizer:
    """Simple TF-IDF vectorizer for text processing."""

    def __init__(self, max_features: int = 1000, min_df: int = 2):
        """
        Initialize TF-IDF vectorizer.

        Args:
            max_features: Maximum number of features
            min_df: Minimum document frequency
        """
        self.max_features = max_features
        self.min_df = min_df
        self.vocabulary: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.n_documents = 0

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        tokens = text.split()
        return [t for t in tokens if len(t) > 2]

    def fit(self, documents: List[str]):
        """Build vocabulary and IDF from documents."""
        logger.info(f"Fitting TF-IDF on {len(documents)} documents")

        self.n_documents = len(documents)

        # Count document frequency
        df: Dict[str, int] = defaultdict(int)
        for doc in documents:
            tokens = set(self._tokenize(doc))
            for token in tokens:
                df[token] += 1

        # Filter by min_df and select top features
        valid_tokens = {token for token, freq in df.items() if freq >= self.min_df}
        top_tokens = sorted(valid_tokens, key=lambda t: df[t], reverse=True)[:self.max_features]

        # Build vocabulary
        self.vocabulary = {token: idx for idx, token in enumerate(top_tokens)}

        # Calculate IDF
        for token in self.vocabulary:
            self.idf[token] = np.log(self.n_documents / (df[token] + 1))

        logger.info(f"Vocabulary size: {len(self.vocabulary)}")

    def transform(self, documents: List[str]) -> np.ndarray:
        """Transform documents to TF-IDF vectors."""
        vectors = np.zeros((len(documents), len(self.vocabulary)))

        for doc_idx, doc in enumerate(documents):
            tokens = self._tokenize(doc)
            tf = Counter(tokens)

            for token, count in tf.items():
                if token in self.vocabulary:
                    token_idx = self.vocabulary[token]
                    # TF * IDF
                    vectors[doc_idx, token_idx] = count * self.idf[token]

        # L2 normalization
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        vectors = vectors / norms

        return vectors


class ContentBasedRecommender:
    """
    Content-Based Recommender using video metadata.
    Recommends videos similar to those the user has watched.
    """

    def __init__(self, category_weight: float = 0.3, text_weight: float = 0.5,
                 tag_weight: float = 0.2):
        """
        Initialize content-based recommender.

        Args:
            category_weight: Weight for category similarity
            text_weight: Weight for text similarity
            tag_weight: Weight for tag similarity
        """
        self.category_weight = category_weight
        self.text_weight = text_weight
        self.tag_weight = tag_weight

        self.tfidf = TFIDFVectorizer()
        self.video_vectors: Dict[str, np.ndarray] = {}
        self.video_categories: Dict[str, str] = {}
        self.video_tags: Dict[str, Set[str]] = {}
        self.user_profiles: Dict[str, np.ndarray] = {}

    def fit(self, videos: List[Dict], user_interactions: List[Tuple[str, str, float]]):
        """
        Build content-based model.

        Args:
            videos: List of video dictionaries with metadata
            user_interactions: List of (user_id, video_id, rating) tuples
        """
        logger.info(f"Training content-based recommender with {len(videos)} videos")

        # Extract video metadata
        video_texts = []
        video_ids = []

        for video in videos:
            video_id = video['video_id']
            video_ids.append(video_id)

            # Combine title and description
            text = f"{video.get('title', '')} {video.get('description', '')}"
            video_texts.append(text)

            # Store category and tags
            self.video_categories[video_id] = video.get('category', 'other')
            self.video_tags[video_id] = set(video.get('tags', []))

        # Build TF-IDF vectors
        self.tfidf.fit(video_texts)
        text_vectors = self.tfidf.transform(video_texts)

        # Store video vectors
        for video_id, text_vector in zip(video_ids, text_vectors):
            self.video_vectors[video_id] = text_vector

        # Build user profiles
        self._build_user_profiles(user_interactions)

        logger.info(f"Built profiles for {len(self.user_profiles)} users")

    def _build_user_profiles(self, interactions: List[Tuple[str, str, float]]):
        """Build user preference profiles from interaction history."""
        user_videos: Dict[str, List[Tuple[str, float]]] = defaultdict(list)

        for user_id, video_id, rating in interactions:
            if video_id in self.video_vectors:
                user_videos[user_id].append((video_id, rating))

        # Create user profile as weighted average of watched videos
        for user_id, videos in user_videos.items():
            weighted_sum = np.zeros_like(next(iter(self.video_vectors.values())))
            total_weight = 0.0

            for video_id, rating in videos:
                if video_id in self.video_vectors:
                    weighted_sum += self.video_vectors[video_id] * rating
                    total_weight += rating

            if total_weight > 0:
                self.user_profiles[user_id] = weighted_sum / total_weight

    def _calculate_similarity(self, video_id: str, user_profile: np.ndarray,
                             user_categories: Counter, user_tags: Set[str]) -> float:
        """Calculate similarity between video and user profile."""
        if video_id not in self.video_vectors:
            return 0.0

        # Text similarity (cosine)
        text_sim = np.dot(self.video_vectors[video_id], user_profile)

        # Category similarity
        video_category = self.video_categories.get(video_id, 'other')
        category_sim = user_categories.get(video_category, 0) / max(sum(user_categories.values()), 1)

        # Tag similarity (Jaccard)
        video_tags = self.video_tags.get(video_id, set())
        if user_tags and video_tags:
            tag_sim = len(user_tags & video_tags) / len(user_tags | video_tags)
        else:
            tag_sim = 0.0

        # Weighted combination
        total_similarity = (
            self.text_weight * text_sim +
            self.category_weight * category_sim +
            self.tag_weight * tag_sim
        )

        return total_similarity

    def recommend(self, user_id: str, n_recommendations: int = 100,
                  exclude_videos: Optional[List[str]] = None) -> List[Tuple[str, float]]:
        """
        Generate content-based recommendations.

        Args:
            user_id: User identifier
            n_recommendations: Number of recommendations
            exclude_videos: Videos to exclude

        Returns:
            List of (video_id, score) tuples
        """
        if user_id not in self.user_profiles:
            logger.warning(f"User {user_id} has no profile")
            return []

        exclude_videos = exclude_videos or []
        exclude_set = set(exclude_videos)

        # Get user preferences
        user_profile = self.user_profiles[user_id]

        # Extract user category and tag preferences (would come from interaction history)
        # For now, using simplified version
        user_categories = Counter()
        user_tags: Set[str] = set()

        # Score all videos
        scores = []
        for video_id in self.video_vectors.keys():
            if video_id not in exclude_set:
                score = self._calculate_similarity(video_id, user_profile,
                                                   user_categories, user_tags)
                scores.append((video_id, score))

        # Sort and return top N
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:n_recommendations]

    def get_similar_videos(self, video_id: str, n_similar: int = 20) -> List[Tuple[str, float]]:
        """
        Find videos similar to a given video.

        Args:
            video_id: Reference video ID
            n_similar: Number of similar videos to return

        Returns:
            List of (video_id, similarity_score) tuples
        """
        if video_id not in self.video_vectors:
            return []

        video_vector = self.video_vectors[video_id]
        video_category = self.video_categories.get(video_id, 'other')
        video_tags = self.video_tags.get(video_id, set())

        similarities = []
        for other_id in self.video_vectors.keys():
            if other_id != video_id:
                # Text similarity
                text_sim = np.dot(self.video_vectors[other_id], video_vector)

                # Category match
                category_sim = 1.0 if self.video_categories.get(other_id) == video_category else 0.0

                # Tag similarity
                other_tags = self.video_tags.get(other_id, set())
                if video_tags and other_tags:
                    tag_sim = len(video_tags & other_tags) / len(video_tags | other_tags)
                else:
                    tag_sim = 0.0

                # Combined similarity
                total_sim = (
                    self.text_weight * text_sim +
                    self.category_weight * category_sim +
                    self.tag_weight * tag_sim
                )
                similarities.append((other_id, total_sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:n_similar]
