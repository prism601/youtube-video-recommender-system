"""
Collaborative Filtering algorithms for video recommendations.
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class MatrixFactorization:
    """
    Matrix Factorization using Alternating Least Squares (ALS).
    Decomposes user-item interaction matrix into user and item latent factors.
    """

    def __init__(self, n_factors: int = 50, n_iterations: int = 20,
                 regularization: float = 0.01, learning_rate: float = 0.01):
        """
        Initialize matrix factorization model.

        Args:
            n_factors: Number of latent factors
            n_iterations: Number of training iterations
            regularization: L2 regularization parameter
            learning_rate: Learning rate for SGD
        """
        self.n_factors = n_factors
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.learning_rate = learning_rate

        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None
        self.user_biases: Optional[np.ndarray] = None
        self.item_biases: Optional[np.ndarray] = None
        self.global_mean: float = 0.0

        self.user_id_map: Dict[str, int] = {}
        self.item_id_map: Dict[str, int] = {}
        self.reverse_item_map: Dict[int, str] = {}

    def fit(self, interactions: List[Tuple[str, str, float]]):
        """
        Train the model on user-item interactions.

        Args:
            interactions: List of (user_id, video_id, rating) tuples
        """
        logger.info(f"Training matrix factorization with {len(interactions)} interactions")

        # Build user and item mappings
        users = sorted(set(user_id for user_id, _, _ in interactions))
        items = sorted(set(item_id for _, item_id, _ in interactions))

        self.user_id_map = {uid: idx for idx, uid in enumerate(users)}
        self.item_id_map = {iid: idx for idx, iid in enumerate(items)}
        self.reverse_item_map = {idx: iid for iid, idx in self.item_id_map.items()}

        n_users = len(users)
        n_items = len(items)

        # Initialize factors randomly
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        self.user_biases = np.zeros(n_users)
        self.item_biases = np.zeros(n_items)

        # Calculate global mean
        ratings = [rating for _, _, rating in interactions]
        self.global_mean = np.mean(ratings)

        # Training loop (SGD)
        for iteration in range(self.n_iterations):
            squared_error = 0.0

            for user_id, item_id, rating in interactions:
                user_idx = self.user_id_map[user_id]
                item_idx = self.item_id_map[item_id]

                # Predict rating
                prediction = self._predict_single(user_idx, item_idx)

                # Calculate error
                error = rating - prediction
                squared_error += error ** 2

                # Update biases
                self.user_biases[user_idx] += self.learning_rate * (
                    error - self.regularization * self.user_biases[user_idx]
                )
                self.item_biases[item_idx] += self.learning_rate * (
                    error - self.regularization * self.item_biases[item_idx]
                )

                # Update factors
                user_factor_old = self.user_factors[user_idx].copy()
                self.user_factors[user_idx] += self.learning_rate * (
                    error * self.item_factors[item_idx] -
                    self.regularization * self.user_factors[user_idx]
                )
                self.item_factors[item_idx] += self.learning_rate * (
                    error * user_factor_old -
                    self.regularization * self.item_factors[item_idx]
                )

            rmse = np.sqrt(squared_error / len(interactions))
            if (iteration + 1) % 5 == 0:
                logger.info(f"Iteration {iteration + 1}/{self.n_iterations}, RMSE: {rmse:.4f}")

    def _predict_single(self, user_idx: int, item_idx: int) -> float:
        """Predict rating for a single user-item pair."""
        prediction = (
            self.global_mean +
            self.user_biases[user_idx] +
            self.item_biases[item_idx] +
            np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
        )
        return prediction

    def recommend(self, user_id: str, n_recommendations: int = 100,
                  exclude_items: Optional[List[str]] = None) -> List[Tuple[str, float]]:
        """
        Generate recommendations for a user.

        Args:
            user_id: User identifier
            n_recommendations: Number of recommendations to return
            exclude_items: Items to exclude from recommendations

        Returns:
            List of (video_id, score) tuples
        """
        if user_id not in self.user_id_map:
            logger.warning(f"User {user_id} not in training data, returning empty recommendations")
            return []

        user_idx = self.user_id_map[user_id]
        exclude_items = exclude_items or []
        exclude_indices = {self.item_id_map[iid] for iid in exclude_items if iid in self.item_id_map}

        # Predict scores for all items
        scores = []
        for item_idx in range(len(self.item_id_map)):
            if item_idx not in exclude_indices:
                score = self._predict_single(user_idx, item_idx)
                item_id = self.reverse_item_map[item_idx]
                scores.append((item_id, score))

        # Sort by score and return top N
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:n_recommendations]

    def get_user_embedding(self, user_id: str) -> Optional[np.ndarray]:
        """Get user embedding vector."""
        if user_id not in self.user_id_map:
            return None
        user_idx = self.user_id_map[user_id]
        return self.user_factors[user_idx]

    def get_item_embedding(self, item_id: str) -> Optional[np.ndarray]:
        """Get item embedding vector."""
        if item_id not in self.item_id_map:
            return None
        item_idx = self.item_id_map[item_id]
        return self.item_factors[item_idx]


class ItemBasedCF:
    """
    Item-based Collaborative Filtering.
    Recommends items similar to those the user has interacted with.
    """

    def __init__(self, k_neighbors: int = 50):
        """
        Initialize item-based CF.

        Args:
            k_neighbors: Number of similar items to consider
        """
        self.k_neighbors = k_neighbors
        self.item_similarity: Dict[str, List[Tuple[str, float]]] = {}
        self.user_items: Dict[str, List[Tuple[str, float]]] = defaultdict(list)

    def fit(self, interactions: List[Tuple[str, str, float]]):
        """
        Build item similarity matrix.

        Args:
            interactions: List of (user_id, video_id, rating) tuples
        """
        logger.info("Building item-based CF similarity matrix")

        # Build user-item interaction map
        for user_id, item_id, rating in interactions:
            self.user_items[user_id].append((item_id, rating))

        # Build item-user inverted index
        item_users: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        for user_id, item_id, rating in interactions:
            item_users[item_id].append((user_id, rating))

        # Calculate item-item similarities (cosine similarity)
        items = list(item_users.keys())
        for i, item1 in enumerate(items):
            similarities = []

            for item2 in items:
                if item1 != item2:
                    similarity = self._cosine_similarity(item_users[item1], item_users[item2])
                    if similarity > 0:
                        similarities.append((item2, similarity))

            # Keep top K similar items
            similarities.sort(key=lambda x: x[1], reverse=True)
            self.item_similarity[item1] = similarities[:self.k_neighbors]

            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(items)} items")

    def _cosine_similarity(self, users1: List[Tuple[str, float]],
                          users2: List[Tuple[str, float]]) -> float:
        """Calculate cosine similarity between two items based on user ratings."""
        # Create rating vectors
        ratings1 = {uid: rating for uid, rating in users1}
        ratings2 = {uid: rating for uid, rating in users2}

        # Find common users
        common_users = set(ratings1.keys()) & set(ratings2.keys())
        if not common_users:
            return 0.0

        # Calculate cosine similarity
        dot_product = sum(ratings1[uid] * ratings2[uid] for uid in common_users)
        norm1 = np.sqrt(sum(ratings1[uid] ** 2 for uid in common_users))
        norm2 = np.sqrt(sum(ratings2[uid] ** 2 for uid in common_users))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def recommend(self, user_id: str, n_recommendations: int = 100,
                  exclude_items: Optional[List[str]] = None) -> List[Tuple[str, float]]:
        """
        Generate recommendations for a user.

        Args:
            user_id: User identifier
            n_recommendations: Number of recommendations
            exclude_items: Items to exclude

        Returns:
            List of (video_id, score) tuples
        """
        if user_id not in self.user_items:
            logger.warning(f"User {user_id} not in training data")
            return []

        exclude_items = exclude_items or []
        user_watched = {item_id for item_id, _ in self.user_items[user_id]}
        exclude_set = set(exclude_items) | user_watched

        # Score all candidate items
        candidate_scores: Dict[str, float] = defaultdict(float)

        for item_id, rating in self.user_items[user_id]:
            if item_id not in self.item_similarity:
                continue

            # Add scores from similar items
            for similar_item, similarity in self.item_similarity[item_id]:
                if similar_item not in exclude_set:
                    candidate_scores[similar_item] += similarity * rating

        # Convert to list and sort
        recommendations = [(item_id, score) for item_id, score in candidate_scores.items()]
        recommendations.sort(key=lambda x: x[1], reverse=True)

        return recommendations[:n_recommendations]
