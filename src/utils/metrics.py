"""
Evaluation metrics for recommendation system.
"""
import numpy as np
from typing import List, Set, Dict
from collections import defaultdict


def precision_at_k(recommended: List[str], relevant: Set[str], k: int = 10) -> float:
    """
    Calculate Precision@K.

    Args:
        recommended: List of recommended item IDs
        relevant: Set of relevant item IDs
        k: Number of top items to consider

    Returns:
        Precision@K score
    """
    if not recommended or not relevant:
        return 0.0

    recommended_k = recommended[:k]
    hits = sum(1 for item in recommended_k if item in relevant)

    return hits / k


def recall_at_k(recommended: List[str], relevant: Set[str], k: int = 10) -> float:
    """
    Calculate Recall@K.

    Args:
        recommended: List of recommended item IDs
        relevant: Set of relevant item IDs
        k: Number of top items to consider

    Returns:
        Recall@K score
    """
    if not recommended or not relevant:
        return 0.0

    recommended_k = recommended[:k]
    hits = sum(1 for item in recommended_k if item in relevant)

    return hits / len(relevant)


def average_precision(recommended: List[str], relevant: Set[str]) -> float:
    """
    Calculate Average Precision (AP).

    Args:
        recommended: List of recommended item IDs
        relevant: Set of relevant item IDs

    Returns:
        Average Precision score
    """
    if not recommended or not relevant:
        return 0.0

    hits = 0
    sum_precisions = 0.0

    for i, item in enumerate(recommended, 1):
        if item in relevant:
            hits += 1
            precision_at_i = hits / i
            sum_precisions += precision_at_i

    return sum_precisions / len(relevant)


def mean_average_precision(recommendations: Dict[str, List[str]],
                           relevance: Dict[str, Set[str]]) -> float:
    """
    Calculate Mean Average Precision (MAP) across all users.

    Args:
        recommendations: Dict mapping user_id to list of recommended items
        relevance: Dict mapping user_id to set of relevant items

    Returns:
        MAP score
    """
    ap_scores = []

    for user_id in recommendations:
        if user_id in relevance:
            ap = average_precision(recommendations[user_id], relevance[user_id])
            ap_scores.append(ap)

    return np.mean(ap_scores) if ap_scores else 0.0


def ndcg_at_k(recommended: List[str], relevant: Set[str],
              relevance_scores: Dict[str, float] = None,
              k: int = 10) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG@K).

    Args:
        recommended: List of recommended item IDs
        relevant: Set of relevant item IDs
        relevance_scores: Optional dict mapping item to relevance score
        k: Number of top items to consider

    Returns:
        NDCG@K score
    """
    if not recommended or not relevant:
        return 0.0

    # Use binary relevance if scores not provided
    if relevance_scores is None:
        relevance_scores = {item: 1.0 for item in relevant}

    # Calculate DCG
    dcg = 0.0
    for i, item in enumerate(recommended[:k], 1):
        if item in relevant:
            rel = relevance_scores.get(item, 0.0)
            dcg += rel / np.log2(i + 1)

    # Calculate IDCG (ideal DCG)
    ideal_items = sorted(relevant, key=lambda x: relevance_scores.get(x, 0.0), reverse=True)
    idcg = 0.0
    for i, item in enumerate(ideal_items[:k], 1):
        rel = relevance_scores.get(item, 0.0)
        idcg += rel / np.log2(i + 1)

    return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(recommended: List[str], relevant: Set[str], k: int = 10) -> float:
    """
    Calculate Hit Rate@K (whether any relevant item is in top K).

    Args:
        recommended: List of recommended item IDs
        relevant: Set of relevant item IDs
        k: Number of top items to consider

    Returns:
        1.0 if hit, 0.0 otherwise
    """
    recommended_k = set(recommended[:k])
    return 1.0 if len(recommended_k & relevant) > 0 else 0.0


def mrr(recommended: List[str], relevant: Set[str]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).

    Args:
        recommended: List of recommended item IDs
        relevant: Set of relevant item IDs

    Returns:
        Reciprocal rank of first relevant item
    """
    for i, item in enumerate(recommended, 1):
        if item in relevant:
            return 1.0 / i
    return 0.0


def diversity_score(recommended: List[str], item_categories: Dict[str, str]) -> float:
    """
    Calculate diversity of recommendations based on category distribution.

    Args:
        recommended: List of recommended item IDs
        item_categories: Dict mapping item_id to category

    Returns:
        Diversity score (0-1, higher is more diverse)
    """
    if not recommended:
        return 0.0

    # Count categories
    categories = [item_categories.get(item, 'unknown') for item in recommended]
    category_counts = defaultdict(int)
    for cat in categories:
        category_counts[cat] += 1

    # Calculate entropy
    n = len(recommended)
    entropy = 0.0
    for count in category_counts.values():
        p = count / n
        entropy -= p * np.log2(p)

    # Normalize by max entropy
    max_entropy = np.log2(len(category_counts))
    return entropy / max_entropy if max_entropy > 0 else 0.0


def coverage(all_recommended: Set[str], all_items: Set[str]) -> float:
    """
    Calculate catalog coverage (percentage of items recommended).

    Args:
        all_recommended: Set of all recommended items across users
        all_items: Set of all available items

    Returns:
        Coverage score (0-1)
    """
    if not all_items:
        return 0.0

    return len(all_recommended & all_items) / len(all_items)


def evaluate_recommendations(recommendations: Dict[str, List[str]],
                             relevance: Dict[str, Set[str]],
                             k_values: List[int] = [5, 10, 20]) -> Dict[str, float]:
    """
    Evaluate recommendations using multiple metrics.

    Args:
        recommendations: Dict mapping user_id to list of recommended items
        relevance: Dict mapping user_id to set of relevant items
        k_values: List of K values to evaluate

    Returns:
        Dictionary of metric scores
    """
    results = {}

    for k in k_values:
        precision_scores = []
        recall_scores = []
        ndcg_scores = []
        hit_rates = []

        for user_id in recommendations:
            if user_id not in relevance:
                continue

            rec = recommendations[user_id]
            rel = relevance[user_id]

            precision_scores.append(precision_at_k(rec, rel, k))
            recall_scores.append(recall_at_k(rec, rel, k))
            ndcg_scores.append(ndcg_at_k(rec, rel, k=k))
            hit_rates.append(hit_rate_at_k(rec, rel, k))

        results[f'precision@{k}'] = np.mean(precision_scores)
        results[f'recall@{k}'] = np.mean(recall_scores)
        results[f'ndcg@{k}'] = np.mean(ndcg_scores)
        results[f'hit_rate@{k}'] = np.mean(hit_rates)

    # MAP and MRR
    results['map'] = mean_average_precision(recommendations, relevance)

    mrr_scores = []
    for user_id in recommendations:
        if user_id in relevance:
            mrr_scores.append(mrr(recommendations[user_id], relevance[user_id]))
    results['mrr'] = np.mean(mrr_scores) if mrr_scores else 0.0

    return results
