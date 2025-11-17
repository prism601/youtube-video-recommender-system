"""
Utility functions for YouTube recommendation system.
"""
from .data_generator import SampleDataGenerator
from .metrics import (
    precision_at_k,
    recall_at_k,
    average_precision,
    mean_average_precision,
    ndcg_at_k,
    hit_rate_at_k,
    mrr,
    diversity_score,
    coverage,
    evaluate_recommendations,
)

__all__ = [
    'SampleDataGenerator',
    'precision_at_k',
    'recall_at_k',
    'average_precision',
    'mean_average_precision',
    'ndcg_at_k',
    'hit_rate_at_k',
    'mrr',
    'diversity_score',
    'coverage',
    'evaluate_recommendations',
]
