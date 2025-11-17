"""
Example script to evaluate the recommendation system.
"""
import sys
sys.path.insert(0, '..')

from src.utils.data_generator import SampleDataGenerator
from src.utils.metrics import evaluate_recommendations
from src.algorithms.hybrid_recommender import HybridRecommender
from collections import defaultdict


def main():
    """Evaluate recommendation system using various metrics."""

    print("=" * 80)
    print("YouTube Video Recommendation System - Evaluation")
    print("=" * 80)

    # Generate data
    print("\n[1] Generating dataset...")
    generator = SampleDataGenerator(seed=42)
    dataset = generator.generate_complete_dataset(
        n_users=100,
        n_videos=1000,
        interactions_per_user=40
    )

    users = dataset['users']
    videos = dataset['videos']
    interactions = dataset['interactions']
    video_stats = dataset['video_stats']

    # Split into train and test
    print("\n[2] Splitting data into train/test...")
    user_interactions = defaultdict(list)
    for user_id, video_id, rating in interactions:
        user_interactions[user_id].append((video_id, rating))

    train_interactions = []
    test_interactions = defaultdict(set)

    for user_id, items in user_interactions.items():
        # Use 80% for training, 20% for testing
        split_idx = int(len(items) * 0.8)
        train_items = items[:split_idx]
        test_items = items[split_idx:]

        for video_id, rating in train_items:
            train_interactions.append((user_id, video_id, rating))

        # Test set: videos with high ratings (4+)
        for video_id, rating in test_items:
            if rating >= 4.0:
                test_interactions[user_id].add(video_id)

    print(f"  Training interactions: {len(train_interactions)}")
    print(f"  Test users: {len(test_interactions)}")

    # Train model
    print("\n[3] Training recommendation models...")
    recommender = HybridRecommender()
    recommender.train(train_interactions, videos, video_stats)
    print("  ✓ Training complete")

    # Generate recommendations for test users
    print("\n[4] Generating recommendations for test users...")
    recommendations = {}

    for user_id in list(test_interactions.keys())[:50]:  # Evaluate on subset
        recs = recommender.recommend(
            user_id=user_id,
            n_recommendations=20,
            enable_diversity=True,
            enable_exploration=False
        )
        recommendations[user_id] = [video_id for video_id, _ in recs]

    print(f"  Generated recommendations for {len(recommendations)} users")

    # Evaluate
    print("\n[5] Evaluating recommendations...")
    print("=" * 80)

    # Filter test_interactions to only include users we have recommendations for
    filtered_test = {uid: test_interactions[uid] for uid in recommendations.keys()
                    if uid in test_interactions}

    results = evaluate_recommendations(
        recommendations=recommendations,
        relevance=filtered_test,
        k_values=[5, 10, 20]
    )

    print("\nEVALUATION RESULTS")
    print("=" * 80)
    print(f"\n{'Metric':<20} {'Score':<10}")
    print("-" * 30)

    for metric, score in sorted(results.items()):
        print(f"{metric:<20} {score:.4f}")

    # Algorithm comparison
    print("\n\n[6] Algorithm Comparison")
    print("=" * 80)

    algorithms = {
        'Collaborative (MF)': {'collaborative_mf': 1.0, 'collaborative_item': 0.0,
                               'content_based': 0.0, 'trending': 0.0, 'new_uploads': 0.0},
        'Content-Based': {'collaborative_mf': 0.0, 'collaborative_item': 0.0,
                         'content_based': 1.0, 'trending': 0.0, 'new_uploads': 0.0},
        'Hybrid': {'collaborative_mf': 0.3, 'collaborative_item': 0.2,
                  'content_based': 0.25, 'trending': 0.15, 'new_uploads': 0.1},
    }

    comparison_results = {}

    for algo_name, weights in algorithms.items():
        print(f"\nEvaluating {algo_name}...")

        # Update weights
        test_recommender = HybridRecommender(weights=weights)
        test_recommender.train(train_interactions, videos, video_stats)

        # Generate recommendations
        algo_recs = {}
        for user_id in list(filtered_test.keys())[:30]:
            recs = test_recommender.recommend(
                user_id=user_id,
                n_recommendations=10,
                enable_diversity=False,
                enable_exploration=False
            )
            algo_recs[user_id] = [video_id for video_id, _ in recs]

        # Evaluate
        filtered_test_subset = {uid: filtered_test[uid] for uid in algo_recs.keys()}
        algo_results = evaluate_recommendations(algo_recs, filtered_test_subset, k_values=[10])

        comparison_results[algo_name] = algo_results

    print("\n\nALGORITHM COMPARISON")
    print("=" * 80)
    print(f"\n{'Algorithm':<20} {'Precision@10':<15} {'Recall@10':<15} {'NDCG@10':<15}")
    print("-" * 65)

    for algo_name, results in comparison_results.items():
        print(f"{algo_name:<20} "
              f"{results.get('precision@10', 0):<15.4f} "
              f"{results.get('recall@10', 0):<15.4f} "
              f"{results.get('ndcg@10', 0):<15.4f}")

    print("\n" + "=" * 80)
    print("Evaluation completed!")
    print("=" * 80)


if __name__ == '__main__':
    main()
