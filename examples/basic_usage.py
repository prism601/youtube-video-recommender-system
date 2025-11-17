"""
Basic usage example of the YouTube recommendation system.
"""
import sys
sys.path.insert(0, '..')

from src.utils.data_generator import SampleDataGenerator
from src.algorithms.hybrid_recommender import HybridRecommender
from src.api.recommendation_service import RecommendationService
from src.models import RecommendationRequest


def main():
    """Demonstrate basic usage of the recommendation system."""

    print("=" * 80)
    print("YouTube Video Recommendation System - Basic Usage Example")
    print("=" * 80)

    # Step 1: Generate sample data
    print("\n[Step 1] Generating sample data...")
    generator = SampleDataGenerator(seed=42)
    dataset = generator.generate_complete_dataset(
        n_users=50,
        n_videos=500,
        interactions_per_user=30
    )

    users = dataset['users']
    videos = dataset['videos']
    interactions = dataset['interactions']
    video_stats = dataset['video_stats']

    print(f"  ✓ Generated {len(users)} users")
    print(f"  ✓ Generated {len(videos)} videos")
    print(f"  ✓ Generated {len(interactions)} interactions")

    # Step 2: Initialize and train recommender
    print("\n[Step 2] Training recommendation models...")
    recommender = HybridRecommender()
    recommender.train(interactions, videos, video_stats)
    print("  ✓ Models trained successfully")

    # Step 3: Create recommendation service
    print("\n[Step 3] Creating recommendation service...")
    service = RecommendationService(recommender)
    print("  ✓ Service initialized")

    # Step 4: Generate recommendations for sample users
    print("\n[Step 4] Generating recommendations...")

    sample_user = users[0]
    user_id = sample_user['user_id']

    print(f"\nGenerating recommendations for user: {user_id}")
    print(f"  Preferred categories: {sample_user['preferred_categories']}")

    # Create recommendation request
    request = RecommendationRequest(
        user_id=user_id,
        num_recommendations=10,
        context='homepage',
        enable_diversity=True,
        enable_exploration=True
    )

    # Get recommendations
    result = service.get_recommendations(
        request=request,
        video_metadata={v['video_id']: v for v in videos}
    )

    print(f"\n  ✓ Generated {len(result.recommendations)} recommendations")
    print(f"  ✓ Generation time: {result.generation_time_ms:.2f}ms")

    # Display recommendations
    print("\n" + "=" * 80)
    print("TOP RECOMMENDATIONS")
    print("=" * 80)

    for i, rec in enumerate(result.recommendations[:10], 1):
        video = next((v for v in videos if v['video_id'] == rec.video_id), None)
        if video:
            print(f"\n{i}. {rec.title}")
            print(f"   Channel: {rec.channel_name}")
            print(f"   Category: {video['category']}")
            print(f"   Score: {rec.score.final_score:.4f}")
            print(f"   Reason: {rec.reason}")
            print(f"   Views: {video['view_count']:,}")

    # Step 5: Get trending videos
    print("\n" + "=" * 80)
    print("TRENDING VIDEOS")
    print("=" * 80)

    trending = service.get_trending_recommendations(n_recommendations=5)
    for i, trend in enumerate(trending[:5], 1):
        video = next((v for v in videos if v['video_id'] == trend['video_id']), None)
        if video:
            print(f"\n{i}. {video['title']}")
            print(f"   Category: {video['category']}")
            print(f"   Trending Score: {trend['trending_score']:.2f}")
            print(f"   Views: {video['view_count']:,}")

    # Step 6: Get similar videos
    print("\n" + "=" * 80)
    print("SIMILAR VIDEOS")
    print("=" * 80)

    reference_video = videos[10]
    print(f"\nFinding videos similar to: {reference_video['title']}")
    print(f"Category: {reference_video['category']}")

    similar = service.get_similar_videos(
        video_id=reference_video['video_id'],
        n_similar=5
    )

    for i, sim in enumerate(similar[:5], 1):
        video = next((v for v in videos if v['video_id'] == sim['video_id']), None)
        if video:
            print(f"\n{i}. {video['title']}")
            print(f"   Category: {video['category']}")
            print(f"   Similarity: {sim['similarity_score']:.4f}")

    # Step 7: Log interaction and get updated recommendations
    print("\n" + "=" * 80)
    print("LOGGING INTERACTION")
    print("=" * 80)

    watched_video = result.recommendations[0].video_id
    print(f"\nLogging: User watched video {watched_video}")
    service.log_interaction(user_id, watched_video, 'view')
    print("  ✓ Interaction logged (cache invalidated)")

    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)


if __name__ == '__main__':
    main()
