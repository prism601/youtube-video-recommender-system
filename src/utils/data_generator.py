"""
Utility to generate sample data for testing the recommendation system.
"""
import random
from datetime import datetime, timedelta
from typing import List, Tuple, Dict
import string


class SampleDataGenerator:
    """Generate sample data for testing recommendation algorithms."""

    def __init__(self, seed: int = 42):
        """Initialize data generator with random seed."""
        random.seed(seed)
        self.video_categories = [
            'music', 'gaming', 'education', 'entertainment',
            'news', 'sports', 'technology', 'comedy'
        ]
        self.channels = [f'channel_{i}' for i in range(50)]

    def generate_videos(self, n_videos: int = 1000) -> List[Dict]:
        """
        Generate sample video metadata.

        Args:
            n_videos: Number of videos to generate

        Returns:
            List of video dictionaries
        """
        videos = []

        for i in range(n_videos):
            video_id = f'video_{i:06d}'

            # Generate title
            title_words = random.sample(
                ['How', 'to', 'Best', 'Amazing', 'Tutorial', 'Guide', 'Tips',
                 'Top', '10', 'Ultimate', 'Learn', 'Master', 'Complete'],
                k=random.randint(3, 6)
            )
            title = ' '.join(title_words)

            # Generate description
            description = f"This is a video about {random.choice(self.video_categories)}. " * 3

            # Random metadata
            category = random.choice(self.video_categories)
            channel_id = random.choice(self.channels)
            duration = random.randint(60, 3600)  # 1 min to 1 hour

            # Generate upload date (last 365 days)
            days_ago = random.randint(0, 365)
            upload_date = datetime.now() - timedelta(days=days_ago)

            # Generate tags
            tags = random.sample(
                ['tutorial', 'how-to', 'guide', 'tips', 'tricks', 'review',
                 'gameplay', 'music', 'comedy', 'vlog', 'news', 'tech'],
                k=random.randint(2, 5)
            )

            # Statistics (power law distribution)
            view_count = int(random.paretovariate(1.5) * 1000)
            like_count = int(view_count * random.uniform(0.01, 0.05))
            dislike_count = int(like_count * random.uniform(0.05, 0.2))
            comment_count = int(view_count * random.uniform(0.001, 0.01))
            share_count = int(view_count * random.uniform(0.001, 0.005))

            video = {
                'video_id': video_id,
                'title': title,
                'description': description,
                'category': category,
                'channel_id': channel_id,
                'channel_name': f'Channel {channel_id.split("_")[1]}',
                'duration': duration,
                'upload_date': upload_date,
                'tags': tags,
                'view_count': view_count,
                'like_count': like_count,
                'dislike_count': dislike_count,
                'comment_count': comment_count,
                'share_count': share_count,
                'thumbnail_url': f'https://example.com/thumbnails/{video_id}.jpg',
            }

            videos.append(video)

        return videos

    def generate_users(self, n_users: int = 100) -> List[Dict]:
        """
        Generate sample user profiles.

        Args:
            n_users: Number of users to generate

        Returns:
            List of user dictionaries
        """
        users = []

        for i in range(n_users):
            user_id = f'user_{i:05d}'

            # Random preferences
            preferred_categories = random.sample(
                self.video_categories,
                k=random.randint(2, 4)
            )

            # Random subscriptions
            subscriptions = random.sample(
                self.channels,
                k=random.randint(5, 15)
            )

            user = {
                'user_id': user_id,
                'username': f'user_{i}',
                'created_at': datetime.now() - timedelta(days=random.randint(30, 1000)),
                'preferred_categories': preferred_categories,
                'subscriptions': subscriptions,
            }

            users.append(user)

        return users

    def generate_interactions(self,
                            users: List[Dict],
                            videos: List[Dict],
                            interactions_per_user: int = 50) -> List[Tuple[str, str, float]]:
        """
        Generate user-video interactions.

        Args:
            users: List of user dictionaries
            videos: List of video dictionaries
            interactions_per_user: Average interactions per user

        Returns:
            List of (user_id, video_id, rating) tuples
        """
        interactions = []

        # Create video lookup by category
        videos_by_category = {}
        for video in videos:
            category = video['category']
            if category not in videos_by_category:
                videos_by_category[category] = []
            videos_by_category[category].append(video)

        for user in users:
            user_id = user['user_id']
            preferred_categories = user.get('preferred_categories', [])

            # Number of interactions for this user
            n_interactions = max(10, int(random.gauss(interactions_per_user, 15)))

            # Generate interactions biased towards preferred categories
            for _ in range(n_interactions):
                # 70% chance to pick from preferred categories
                if preferred_categories and random.random() < 0.7:
                    category = random.choice(preferred_categories)
                    if category in videos_by_category:
                        video = random.choice(videos_by_category[category])
                    else:
                        video = random.choice(videos)
                else:
                    video = random.choice(videos)

                video_id = video['video_id']

                # Generate rating (implicit from watch time)
                # Higher rating if video is from preferred category
                if video['category'] in preferred_categories:
                    rating = random.uniform(3.5, 5.0)
                else:
                    rating = random.uniform(1.0, 4.5)

                interactions.append((user_id, video_id, rating))

        return interactions

    def generate_complete_dataset(self,
                                 n_users: int = 100,
                                 n_videos: int = 1000,
                                 interactions_per_user: int = 50) -> Dict:
        """
        Generate complete dataset with users, videos, and interactions.

        Args:
            n_users: Number of users
            n_videos: Number of videos
            interactions_per_user: Average interactions per user

        Returns:
            Dictionary with 'users', 'videos', 'interactions', 'video_stats'
        """
        print(f"Generating {n_videos} videos...")
        videos = self.generate_videos(n_videos)

        print(f"Generating {n_users} users...")
        users = self.generate_users(n_users)

        print(f"Generating interactions...")
        interactions = self.generate_interactions(users, videos, interactions_per_user)

        # Create video stats dictionary
        video_stats = {}
        for video in videos:
            video_stats[video['video_id']] = {
                'view_count': video['view_count'],
                'like_count': video['like_count'],
                'dislike_count': video['dislike_count'],
                'comment_count': video['comment_count'],
                'share_count': video['share_count'],
                'upload_date': video['upload_date'],
                'category': video['category'],
            }

        print(f"Generated dataset:")
        print(f"  - {len(users)} users")
        print(f"  - {len(videos)} videos")
        print(f"  - {len(interactions)} interactions")

        return {
            'users': users,
            'videos': videos,
            'interactions': interactions,
            'video_stats': video_stats,
        }
