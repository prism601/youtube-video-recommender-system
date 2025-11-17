# Quick Start Guide

Get started with the YouTube Video Recommendation System in minutes.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd youtube-video-recommender-system
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## Running the Examples

### Basic Usage Example

Run the basic usage example to see the system in action:

```bash
cd examples
python basic_usage.py
```

This will:
1. Generate sample data (users, videos, interactions)
2. Train recommendation models
3. Generate personalized recommendations
4. Show trending videos
5. Find similar videos

**Expected Output:**
```
================================================================================
YouTube Video Recommendation System - Basic Usage Example
================================================================================

[Step 1] Generating sample data...
  ✓ Generated 50 users
  ✓ Generated 500 videos
  ✓ Generated 1500 interactions

[Step 2] Training recommendation models...
  ✓ Models trained successfully

[Step 3] Creating recommendation service...
  ✓ Service initialized

[Step 4] Generating recommendations...
  ✓ Generated 10 recommendations in 42.15ms

TOP RECOMMENDATIONS
================================================================================
1. Amazing Tutorial Guide
   Channel: Channel 15
   Category: education
   Score: 0.8542
   Reason: Because you watch education videos
   ...
```

### Evaluation Example

Evaluate the recommendation system using standard metrics:

```bash
cd examples
python evaluate_system.py
```

This will:
1. Generate a larger dataset
2. Split into training and test sets
3. Train models
4. Generate recommendations for test users
5. Calculate evaluation metrics (Precision, Recall, NDCG, MAP)
6. Compare different algorithms

**Expected Output:**
```
EVALUATION RESULTS
================================================================================

Metric               Score
------------------------------
precision@5          0.1234
recall@5             0.0876
ndcg@5               0.2145
map                  0.1567
mrr                  0.2341
...
```

## Running the API Server

### Start the Flask Server

```bash
cd src/api
python flask_app.py
```

The API will be available at `http://localhost:5000`

### Test the API

#### 1. Health Check

```bash
curl http://localhost:5000/health
```

#### 2. Get Recommendations

First, you need to train the models using sample data. In a new terminal:

```python
# Python script to train models
from src.utils.data_generator import SampleDataGenerator
from src.algorithms.hybrid_recommender import HybridRecommender
from src.api.recommendation_service import RecommendationService
import pickle

# Generate data
generator = SampleDataGenerator()
dataset = generator.generate_complete_dataset(n_users=100, n_videos=1000)

# Train models
recommender = HybridRecommender()
recommender.train(dataset['interactions'], dataset['videos'], dataset['video_stats'])

# Save trained model
with open('trained_model.pkl', 'wb') as f:
    pickle.dump(recommender, f)
```

Then get recommendations:

```bash
curl "http://localhost:5000/api/v1/recommendations/user_00001?num_recommendations=10"
```

#### 3. Get Trending Videos

```bash
curl "http://localhost:5000/api/v1/trending?num_videos=20"
```

#### 4. Log Interaction

```bash
curl -X POST "http://localhost:5000/api/v1/interactions" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_00001",
    "video_id": "video_000123",
    "interaction_type": "view"
  }'
```

## Using the System Programmatically

### Basic Recommendation Pipeline

```python
from src.utils.data_generator import SampleDataGenerator
from src.algorithms.hybrid_recommender import HybridRecommender
from src.models import RecommendationRequest
from src.api.recommendation_service import RecommendationService

# 1. Generate or load data
generator = SampleDataGenerator()
dataset = generator.generate_complete_dataset(
    n_users=100,
    n_videos=1000,
    interactions_per_user=50
)

# 2. Initialize and train recommender
recommender = HybridRecommender()
recommender.train(
    interactions=dataset['interactions'],
    videos=dataset['videos'],
    video_stats=dataset['video_stats']
)

# 3. Create service
service = RecommendationService(recommender)

# 4. Create recommendation request
request = RecommendationRequest(
    user_id='user_00001',
    num_recommendations=20,
    context='homepage',
    enable_diversity=True,
    enable_exploration=True
)

# 5. Get recommendations
video_metadata = {v['video_id']: v for v in dataset['videos']}
result = service.get_recommendations(
    request=request,
    video_metadata=video_metadata
)

# 6. Display results
for rec in result.recommendations[:10]:
    print(f"{rec.rank}. {rec.title} (score: {rec.score.final_score:.4f})")
```

### Custom Algorithm Weights

```python
from src.algorithms.hybrid_recommender import HybridRecommender

# Create recommender with custom weights
custom_weights = {
    'collaborative_mf': 0.4,      # Increase collaborative filtering
    'collaborative_item': 0.2,
    'content_based': 0.3,         # Increase content-based
    'trending': 0.05,             # Reduce trending
    'new_uploads': 0.05,
}

recommender = HybridRecommender(weights=custom_weights)
recommender.train(interactions, videos, video_stats)
```

### Individual Algorithm Usage

```python
from src.algorithms import MatrixFactorization, ContentBasedRecommender, TrendingRecommender

# Use only collaborative filtering
cf_model = MatrixFactorization(n_factors=50, n_iterations=20)
cf_model.fit(interactions)
recommendations = cf_model.recommend(user_id='user_00001', n_recommendations=20)

# Use only content-based filtering
content_model = ContentBasedRecommender()
content_model.fit(videos, interactions)
recommendations = content_model.recommend(user_id='user_00001', n_recommendations=20)

# Use only trending
trending_model = TrendingRecommender()
for video_id, stats in video_stats.items():
    trending_model.update_video_stats(video_id, stats)
trending_videos = trending_model.get_trending_videos(n_videos=50)
```

## Next Steps

- Read the [Architecture Documentation](ARCHITECTURE.md) for system design details
- Check the [API Documentation](API_DOCUMENTATION.md) for complete API reference
- See [Algorithm Details](ALGORITHMS.md) for recommendation algorithm explanations
- Review configuration options in `config/config.yaml`

## Troubleshooting

### Common Issues

**Issue:** `ModuleNotFoundError: No module named 'src'`

**Solution:** Make sure you're running scripts from the correct directory or add the project root to PYTHONPATH:

```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/youtube-video-recommender-system"
```

**Issue:** Models return empty recommendations

**Solution:** Ensure models are trained with sufficient data. Minimum requirements:
- At least 10 users
- At least 50 videos
- At least 5 interactions per user

**Issue:** API returns 500 error

**Solution:** Check that models are trained before making API requests. The system requires trained models to generate recommendations.
