# Recommendation Algorithms

This document provides detailed information about the recommendation algorithms used in the YouTube Video Recommendation System.

## Algorithm Overview

The system uses a **hybrid approach** combining multiple recommendation strategies:

1. **Collaborative Filtering** (50% weight)
   - Matrix Factorization (30%)
   - Item-Based CF (20%)

2. **Content-Based Filtering** (25% weight)

3. **Trending/Popular** (15% weight)

4. **New Uploads** (10% weight)

---

## 1. Collaborative Filtering

### 1.1 Matrix Factorization (MF)

**Algorithm:** Alternating Least Squares (ALS) / Stochastic Gradient Descent (SGD)

**How it works:**
- Decomposes the user-item interaction matrix into two lower-rank matrices
- User matrix: `U (n_users × k_factors)`
- Item matrix: `V (n_items × k_factors)`
- Predicted rating: `R̂ᵤᵢ = μ + bᵤ + bᵢ + Uᵤ · Vᵢᵀ`

**Parameters:**
- `n_factors`: 50 (latent dimensions)
- `n_iterations`: 20 (training epochs)
- `learning_rate`: 0.01
- `regularization`: 0.01 (L2 penalty)

**Advantages:**
- Captures latent user preferences
- Scalable to large datasets
- Handles sparse data well

**Disadvantages:**
- Cold start problem for new users/items
- Requires sufficient interaction data

**Code Example:**
```python
from src.algorithms import MatrixFactorization

mf = MatrixFactorization(n_factors=50, n_iterations=20)
mf.fit(interactions)  # List of (user_id, video_id, rating)
recommendations = mf.recommend(user_id='user_123', n_recommendations=20)
```

---

### 1.2 Item-Based Collaborative Filtering

**Algorithm:** Item-Item Similarity with Cosine Distance

**How it works:**
- Computes similarity between all pairs of items based on user co-interactions
- For user U, recommends items similar to items U has liked
- Similarity metric: Cosine similarity on user rating vectors

**Formula:**
```
similarity(i, j) = cos(Rᵢ, Rⱼ) = (Rᵢ · Rⱼ) / (||Rᵢ|| ||Rⱼ||)

score(u, i) = Σⱼ∈rated(u) similarity(i, j) × rating(u, j)
```

**Parameters:**
- `k_neighbors`: 50 (top similar items to consider)

**Advantages:**
- More stable than user-based CF
- Explains recommendations easily ("similar to videos you watched")
- Works well for users with limited history

**Disadvantages:**
- Popularity bias
- Computationally expensive similarity computation

---

## 2. Content-Based Filtering

**Algorithm:** TF-IDF + Category/Tag Matching

**How it works:**
- Builds content representation for each video using:
  - Text features (title, description) → TF-IDF vectors
  - Category labels
  - Tags/keywords
- Creates user profile as weighted average of watched videos
- Recommends videos with high similarity to user profile

**Similarity Calculation:**
```
similarity(video, user_profile) =
    w₁ × cosine(TF-IDF_video, TF-IDF_user) +
    w₂ × category_match(video, user) +
    w₃ × tag_jaccard(video, user)
```

**Weights:**
- Text similarity: 50%
- Category similarity: 30%
- Tag similarity: 20%

**Parameters:**
- `max_features`: 1000 (vocabulary size)
- `min_df`: 2 (minimum document frequency)

**Advantages:**
- No cold start for items (can recommend new videos)
- Explainable ("because you watch category X")
- User-independent (doesn't need other users' data)

**Disadvantages:**
- Limited discovery (recommends similar content)
- Requires good metadata
- Over-specialization risk (filter bubble)

---

## 3. Trending & Popular Videos

### 3.1 Trending Score

**Algorithm:** Time-weighted engagement metrics

**How it works:**
- Combines view velocity, engagement rate, and recency
- Uses exponential time decay to favor recent content

**Formula:**
```
trending_score =
    0.4 × velocity_score +
    0.3 × engagement_score +
    0.3 × recency_boost

where:
    velocity_score = normalized(views / age_hours)
    engagement_score = normalized((likes + 2×comments + 3×shares) / views)
    recency_boost = exp(-age_hours / decay_hours)
```

**Parameters:**
- `time_decay_hours`: 24.0 (decay half-life)
- `min_views`: 1000 (minimum views to be trending)

**Advantages:**
- Captures viral content
- Real-time trending detection
- Helps with discovery

---

### 3.2 Popular Videos

**Algorithm:** View count + engagement rate

**Formula:**
```
popularity_score =
    0.6 × normalized(view_count) +
    0.4 × normalized(engagement_rate)
```

**Advantages:**
- Simple and effective
- Stable recommendations
- Good for new/inactive users

---

## 4. Hybrid Fusion

### 4.1 Candidate Generation

Each algorithm generates top-K candidates independently:
- Collaborative MF: 100 candidates
- Item-based CF: 100 candidates
- Content-based: 100 candidates
- Trending: 50 candidates
- New uploads: 25 candidates

**Total candidate pool:** ~300-400 videos (after deduplication)

---

### 4.2 Score Fusion

**Algorithm:** Weighted Linear Combination

**Process:**
1. Normalize scores from each algorithm to [0, 1]
2. Apply algorithm weights
3. Sum weighted scores

**Formula:**
```
final_score(video) = Σₐ wₐ × normalized_scoreₐ(video)

where a ∈ {collaborative_mf, collaborative_item, content_based, trending, new_uploads}
```

**Default Weights:**
```python
{
    'collaborative_mf': 0.30,
    'collaborative_item': 0.20,
    'content_based': 0.25,
    'trending': 0.15,
    'new_uploads': 0.10,
}
```

---

### 4.3 Re-ranking & Post-processing

#### Diversity Boosting

Applies penalty to over-represented categories/channels:

```
diversity_penalty =
    0.1 × category_count +
    0.15 × channel_count

adjusted_score = score × (1 - diversity_factor × diversity_penalty)
```

**Parameter:** `diversity_factor` = 0.3

#### Exploration (ε-greedy)

Replaces X% of recommendations with random videos:

```
final_recommendations =
    (1 - ε) × top_ranked_videos +
    ε × random_videos

where ε = exploration_rate (default: 0.15)
```

**Benefits:**
- Breaks filter bubbles
- Enables serendipitous discovery
- Gathers data for new content

---

## 5. Special Cases

### 5.1 Cold Start - New User

**Strategy:**
1. Use trending/popular videos (60%)
2. Sample from diverse categories (30%)
3. Random exploration (10%)

### 5.2 Cold Start - New Video

**Strategy:**
- Content-based filtering can recommend immediately
- Boost based on channel popularity
- A/B test with small user segment

### 5.3 Sparse Data

**Strategy:**
- Increase weight on content-based and trending
- Use category-based recommendations
- Leverage global popularity

---

## 6. Evaluation Metrics

### Offline Metrics

**Precision@K:** Fraction of recommended items that are relevant
```
Precision@K = |relevant ∩ recommended@K| / K
```

**Recall@K:** Fraction of relevant items that are recommended
```
Recall@K = |relevant ∩ recommended@K| / |relevant|
```

**NDCG@K:** Discounted cumulative gain (accounts for ranking)
```
NDCG@K = DCG@K / IDCG@K
where DCG@K = Σᵢ₌₁ᴷ (2^relᵢ - 1) / log₂(i + 1)
```

**MAP:** Mean Average Precision across all users

**MRR:** Mean Reciprocal Rank of first relevant item

### Online Metrics (A/B Testing)

- **CTR:** Click-through rate
- **Watch Time:** Average watch duration
- **Engagement Rate:** Likes, comments, shares per view
- **User Retention:** Return rate within 7 days
- **Session Duration:** Time spent on platform

---

## 7. Algorithm Selection Guide

**Use Collaborative Filtering when:**
- You have rich interaction data
- Users have established viewing patterns
- You want to find similar users' preferences

**Use Content-Based when:**
- Videos have good metadata
- You want explainable recommendations
- Dealing with new videos (cold start)

**Use Trending when:**
- You want to surface viral content
- User has no history
- Homepage/discovery context

**Use Hybrid when:**
- You want best overall performance
- You have diverse use cases
- You want to balance exploration/exploitation

---

## 8. Tuning Guidelines

### Improving Precision
- Increase collaborative filtering weight
- Decrease exploration rate
- Use stricter relevance thresholds

### Improving Diversity
- Increase diversity_factor
- Increase trending weight
- Boost exploration rate

### Improving Coverage
- Increase content-based weight
- Enable exploration
- Reduce popularity bias

### Faster Training
- Reduce n_iterations for MF
- Reduce k_neighbors for item-based CF
- Reduce max_features for TF-IDF

### Better Cold Start
- Increase content-based weight
- Increase trending weight
- Use category-based fallbacks
