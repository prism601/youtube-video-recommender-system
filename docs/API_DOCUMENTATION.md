# API Documentation

## Overview

The YouTube Video Recommendation System provides a RESTful API for generating personalized video recommendations.

Base URL: `http://localhost:5000/api/v1`

## Endpoints

### 1. Get Recommendations

Get personalized video recommendations for a user.

**Endpoint:** `GET /recommendations/{user_id}`

**Parameters:**
- `user_id` (path, required): User identifier
- `num_recommendations` (query, optional): Number of recommendations (default: 20, max: 100)
- `context` (query, optional): Context of request (`homepage`, `search`, `watch_page`)
- `enable_diversity` (query, optional): Enable diversity boosting (default: true)
- `enable_exploration` (query, optional): Enable exploration (default: true)
- `exploration_rate` (query, optional): Exploration rate 0-1 (default: 0.15)

**Example Request:**
```bash
curl "http://localhost:5000/api/v1/recommendations/user_12345?num_recommendations=10&context=homepage"
```

**Example Response:**
```json
{
  "user_id": "user_12345",
  "recommendations": [
    {
      "video_id": "video_001",
      "title": "How to Learn Python",
      "channel_name": "Tech Academy",
      "thumbnail_url": "https://example.com/thumb.jpg",
      "duration": 600,
      "rank": 1,
      "score": 0.95,
      "reason": "Because you watch education videos"
    }
  ],
  "generated_at": "2025-11-17T10:30:00",
  "total_candidates": 100,
  "generation_time_ms": 45.2,
  "algorithm_version": "1.0"
}
```

---

### 2. Get Trending Videos

Get currently trending videos.

**Endpoint:** `GET /trending`

**Parameters:**
- `num_videos` (query, optional): Number of videos (default: 50)
- `category` (query, optional): Filter by category

**Example Request:**
```bash
curl "http://localhost:5000/api/v1/trending?num_videos=20&category=gaming"
```

**Example Response:**
```json
{
  "trending_videos": [
    {
      "video_id": "video_123",
      "trending_score": 87.5
    }
  ],
  "count": 20,
  "category": "gaming"
}
```

---

### 3. Get Similar Videos

Find videos similar to a given video.

**Endpoint:** `GET /similar/{video_id}`

**Parameters:**
- `video_id` (path, required): Reference video ID
- `num_similar` (query, optional): Number of similar videos (default: 20)

**Example Request:**
```bash
curl "http://localhost:5000/api/v1/similar/video_12345?num_similar=10"
```

**Example Response:**
```json
{
  "video_id": "video_12345",
  "similar_videos": [
    {
      "video_id": "video_678",
      "similarity_score": 0.89
    }
  ],
  "count": 10
}
```

---

### 4. Log Interaction

Log user interaction with a video.

**Endpoint:** `POST /interactions`

**Request Body:**
```json
{
  "user_id": "user_12345",
  "video_id": "video_678",
  "interaction_type": "view"
}
```

**Interaction Types:**
- `view`: User watched video
- `like`: User liked video
- `share`: User shared video
- `comment`: User commented on video

**Example Request:**
```bash
curl -X POST "http://localhost:5000/api/v1/interactions" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_12345",
    "video_id": "video_678",
    "interaction_type": "view"
  }'
```

**Example Response:**
```json
{
  "status": "success",
  "message": "Interaction logged"
}
```

---

### 5. Train Models

Trigger model training (admin endpoint).

**Endpoint:** `POST /train`

**Request Body:**
```json
{
  "interactions": [
    ["user_1", "video_1", 4.5],
    ["user_2", "video_2", 3.8]
  ],
  "videos": [
    {
      "video_id": "video_1",
      "title": "Sample Video",
      "category": "education"
    }
  ],
  "video_stats": {
    "video_1": {
      "view_count": 10000,
      "like_count": 500
    }
  }
}
```

**Example Response:**
```json
{
  "status": "success",
  "message": "Models trained successfully",
  "num_interactions": 1000,
  "num_videos": 500
}
```

---

### 6. Cache Management

#### Clear Cache

**Endpoint:** `POST /cache/clear`

**Example Request:**
```bash
curl -X POST "http://localhost:5000/api/v1/cache/clear"
```

**Example Response:**
```json
{
  "status": "success",
  "message": "Cache cleared"
}
```

#### Get Cache Statistics

**Endpoint:** `GET /cache/stats`

**Example Request:**
```bash
curl "http://localhost:5000/api/v1/cache/stats"
```

**Example Response:**
```json
{
  "cache_size": 15,
  "cache_ttl_seconds": 300
}
```

---

### 7. Health Check

Check API health status.

**Endpoint:** `GET /health`

**Example Request:**
```bash
curl "http://localhost:5000/health"
```

**Example Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-11-17T10:30:00",
  "service": "youtube-recommendation-api"
}
```

---

## Error Responses

All endpoints may return error responses in the following format:

```json
{
  "error": "Error type",
  "message": "Detailed error message"
}
```

**Common HTTP Status Codes:**
- `200 OK`: Request successful
- `400 Bad Request`: Invalid request parameters
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server error

---

## Rate Limiting

(To be implemented)

- Rate limit: 100 requests per minute per user
- Burst limit: 10 requests per second

---

## Authentication

(To be implemented)

Future versions will require API key authentication:

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
  "http://localhost:5000/api/v1/recommendations/user_12345"
```

---

## Webhooks

(To be implemented)

Subscribe to real-time events:
- User interaction events
- New video uploads
- Trending changes
