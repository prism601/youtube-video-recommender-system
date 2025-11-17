"""
Flask REST API for YouTube recommendation system.
"""
from flask import Flask, request, jsonify
from typing import Dict, Any
import logging
from datetime import datetime

from .recommendation_service import RecommendationService
from ..models import RecommendationRequest, UserProfile
from ..algorithms import HybridRecommender

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Initialize recommendation service
recommender = HybridRecommender()
recommendation_service = RecommendationService(recommender)


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'youtube-recommendation-api'
    })


@app.route('/api/v1/recommendations/<user_id>', methods=['GET'])
def get_recommendations(user_id: str):
    """
    Get personalized recommendations for a user.

    Query Parameters:
        - num_recommendations (int): Number of recommendations (default: 20)
        - context (str): Context (homepage, search, watch_page)
        - enable_diversity (bool): Enable diversity (default: true)
        - enable_exploration (bool): Enable exploration (default: true)
        - exploration_rate (float): Exploration rate (default: 0.15)

    Returns:
        JSON with recommendations
    """
    try:
        # Parse query parameters
        num_recommendations = int(request.args.get('num_recommendations', 20))
        context = request.args.get('context', 'homepage')
        enable_diversity = request.args.get('enable_diversity', 'true').lower() == 'true'
        enable_exploration = request.args.get('enable_exploration', 'true').lower() == 'true'
        exploration_rate = float(request.args.get('exploration_rate', 0.15))

        # Create recommendation request
        rec_request = RecommendationRequest(
            user_id=user_id,
            num_recommendations=num_recommendations,
            context=context,
            enable_diversity=enable_diversity,
            enable_exploration=enable_exploration,
            exploration_rate=exploration_rate
        )

        # Get recommendations
        result = recommendation_service.get_recommendations(rec_request)

        return jsonify(result.to_dict()), 200

    except Exception as e:
        logger.error(f"Error generating recommendations: {e}", exc_info=True)
        return jsonify({
            'error': 'Failed to generate recommendations',
            'message': str(e)
        }), 500


@app.route('/api/v1/trending', methods=['GET'])
def get_trending():
    """
    Get trending videos.

    Query Parameters:
        - num_videos (int): Number of videos (default: 50)
        - category (str): Filter by category (optional)

    Returns:
        JSON with trending videos
    """
    try:
        num_videos = int(request.args.get('num_videos', 50))
        category = request.args.get('category')

        trending = recommendation_service.get_trending_recommendations(
            n_recommendations=num_videos,
            category=category
        )

        return jsonify({
            'trending_videos': trending,
            'count': len(trending),
            'category': category
        }), 200

    except Exception as e:
        logger.error(f"Error getting trending videos: {e}", exc_info=True)
        return jsonify({
            'error': 'Failed to get trending videos',
            'message': str(e)
        }), 500


@app.route('/api/v1/similar/<video_id>', methods=['GET'])
def get_similar(video_id: str):
    """
    Get videos similar to a given video.

    Query Parameters:
        - num_similar (int): Number of similar videos (default: 20)

    Returns:
        JSON with similar videos
    """
    try:
        num_similar = int(request.args.get('num_similar', 20))

        similar_videos = recommendation_service.get_similar_videos(
            video_id=video_id,
            n_similar=num_similar
        )

        return jsonify({
            'video_id': video_id,
            'similar_videos': similar_videos,
            'count': len(similar_videos)
        }), 200

    except Exception as e:
        logger.error(f"Error getting similar videos: {e}", exc_info=True)
        return jsonify({
            'error': 'Failed to get similar videos',
            'message': str(e)
        }), 500


@app.route('/api/v1/interactions', methods=['POST'])
def log_interaction():
    """
    Log user interaction with a video.

    Request Body:
        {
            "user_id": "user123",
            "video_id": "video456",
            "interaction_type": "view|like|share|comment"
        }

    Returns:
        Success message
    """
    try:
        data = request.get_json()

        user_id = data.get('user_id')
        video_id = data.get('video_id')
        interaction_type = data.get('interaction_type')

        if not all([user_id, video_id, interaction_type]):
            return jsonify({
                'error': 'Missing required fields',
                'required': ['user_id', 'video_id', 'interaction_type']
            }), 400

        recommendation_service.log_interaction(
            user_id=user_id,
            video_id=video_id,
            interaction_type=interaction_type
        )

        return jsonify({
            'status': 'success',
            'message': 'Interaction logged'
        }), 200

    except Exception as e:
        logger.error(f"Error logging interaction: {e}", exc_info=True)
        return jsonify({
            'error': 'Failed to log interaction',
            'message': str(e)
        }), 500


@app.route('/api/v1/train', methods=['POST'])
def train_models():
    """
    Trigger model training.

    Request Body:
        {
            "interactions": [[user_id, video_id, rating], ...],
            "videos": [{video metadata}, ...],
            "video_stats": {video_id: {stats}, ...}
        }

    Returns:
        Training status
    """
    try:
        data = request.get_json()

        interactions = data.get('interactions', [])
        videos = data.get('videos', [])
        video_stats = data.get('video_stats', {})

        # Convert interactions to tuples
        interactions = [tuple(i) for i in interactions]

        recommendation_service.train_models(
            interactions=interactions,
            videos=videos,
            video_stats=video_stats
        )

        return jsonify({
            'status': 'success',
            'message': 'Models trained successfully',
            'num_interactions': len(interactions),
            'num_videos': len(videos)
        }), 200

    except Exception as e:
        logger.error(f"Error training models: {e}", exc_info=True)
        return jsonify({
            'error': 'Failed to train models',
            'message': str(e)
        }), 500


@app.route('/api/v1/cache/clear', methods=['POST'])
def clear_cache():
    """Clear recommendation cache."""
    try:
        recommendation_service.clear_cache()
        return jsonify({
            'status': 'success',
            'message': 'Cache cleared'
        }), 200
    except Exception as e:
        logger.error(f"Error clearing cache: {e}", exc_info=True)
        return jsonify({
            'error': 'Failed to clear cache',
            'message': str(e)
        }), 500


@app.route('/api/v1/cache/stats', methods=['GET'])
def get_cache_stats():
    """Get cache statistics."""
    try:
        stats = recommendation_service.get_cache_stats()
        return jsonify(stats), 200
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}", exc_info=True)
        return jsonify({
            'error': 'Failed to get cache stats',
            'message': str(e)
        }), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        'error': 'Not found',
        'message': 'The requested resource was not found'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
