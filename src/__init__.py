"""
YouTube Video Recommendation System

A comprehensive recommendation system for suggesting personalized videos
to users on their YouTube homepage.
"""

__version__ = '1.0.0'
__author__ = 'YouTube Recommendation Team'

from . import models
from . import algorithms
from . import api
from . import utils

__all__ = ['models', 'algorithms', 'api', 'utils']
