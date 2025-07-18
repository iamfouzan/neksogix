#!/usr/bin/env python3
"""
Test script for single movie scraping.
"""

import logging
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scraper.imdb_spider import IMDbSpider
from config import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_single_movie():
    """Test scraping a single movie."""
    
    # Test with Shawshank Redemption
    movie_id = "tt0111161"
    movie_name = "The Shawshank Redemption"
    
    logger.info(f"Testing scraper with movie: {movie_name} (ID: {movie_id})")
    
    try:
        # Create spider instance
        spider = IMDbSpider(movie_id=movie_id)
        
        # Scrape reviews
        reviews = spider.scrape_movie_reviews(movie_id, min_comments=10)
        
        logger.info(f"Successfully scraped {len(reviews)} reviews")
        
        # Print first few reviews
        for i, review in enumerate(reviews[:3]):
            logger.info(f"Review {i+1}:")
            logger.info(f"  Text: {review.get('text', '')[:100]}...")
            logger.info(f"  Username: {review.get('username', 'Unknown')}")
            logger.info(f"  Rating: {review.get('rating', 'N/A')}")
            logger.info(f"  Date: {review.get('date', 'N/A')}")
            logger.info("---")
        
        return len(reviews) > 0
        
    except Exception as e:
        logger.error(f"Error testing scraper: {e}")
        return False

if __name__ == "__main__":
    success = test_single_movie()
    if success:
        logger.info("Test passed!")
    else:
        logger.error("Test failed!") 