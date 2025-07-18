#!/usr/bin/env python3
"""
Test script to verify IMDb selectors work with the new structure.

This script tests the updated selectors against a real IMDb movie page
to ensure they can extract reviews correctly.
"""

import logging
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scraper.imdb_spider import IMDbSpider

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imdb_selectors(movie_id="tt0111161"):
    """Test the IMDb selectors with a known movie."""
    try:
        logger.info(f"Testing IMDb selectors with movie ID: {movie_id}")
        
        # Create spider instance and test scraping
        spider = IMDbSpider(movie_id=movie_id)
        reviews = spider.scrape_movie_reviews(movie_id, min_comments=5)
        
        logger.info(f"Extracted {len(reviews)} reviews")
        
        if reviews:
            logger.info("Sample reviews:")
            for i, review in enumerate(reviews[:3]):
                logger.info(f"\nReview {i+1}:")
                logger.info(f"Text: {review.get('text', 'N/A')[:100]}...")
                logger.info(f"Username: {review.get('username', 'N/A')}")
                logger.info(f"Rating: {review.get('rating', 'N/A')}")
                logger.info(f"Date: {review.get('date', 'N/A')}")
            
            return True
        else:
            logger.error("No reviews were extracted - selectors may need updating")
            return False
        
    except Exception as e:
        logger.error(f"Error testing IMDb selectors: {e}")
        return False

if __name__ == "__main__":
    # Test with The Shawshank Redemption (known to have many reviews)
    success = test_imdb_selectors("tt0111161")
    
    if success:
        logger.info("✅ IMDb selectors test PASSED")
    else:
        logger.error("❌ IMDb selectors test FAILED")
        sys.exit(1) 