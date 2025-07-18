#!/usr/bin/env python3
"""
Test script to debug text extraction issues.
"""

from scraper.imdb_spider import IMDbSpider

def test_text_extraction():
    """Test the text extraction from reviews."""
    print("Testing text extraction...")
    
    spider = IMDbSpider(movie_id='tt0111161')
    movie_data = spider.scrape_movie_reviews('tt0111161', 1)
    
    print(f"Scraped {len(movie_data)} reviews")
    
    if movie_data:
        for i, review in enumerate(movie_data[:3]):  # Check first 3 reviews
            print(f"\nReview {i+1}:")
            print(f"Text length: {len(review['text'])}")
            print(f"Text: '{review['text'][:200]}...'")
            print(f"Username: {review['username']}")
            print(f"Rating: {review['rating']}")
            print(f"Date: {review['date']}")
    
    print("Test completed")

if __name__ == "__main__":
    test_text_extraction() 