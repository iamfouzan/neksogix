#!/usr/bin/env python3
"""
Simple test script to check IMDb scraping functionality.
"""

import requests
import logging
from bs4 import BeautifulSoup

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imdb_access():
    """Test if we can access IMDb and see the page structure."""
    
    # Test URLs
    test_urls = [
        "https://www.imdb.com/title/tt0111161/reviews",  # Shawshank Redemption
        "https://www.imdb.com/title/tt0111161/",  # Main page
        "https://www.imdb.com/find?q=The+Shawshank+Redemption&s=tt&ttype=ft"  # Search
    ]
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
    
    for url in test_urls:
        try:
            logger.info(f"Testing URL: {url}")
            response = requests.get(url, headers=headers, timeout=10)
            logger.info(f"Status code: {response.status_code}")
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Check page title
                title = soup.find('title')
                logger.info(f"Page title: {title.text if title else 'No title'}")
                
                # Check for review elements
                reviews = soup.find_all('div', class_='review-container')
                logger.info(f"Found {len(reviews)} review-container elements")
                
                # Check for other possible review selectors
                review_divs = soup.find_all('div', class_=lambda x: x and 'review' in x.lower())
                logger.info(f"Found {len(review_divs)} divs with 'review' in class")
                
                # Check for movie title
                movie_title = soup.find('h1')
                logger.info(f"Movie title: {movie_title.text if movie_title else 'No title found'}")
                
                # Check for links
                links = soup.find_all('a', href=True)
                review_links = [link for link in links if 'review' in link.get('href', '')]
                logger.info(f"Found {len(review_links)} links containing 'review'")
                
                # For reviews page, examine the structure more closely
                if '/reviews' in url:
                    logger.info("=== REVIEWS PAGE ANALYSIS ===")
                    
                    # Look for all div elements and their classes
                    all_divs = soup.find_all('div')
                    logger.info(f"Total div elements: {len(all_divs)}")
                    
                    # Check for common patterns
                    for div in all_divs[:20]:  # Check first 20 divs
                        class_attr = div.get('class', [])
                        if class_attr:
                            logger.info(f"Div class: {class_attr}")
                    
                    # Look for text content that might be reviews
                    text_elements = soup.find_all(['p', 'div'], text=True)
                    logger.info(f"Text elements found: {len(text_elements)}")
                    
                    # Check for specific patterns
                    for elem in text_elements[:10]:
                        text = elem.get_text().strip()
                        if len(text) > 50:  # Likely a review
                            logger.info(f"Potential review text: {text[:100]}...")
                
            else:
                logger.error(f"Failed to access {url}: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error testing {url}: {e}")

if __name__ == "__main__":
    test_imdb_access() 