"""
IMDb Spider for scraping movie reviews.

This module contains Scrapy spider for extracting movie reviews from IMDb
with proper pagination handling, rate limiting, and error handling.
"""

import scrapy
import re
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from urllib.parse import urljoin, quote_plus
import time
import random
import traceback
import os

from config import config
from utils.helpers import validate_text, sanitize_text, retry_on_error

logger = logging.getLogger(__name__)

USER_AGENTS = [
    # Chrome
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
    # Firefox
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0',
    # Edge
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0',
    # Safari
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15',
    # Linux Chrome
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
    # Linux Firefox
    'Mozilla/5.0 (X11; Linux x86_64; rv:126.0) Gecko/20100101 Firefox/126.0',
]


class IMDbSpider(scrapy.Spider):
    """Scrapy spider for scraping IMDb movie reviews."""
    
    name = 'imdb_spider'
    
    def __init__(self, movie_name: str = None, movie_id: str = None, reviews: list = None, *args, **kwargs):
        """Initialize spider with movie name or ID."""
        super(IMDbSpider, self).__init__(*args, **kwargs)
        self.movie_name = movie_name
        self.movie_id = movie_id
        self.reviews = reviews if reviews is not None else []
        self.start_urls = []
        
        if movie_id:
            # Direct movie ID provided
            self.start_urls = [f'https://www.imdb.com/title/{movie_id}/reviews']
        elif movie_name:
            # Search for movie first
            search_url = f'https://www.imdb.com/find?q={quote_plus(movie_name)}&s=tt&ttype=ft'
            self.start_urls = [search_url]
        else:
            raise ValueError("Either movie_name or movie_id must be provided")
    
    async def start(self):
        """Generate initial requests using modern async approach."""
        for url in self.start_urls:
            yield scrapy.Request(
                url=url,
                callback=self.parse,
                headers=self._get_headers(),
                meta={'dont_cache': True}
            )
    
    def _get_headers(self) -> Dict[str, str]:
        """Get random user agent headers."""
        user_agent = random.choice(config.USER_AGENTS)
        return {
            'User-Agent': user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
    
    def parse(self, response):
        """Parse the response based on URL type."""
        logger.info(f"Parsing URL: {response.url}")
        logger.info(f"Response status: {response.status}")
        logger.info(f"Response body length: {len(response.body)}")
        
        # Check for HTTP errors
        if response.status >= 400:
            logger.error(f"HTTP error {response.status} for URL: {response.url}")
            return
        
        # Check if response has content
        if not response.body:
            logger.error(f"Empty response for URL: {response.url}")
            return
        
        # Log page title for debugging
        page_title = response.css('title::text').get()
        logger.info(f"Page title: {page_title}")
        
        # Log some page content for debugging
        page_content = response.text[:1000] if response.text else ""
        logger.info(f"Page content preview: {page_content[:200]}...")
        
        # Check for common IMDb elements
        imdb_elements = response.css('div[class*="imdb"], div[class*="ipc"], article[class*="user-review"]').getall()
        logger.info(f"Found {len(imdb_elements)} potential IMDb elements")
        
        if 'find?' in response.url:
            # This is a search page, find the first movie
            logger.info("Detected search page")
            yield from self._parse_search_results(response)
        elif '/reviews' in response.url:
            # This is a reviews page
            logger.info("Detected reviews page")
            yield from self._parse_reviews_page(response)
        else:
            logger.error(f"Unexpected URL: {response.url}")
            # Try to extract any useful information
            logger.info(f"Page title: {response.css('title::text').get()}")
            logger.info(f"Available links: {response.css('a::attr(href)').getall()[:5]}")
    
    def _parse_search_results(self, response):
        """Parse search results to find movie ID."""
        try:
            # Updated selectors for IMDb's current structure
            movie_links = response.css('li.find-result-item a[href*="/title/tt"]::attr(href)').getall()
            
            if not movie_links:
                # Try alternative selector
                movie_links = response.css('a[href*="/title/tt"]::attr(href)').getall()
            
            if not movie_links:
                logger.error("No movie links found in search results")
                return
            
            # Get the first movie link (most relevant)
            movie_url = movie_links[0]
            movie_id = re.search(r'/title/(tt\d+)/', movie_url)
            
            if movie_id:
                movie_id = movie_id.group(1)
                reviews_url = f'https://www.imdb.com/title/{movie_id}/reviews'
                logger.info(f"Found movie ID: {movie_id}, proceeding to reviews")
                
                yield scrapy.Request(
                    url=reviews_url,
                    callback=self._parse_reviews_page,
                    headers=self._get_headers(),
                    meta={'movie_id': movie_id}
                )
            else:
                logger.error("Could not extract movie ID from search results")
                
        except Exception as e:
            logger.error(f"Error parsing search results: {e}")
    
    def _parse_reviews_page(self, response):
        """Parse reviews page and extract review data using new IMDb structure."""
        try:
            movie_title = response.css('h1[data-testid="hero-title-block__title"]::text, h1::text').get()
            movie_id = response.meta.get('movie_id')
            if not movie_id:
                movie_id_match = re.search(r'/title/(tt\d+)/', response.url)
                movie_id = movie_id_match.group(1) if movie_id_match else None
            logger.info(f"Parsing reviews for movie: {movie_title} (ID: {movie_id})")

            # Use the new article selector for reviews based on the image
            reviews = response.css('article.user-review-item, article.sc-a77dbebd-1.iJQoqi.user-review-item, article[class*="user-review-item"]')
            logger.info(f"Found {len(reviews)} review articles with primary selectors")
            
            # If no reviews found, try alternative selectors
            if not reviews:
                logger.warning("No reviews found with primary selectors, trying alternatives...")
                reviews = response.css('div[data-testid="review-card-parent"], div[class*="review-item"], div[class*="user-review"]')
                logger.info(f"Found {len(reviews)} reviews with alternative selectors")
            
            # If still no reviews, try fallback method
            if not reviews:
                logger.warning("No reviews found with any selectors, trying fallback method...")
                reviews = self._extract_reviews_fallback(response)
                logger.info(f"Found {len(reviews)} reviews with fallback method")
            
            # If still no reviews, try to find any text content
            if not reviews:
                logger.warning("No reviews found with any method. Checking page content...")
                all_text = response.css('*::text').getall()
                logger.info(f"Total text elements found: {len(all_text)}")
                if all_text:
                    logger.info(f"Sample text elements: {all_text[:5]}")
                
                # Try to find any div that might contain reviews
                all_divs = response.css('div').getall()
                logger.info(f"Total div elements found: {len(all_divs)}")
                
                # Look for any text that might be a review
                substantial_text = [text for text in all_text if len(text.strip()) > 100]
                logger.info(f"Substantial text blocks found: {len(substantial_text)}")
                if substantial_text:
                    logger.info(f"Sample substantial text: {substantial_text[0][:200]}...")

            for review in reviews:
                review_data = self._extract_review_data_article(review, movie_id)
                if review_data:
                    self.reviews.append(review_data)
                    yield review_data

            # Check for "Load More" button and handle pagination
            load_more_url = self._get_load_more_url(response)
            if load_more_url:
                logger.info(f"Found load more URL: {load_more_url}")
                yield scrapy.Request(
                    url=load_more_url,
                    callback=self._parse_reviews_page,
                    headers=self._get_headers(),
                    meta={'movie_id': movie_id}
                )
        except Exception as e:
            logger.error(f"Error parsing reviews page: {e}")

    def _extract_review_data_article(self, article, movie_id: str) -> Optional[Dict[str, Any]]:
        """Extract review data from a review <article> element (new IMDb structure)."""
        try:
            review_text = ''
            # Try primary selectors based on the new structure
            text_selectors = [
                '[data-testid="review-summary"]::text',
                '[data-testid="review-summary"] *::text',
                'div[data-testid="review-summary"]::text',
                'div[data-testid="review-summary"] *::text',
                'div.ipc-title::text',
                'div.ipc-title *::text',
                'div[class*="ipc-title"]::text',
                'div[class*="ipc-title"] *::text',
                'div[class*="review"]::text',
                'div[class*="review"] *::text',
                'p::text',
                'div.review-content::text',
                'div.review-content *::text',
                'span[data-testid="review-text"]::text',
                'span[data-testid="review-text"] *::text'
            ]
            for selector in text_selectors:
                text_parts = article.css(selector).getall()
                if text_parts:
                    review_text = ' '.join([t.strip() for t in text_parts if t.strip()])
                    if review_text:
                        break
            if not review_text:
                all_text = article.css('*::text').getall()
                review_text = ' '.join([t.strip() for t in all_text if t.strip() and len(t.strip()) > 10])
            if review_text:
                review_text = review_text.strip()
                skip_phrases = ['helpful', 'unhelpful', 'report', 'sign in', 'register', 'menu', 'navigation']
                for phrase in skip_phrases:
                    if phrase in review_text.lower():
                        review_text = review_text.replace(phrase, '').strip()

            # --- Enhanced Username Extraction ---
            username = None
            username_selectors = [
                # Most specific selectors first
                'span[data-testid="author-name"]::text',
                'a[data-testid="review-author"]::text',
                'div[data-testid="reviews-author"] *::text',
                'span[class*="author"]::text',
                'span.username::text',
                'span.user::text',
                'span::text',
            ]
            for selector in username_selectors:
                candidates = [u.strip() for u in article.css(selector).getall() if u.strip()]
                # Filter out numbers-only usernames (e.g., "10") unless nothing else is found
                filtered = [u for u in candidates if not u.isdigit() and len(u) > 1]
                if filtered:
                    username = filtered[0]
                    break
                elif candidates and not username:
                    username = candidates[0]  # fallback to first if nothing else
            if not username:
                username = 'Anonymous'

            # --- Enhanced Rating Extraction ---
            rating = None
            rating_selectors = [
                '[data-testid="review-rating"]::text',
                'span[aria-label*="star"]::attr(aria-label)',
                'div[class*="rating"]::text',
                'span.rating-other-user-rating span::text',
                'span[class*="starRating"]::text',
            ]
            for selector in rating_selectors:
                rating_texts = [r for r in article.css(selector).getall() if r.strip()]
                for rating_text in rating_texts:
                    # Try to extract a number (float or int)
                    match = re.search(r'(\d+(?:\.\d+)?)', rating_text)
                    if match:
                        try:
                            rating = float(match.group(1))
                            break
                        except Exception:
                            continue
                if rating is not None:
                    break

            # --- Enhanced Date Extraction ---
            date = None
            date_selectors = [
                'span[data-testid="review-date"]::text',
                'span[class*="date"]::text',
                'div[class*="date"]::text',
                'span.review-date::text',
                'span.date::text',
            ]
            date_text = None
            for selector in date_selectors:
                date_texts = [d for d in article.css(selector).getall() if d.strip()]
                if date_texts:
                    date_text = date_texts[0].strip()
                    break
            if date_text:
                try:
                    # Try multiple date formats
                    for fmt in [
                        '%d %B %Y', '%B %d, %Y', '%B %d %Y', '%d %b %Y', '%b %d, %Y', '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y'
                    ]:
                        try:
                            date = datetime.strptime(date_text, fmt)
                            break
                        except Exception:
                            continue
                    if not date:
                        year_match = re.search(r'(\d{4})', date_text)
                        if year_match:
                            date = datetime(int(year_match.group(1)), 1, 1)
                except Exception as e:
                    logger.warning(f"Could not parse date '{date_text}': {e}")

            if len(review_text) < 10:
                logger.warning(f"Review text too short ({len(review_text)} chars), skipping")
                return None

            return {
                'movie_id': movie_id,
                'text': review_text,
                'username': username,
                'rating': rating,
                'date': date.isoformat() if date else None,
                'scraped_at': datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Error extracting review data from article: {e}")
            return None
    
    def _extract_reviews_fallback(self, response):
        """Fallback method to extract reviews when CSS selectors fail."""
        try:
            # Get all text content
            all_text = response.css('*::text').getall()
            
            if not all_text:
                logger.warning("No text content found in response")
                return []
            
            # Find substantial text blocks (likely reviews)
            reviews = []
            current_text = ""
            
            for text in all_text:
                text = text.strip()
                if len(text) > 50:  # Substantial text block
                    # Check if it looks like a review (not navigation, not short)
                    if not any(skip in text.lower() for skip in ['menu', 'navigation', 'search', 'imdb', 'sign in', 'register']):
                        current_text += " " + text
                elif current_text and len(current_text.strip()) > 100:
                    # We have a substantial text block, create a review element
                    review_div = type('obj', (object,), {
                        'css': lambda self, selector: type('obj', (object,), {
                            'getall': lambda self: [current_text.strip()],
                            'get': lambda self: current_text.strip()
                        })()
                    })()
                    reviews.append(review_div)
                    current_text = ""
            
            # Add the last review if it exists
            if current_text and len(current_text.strip()) > 100:
                review_div = type('obj', (object,), {
                    'css': lambda self, selector: type('obj', (object,), {
                        'getall': lambda self: [current_text.strip()],
                        'get': lambda self: current_text.strip()
                    })()
                })()
                reviews.append(review_div)
            
            logger.info(f"Fallback method found {len(reviews)} potential reviews")
            return reviews
            
        except Exception as e:
            logger.error(f"Error in fallback review extraction: {e}")
            return []
    
    def _get_load_more_url(self, response) -> Optional[str]:
        """Extract load more URL for pagination."""
        try:
            # Look for load more button with updated selectors for new IMDb structure
            load_more_selectors = [
                'button[data-testid="load-more-button"]::attr(data-key)',
                'button.load-more-button::attr(data-key)',
                'button.load-more::attr(data-key)',
                'button[class*="load-more"]::attr(data-key)',
                'a[data-testid="load-more"]::attr(href)',
                'a[href*="paginationKey"]::attr(href)',
                'button[aria-label*="Load more"]::attr(data-key)',
                'button[aria-label*="load more"]::attr(data-key)'
            ]
            
            for selector in load_more_selectors:
                load_more_button = response.css(selector).get()
                if load_more_button:
                    logger.info(f"Found load more button with selector: {selector}")
                    # Construct load more URL
                    current_url = response.url
                    if '?' in current_url:
                        load_more_url = f"{current_url}&paginationKey={load_more_button}"
                    else:
                        load_more_url = f"{current_url}?paginationKey={load_more_button}"
                    return load_more_url
            
            # Alternative: look for "Load More" link
            load_more_link = response.css('a[href*="paginationKey"]::attr(href)').get()
            if load_more_link:
                return urljoin(response.url, load_more_link)
            
            # Check for any button that might be a load more button
            all_buttons = response.css('button::text').getall()
            load_more_buttons = [btn for btn in all_buttons if 'load' in btn.lower() and 'more' in btn.lower()]
            if load_more_buttons:
                logger.info(f"Found potential load more buttons: {load_more_buttons}")
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting load more URL: {e}")
            return None
    
    def closed(self, reason):
        """Called when spider is closed."""
        logger.info(f"Spider closed: {reason}")
        logger.info(f"Total reviews scraped: {len(self.reviews)}")
        
        # Save results to file
        if self.reviews:
            self._save_results()
    
    def _save_results(self):
        """Save scraped results to CSV file by appending to existing file."""
        try:
            import pandas as pd
            import os
            
            df = pd.DataFrame(self.reviews)
            
            # Check if file exists to determine if we need headers
            file_exists = os.path.exists(config.RAW_COMMENTS_FILE)
            
            # Append to existing file or create new one
            df.to_csv(
                config.RAW_COMMENTS_FILE, 
                mode='a' if file_exists else 'w',
                header=not file_exists,  # Only write header if file doesn't exist
                index=False, 
                encoding='utf-8'
            )
            
            logger.info(f"Saved {len(self.reviews)} reviews to {config.RAW_COMMENTS_FILE} (appended: {file_exists})")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")

    def scrape_movie_reviews(self, movie_id: str, min_comments: int = 100) -> list:
        """Scrape reviews for a given movie ID using a subprocess to avoid Twisted reactor issues."""
        import tempfile
        import subprocess
        import json
        import sys
        import os

        logger.info(f"Starting to scrape reviews for movie ID: {movie_id}")
        reviews = []
        try:
            # Write a temporary script to run the crawl
            with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as script_file, \
                 tempfile.NamedTemporaryFile('w+', suffix='.json', delete=False) as result_file:
                script_path = script_file.name
                result_path = result_file.name
                script_file.write(f'''
import sys
import json
import logging
from scraper.imdb_spider import IMDbSpider
logging.basicConfig(level=logging.INFO)
spider = IMDbSpider(movie_id="{movie_id}")
reviews = []
try:
    reviews = spider._scrape_single_movie()
except Exception as e:
    logging.error(f"Subprocess error: {{e}}")
with open(r"{result_path}", "w", encoding="utf-8") as f:
    json.dump(reviews, f, ensure_ascii=False)
''')
                script_file.flush()

                # Set cwd and PYTHONPATH for subprocess
                project_root = os.path.abspath(os.path.dirname(__file__) + '/../')
                env = os.environ.copy()
                env['PYTHONPATH'] = project_root + (':' + env['PYTHONPATH'] if 'PYTHONPATH' in env else '')
                subprocess.run([sys.executable, script_path], check=False, cwd=project_root, env=env)

                # Read the results
                result_file.seek(0)
                try:
                    with open(result_path, 'r', encoding='utf-8') as f:
                        reviews = json.load(f)
                except Exception as e:
                    logger.error(f"Failed to load reviews from subprocess: {e}")
                finally:
                    os.unlink(script_path)
                    os.unlink(result_path)
        except Exception as e:
            logger.error(f"Error scraping reviews for {movie_id} in subprocess: {e}\n{traceback.format_exc()}")
            return []
        logger.info(f"Scraped {len(reviews)} reviews for movie ID: {movie_id}")
        return reviews

    def _scrape_single_movie(self) -> list:
        """Internal: Scrape reviews for a single movie in a subprocess-safe way."""
        from scrapy.crawler import CrawlerProcess
        from scrapy.utils.project import get_project_settings
        import tempfile
        import shutil
        import os
        reviews = []
        try:
            user_agent = random.choice(USER_AGENTS)
            settings = get_project_settings()
            settings.set('USER_AGENT', user_agent)
            settings.set('ROBOTSTXT_OBEY', False)
            settings.set('DOWNLOAD_DELAY', 4)
            settings.set('CONCURRENT_REQUESTS', 1)
            settings.set('LOG_LEVEL', 'INFO')
            settings.set('DOWNLOAD_TIMEOUT', 30)
            settings.set('RETRY_TIMES', 3)
            settings.set('COOKIES_ENABLED', False)
            settings.set('HTTPCACHE_ENABLED', False)
            temp_dir = tempfile.mkdtemp()
            settings.set('JOBDIR', temp_dir)
            class CollectorSpider(self.__class__):
                custom_settings = dict(settings)
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.collected_reviews = []
                def process_review(self, review):
                    self.collected_reviews.append(review)
                def _save_reviews(self, reviews):
                    for r in reviews:
                        self.process_review(r)
            process = CrawlerProcess(settings)
            process.crawl(CollectorSpider, movie_id=self.movie_id)
            process.start()
            reviews = self.reviews if hasattr(self, 'reviews') else []
            shutil.rmtree(temp_dir)
        except Exception as e:
            logger.error(f"Error in _scrape_single_movie: {e}\n{traceback.format_exc()}")
        return reviews


class IMDbSearchSpider(scrapy.Spider):
    """Spider for searching movies on IMDb."""
    
    name = 'imdb_search'
    
    def __init__(self, movie_name: str, *args, **kwargs):
        """Initialize search spider."""
        super(IMDbSearchSpider, self).__init__(*args, **kwargs)
        self.movie_name = movie_name
        self.start_urls = [f'https://www.imdb.com/find?q={quote_plus(movie_name)}&s=tt&ttype=ft']
    
    async def start(self):
        """Generate initial requests using modern async approach."""
        for url in self.start_urls:
            yield scrapy.Request(
                url=url,
                callback=self.parse,
                headers=self._get_headers()
            )
    
    def _get_headers(self) -> Dict[str, str]:
        """Get random user agent headers."""
        user_agent = random.choice(config.USER_AGENTS)
        return {
            'User-Agent': user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
    
    def parse(self, response):
        """Parse search results."""
        try:
            # Extract movie information
            movies = []
            
            # Look for movie results with updated selectors
            movie_elements = response.css('li.find-result-item, div.find-result-item')
            
            for element in movie_elements[:5]:  # Get top 5 results
                movie_data = self._extract_movie_data(element)
                if movie_data:
                    movies.append(movie_data)
            
            logger.info(f"Found {len(movies)} movies for '{self.movie_name}'")
            
            # Return the first (most relevant) movie
            if movies:
                return movies[0]
            else:
                logger.error(f"No movies found for '{self.movie_name}'")
                return None
                
        except Exception as e:
            logger.error(f"Error parsing search results: {e}")
            return None
    
    def _extract_movie_data(self, element) -> Optional[Dict[str, Any]]:
        """Extract movie data from search result element."""
        try:
            # Extract movie link and ID with updated selectors
            link = element.css('a[href*="/title/tt"]::attr(href)').get()
            if not link:
                return None
            
            movie_id_match = re.search(r'/title/(tt\d+)/', link)
            if not movie_id_match:
                return None
            
            movie_id = movie_id_match.group(1)
            
            # Extract movie title with updated selectors
            title = element.css('a[href*="/title/tt"]::text').get()
            if not title:
                title = element.css('a[href*="/title/tt"] *::text').getall()
                title = ' '.join(title) if title else 'Unknown'
            
            title = sanitize_text(title)
            
            # Extract year with updated selectors
            year_text = element.css('span.result-item-year::text, span.year::text').get()
            year = None
            if year_text:
                year_match = re.search(r'(\d{4})', year_text)
                if year_match:
                    year = int(year_match.group(1))
            
            return {
                'imdb_id': movie_id,
                'title': title,
                'year': year,
                'url': f'https://www.imdb.com{link}'
            }
            
        except Exception as e:
            logger.error(f"Error extracting movie data: {e}")
            return None 