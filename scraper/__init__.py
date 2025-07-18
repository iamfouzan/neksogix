"""
Scraper module for sentiment analysis system.

This module contains web scraping functionality for IMDb movie reviews
using Scrapy framework with proper error handling and rate limiting.
"""

from .imdb_spider import IMDbSpider

__all__ = ['IMDbSpider'] 