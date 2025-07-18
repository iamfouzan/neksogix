# Scrapy settings for IMDb review scraper

BOT_NAME = 'imdb_scraper'

SPIDER_MODULES = ['scraper']
NEWSPIDER_MODULE = 'scraper'

# User agent rotation
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:91.0) Gecko/20100101',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15'
]

# Download delay and concurrency
DOWNLOAD_DELAY = 2.0
CONCURRENT_REQUESTS = 1

# Retry settings
RETRY_ENABLED = True
RETRY_TIMES = 3
RETRY_HTTP_CODES = [429, 403, 500, 502, 503, 504]

# Disable cookies
COOKIES_ENABLED = False

# Obey robots.txt rules
ROBOTSTXT_OBEY = False

# Logging
LOG_LEVEL = 'INFO'

# Export encoding
FEED_EXPORT_ENCODING = 'utf-8' 