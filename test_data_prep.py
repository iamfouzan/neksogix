#!/usr/bin/env python3
"""
Test script to debug data preparation issues.
"""

from ml.data_preparation import DataPreparation
from scraper.imdb_spider import IMDbSpider

def test_data_preparation():
    """Test the data preparation pipeline."""
    print("Testing data preparation...")
    
    dp = DataPreparation()
    print(f"Famous movies: {len(dp.famous_movies)}")
    print(f"JSON movies: {len(dp.json_movies)}")
    
    # Test with first movie
    movie_name, movie_id = dp.famous_movies[0]
    print(f"\nTesting with movie: {movie_name} ({movie_id})")
    
    # Scrape movie reviews
    spider = IMDbSpider(movie_id=movie_id)
    movie_data = spider.scrape_movie_reviews(movie_id, 1)
    print(f"Scraped {len(movie_data)} reviews")
    
    if movie_data:
        print(f"First review text: {movie_data[0]['text'][:100]}...")
        
        # Process with BERT
        texts = [item['text'] for item in movie_data]
        bert_results = dp.bert_analyzer.predict(texts)
        print(f"BERT results: {len(bert_results)}")
        
        if bert_results:
            print(f"First BERT result: {bert_results[0]}")
            
            # Process data like in collect_training_data
            processed_data = []
            for i, item in enumerate(movie_data):
                bert_result = bert_results[i] if i < len(bert_results) else None
                
                if bert_result:
                    processed_item = {
                        'text': item['text'],
                        'cleaned_text': dp.preprocessor.preprocess_text(item['text']),
                        'rating': item.get('rating', 0),
                        'username': item.get('username', ''),
                        'bert_sentiment': bert_result['label'],
                        'bert_confidence': bert_result['confidence'],
                        'movie_name': movie_name
                    }
                    processed_data.append(processed_item)
            
            print(f"Processed {len(processed_data)} items")
            if processed_data:
                print(f"Sample processed item: {processed_data[0]}")
        else:
            print("No BERT results!")
    else:
        print("No movie data!")
    
    print("Test completed")

if __name__ == "__main__":
    test_data_preparation() 