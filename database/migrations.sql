-- Database migration script for sentiment analysis system
-- This script creates all necessary tables for the application

-- Create movies table
CREATE TABLE IF NOT EXISTS movies (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    imdb_id VARCHAR(20) NOT NULL UNIQUE,
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index on movies table
CREATE INDEX IF NOT EXISTS idx_movies_name ON movies(name);
CREATE INDEX IF NOT EXISTS idx_movies_imdb_id ON movies(imdb_id);

-- Create comments table
CREATE TABLE IF NOT EXISTS comments (
    id SERIAL PRIMARY KEY,
    movie_id INTEGER NOT NULL,
    text TEXT NOT NULL,
    cleaned_text TEXT,
    username VARCHAR(100),
    rating FLOAT,
    date TIMESTAMP,
    FOREIGN KEY (movie_id) REFERENCES movies(id) ON DELETE CASCADE
);

-- Create index on comments table
CREATE INDEX IF NOT EXISTS idx_comments_movie_id ON comments(movie_id);
CREATE INDEX IF NOT EXISTS idx_comments_username ON comments(username);
CREATE INDEX IF NOT EXISTS idx_comments_rating ON comments(rating);

-- Create predictions table
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    comment_id INTEGER NOT NULL,
    bert_sentiment VARCHAR(20),
    bert_confidence FLOAT,
    svm_sentiment VARCHAR(20),
    svm_confidence FLOAT,
    final_sentiment VARCHAR(20) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (comment_id) REFERENCES comments(id) ON DELETE CASCADE
);

-- Create index on predictions table
CREATE INDEX IF NOT EXISTS idx_predictions_comment_id ON predictions(comment_id);
CREATE INDEX IF NOT EXISTS idx_predictions_final_sentiment ON predictions(final_sentiment);
CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at);

-- Create composite index for sentiment analysis queries
CREATE INDEX IF NOT EXISTS idx_predictions_sentiment_confidence ON predictions(final_sentiment, bert_confidence, svm_confidence);

-- Add comments for table documentation
COMMENT ON TABLE movies IS 'Stores movie information including IMDb ID and processing timestamp';
COMMENT ON TABLE comments IS 'Stores movie review comments with text and metadata';
COMMENT ON TABLE predictions IS 'Stores sentiment analysis predictions from BERT and SVM models';

-- Create view for sentiment statistics
CREATE OR REPLACE VIEW sentiment_stats AS
SELECT 
    m.id as movie_id,
    m.name as movie_name,
    m.imdb_id,
    p.final_sentiment,
    COUNT(*) as sentiment_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY m.id), 2) as sentiment_percentage
FROM movies m
JOIN comments c ON m.id = c.movie_id
JOIN predictions p ON c.id = p.comment_id
GROUP BY m.id, m.name, m.imdb_id, p.final_sentiment
ORDER BY m.id, p.final_sentiment;

-- Create view for movie summary statistics
CREATE OR REPLACE VIEW movie_summary AS
SELECT 
    m.id,
    m.name,
    m.imdb_id,
    m.processed_at,
    COUNT(c.id) as total_comments,
    COUNT(p.id) as total_predictions,
    AVG(c.rating) as avg_rating,
    COUNT(CASE WHEN p.final_sentiment = 'positive' THEN 1 END) as positive_count,
    COUNT(CASE WHEN p.final_sentiment = 'negative' THEN 1 END) as negative_count,
    COUNT(CASE WHEN p.final_sentiment = 'neutral' THEN 1 END) as neutral_count
FROM movies m
LEFT JOIN comments c ON m.id = c.movie_id
LEFT JOIN predictions p ON c.id = p.comment_id
GROUP BY m.id, m.name, m.imdb_id, m.processed_at
ORDER BY m.processed_at DESC; 