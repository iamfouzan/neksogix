#!/usr/bin/env python3
"""
Convert IMDb TSV file to JSON format for movie training data.
"""

import json
import pandas as pd
import logging
from typing import List, Dict, Any
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_tsv_to_json(tsv_file: str, output_file: str = "data/movies_test.json"):
    """
    Convert IMDb TSV file to JSON format with popular movies from 2000+.
    
    Args:
        tsv_file (str): Path to the TSV file
        output_file (str): Path to output JSON file
    """
    try:
        logger.info(f"Reading TSV file: {tsv_file}")
        
        # Read the TSV file
        df = pd.read_csv(tsv_file, sep='\t', low_memory=False)
        
        logger.info(f"Loaded {len(df)} records from TSV file")
        logger.info(f"Columns: {list(df.columns)}")
        
        # Filter for movies only
        if 'titleType' in df.columns:
            movie_df = df[df['titleType'] == 'movie'].copy()
            logger.info(f"Filtered to {len(movie_df)} movies")
        else:
            movie_df = df.copy()
            logger.info("No titleType column found, using all records")
        
        # Filter for movies from 2000 and above, but exclude 2025 (future movies)
        if 'startYear' in df.columns:
            # Convert startYear to numeric, handling non-numeric values
            movie_df['startYear'] = pd.to_numeric(movie_df['startYear'], errors='coerce')
            
            # Filter for movies from 2000 to 2024 (exclude 2025)
            current_year = 2024  # Exclude 2025 to avoid future movies
            recent_movies = movie_df[
                (movie_df['startYear'] >= 2000) & 
                (movie_df['startYear'] <= current_year)
            ].copy()
            logger.info(f"Filtered to {len(recent_movies)} movies from 2000 to {current_year}")
        else:
            recent_movies = movie_df.copy()
            logger.warning("No startYear column found, using all movies")
        
        # Filter for popular movies (if rating/votes columns exist)
        if 'numVotes' in df.columns and 'averageRating' in df.columns:
            # Convert to numeric, handling non-numeric values
            recent_movies['numVotes'] = pd.to_numeric(recent_movies['numVotes'], errors='coerce')
            recent_movies['averageRating'] = pd.to_numeric(recent_movies['averageRating'], errors='coerce')
            
            popular_movies = recent_movies[
                (recent_movies['numVotes'] >= 1000) &  # At least 1000 votes
                (recent_movies['averageRating'] >= 5.0) &  # At least 5.0 rating
                (recent_movies['startYear'] >= 2000) &  # From 2000 onwards
                (recent_movies['startYear'] <= current_year)  # Up to 2024
            ].copy()
            
            # Sort by number of votes (popularity) and then by year (newer first)
            popular_movies = popular_movies.sort_values(['numVotes', 'startYear'], ascending=[False, False])
            
            # Take top 5000+ movies
            top_movies = popular_movies.head(6000)  # Get extra in case some fail
            logger.info(f"Selected {len(top_movies)} popular movies from 2000+")
        else:
            # If no rating data, get a better distribution of years
            # Group by year and take top movies from each year
            year_groups = []
            for year in range(2000, current_year + 1):
                year_movies = recent_movies[recent_movies['startYear'] == year].copy()
                if len(year_movies) > 0:
                    # Take top movies from each year (more recent years get more movies)
                    movies_per_year = max(50, 5000 // (current_year - 2000 + 1))  # Distribute evenly
                    year_groups.append(year_movies.head(movies_per_year))
            
            if year_groups:
                top_movies = pd.concat(year_groups, ignore_index=True)
                # Sort by year (newer first) and take top 5000
                top_movies = top_movies.sort_values('startYear', ascending=False).head(5000)
            else:
                # Fallback: just take movies from 2000+ sorted by year
                top_movies = recent_movies.sort_values('startYear', ascending=False).head(5000)
            
            logger.info(f"Selected {len(top_movies)} movies from 2000+ (no rating data available)")
        
        # Convert to list of dictionaries
        movies = []
        for _, row in top_movies.iterrows():
            movie = {
                'imdb_id': row['tconst'],
                'title': row['primaryTitle'] if 'primaryTitle' in row else row.get('title', 'Unknown'),
                'original_title': row.get('originalTitle', ''),
                'year': int(row['startYear']) if 'startYear' in row and pd.notna(row['startYear']) else None,
                'rating': float(row['averageRating']) if 'averageRating' in row and pd.notna(row['averageRating']) else None,
                'votes': int(row['numVotes']) if 'numVotes' in row and pd.notna(row['numVotes']) else None,
                'genres': row['genres'].split(',') if 'genres' in row and pd.notna(row['genres']) else []
            }
            movies.append(movie)
        
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        
        # Save to JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(movies, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Successfully converted {len(movies)} movies to {output_file}")
        
        # Print sample movies with year information
        logger.info("Sample movies (2000+):")
        for i, movie in enumerate(movies[:10]):
            year_info = f"({movie['year']})" if movie['year'] else "(Unknown year)"
            rating_info = f" - Rating: {movie['rating']}" if movie['rating'] else ""
            votes_info = f" - Votes: {movie['votes']}" if movie['votes'] else ""
            logger.info(f"  {i+1}. {movie['title']} {year_info}{rating_info}{votes_info} - {movie['imdb_id']}")
        
        # Print year distribution
        years = [movie['year'] for movie in movies if movie['year']]
        if years:
            logger.info(f"Year range: {min(years)} - {max(years)}")
            logger.info(f"Average year: {sum(years) / len(years):.1f}")
        
        return movies
        
    except Exception as e:
        logger.error(f"Error converting TSV to JSON: {e}")
        return None

def main():
    """Main function."""
    # You can change this path to match your downloaded file
    tsv_file = "title.basics.tsv"  # or "title.basics.tsv.gz" if compressed
    
    if not os.path.exists(tsv_file):
        logger.error(f"TSV file not found: {tsv_file}")
        logger.info("Please specify the correct path to your downloaded TSV file")
        return
    
    movies = convert_tsv_to_json(tsv_file)
    
    if movies:
        logger.info(f"Successfully created JSON with {len(movies)} movies from 2000+")
    else:
        logger.error("Failed to convert TSV to JSON")

if __name__ == "__main__":
    main() 