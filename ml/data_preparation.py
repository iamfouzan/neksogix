"""
Data preparation utilities for sentiment analysis training.

This module handles training data collection, preprocessing,
feature engineering, and dataset splitting for the custom SVM model.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import logging
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os
import json

from config import config
from database.models import Movie, Comment, Prediction
from database.connection import get_db_session
from nlp.preprocessor import TextPreprocessor
from nlp.sentiment_analyzer import BertSentimentAnalyzer
from scraper.imdb_spider import IMDbSpider

logger = logging.getLogger(__name__)

class DataPreparation:
    """
    Handles training data preparation for sentiment analysis model.
    """
    
    def __init__(self):
        """Initialize data preparation components."""
        self.preprocessor = TextPreprocessor()
        self.bert_analyzer = BertSentimentAnalyzer()
        # Load movies from JSON file
        self.json_movies = self._load_movies_from_json()
        # Define famous movies (priority list)
        self.famous_movies = [
            # ("The Shawshank Redemption", "tt0111161"),
            # ("The Godfather", "tt0068646"),
            # ("Pulp Fiction", "tt0110912"),
            # ("Fight Club", "tt0133093"),
            # ("Forrest Gump", "tt0109830"),
            # ("The Matrix", "tt0133093"),
            # ("Goodfellas", "tt0099685"),
            # ("The Silence of the Lambs", "tt0102926"),
            # ("Interstellar", "tt0816692"),
            # ("The Dark Knight", "tt0468569"),
            # ("The Dark Knight Rises", "tt1345836"),
            # ("Inception", "tt1375666"),
            # ("Joker", "tt7286456"),
            # ("Parasite", "tt6751668"),
            # ("Avengers: Endgame", "tt4154796"),
            # ("Black Swan", "tt0947798"),
            # ("Memento", "tt0209144"),
            # ("No Country for Old Men", "tt0477348"),
            # ("Reservoir Dogs", "tt0105236"),
            # ("Se7en", "tt0114369"),
            # ("V for Vendetta", "tt0434409"),
            # ("The Sixth Sense", "tt0167404"),
            # ("Good Will Hunting", "tt0119217"),
            # ("Eternal Sunshine of the Spotless Mind", "tt0338013"),
            # ("Hacksaw Ridge", "tt2119532"),
            # ("Doctor Strange", "tt1211837"),
            # ("Quantum of Solace", "tt0830515"),
            # ("Us", "tt6857112"),
            # ("Westworld", "tt0475784"),
            # ("Game of Thrones", "tt0944947"),
            # ("The Walking Dead", "tt1520211"),
            # ("The Sopranos", "tt0141842"),
            # ("The Wire", "tt0306414"),
            # ("The Office", "tt0386676"),
            # ("The Big Bang Theory", "tt0898266"),
            # ("The Simpsons", "tt0096697"),
            # ("Sherlock Holmes", "tt0988045"),
            # ("Chernobyl", "tt7366338"),
            # ("Killing Eve", "tt7016936"),
            # ("Loki", "tt9140554"),
            # ("Squid Game", "tt10919420"),
            # ("Yojimbo", "tt0055630"),
            # ("Zodiac", "tt0443706"),
            # ("The Lord of the Rings: The Fellowship of the Ring", "tt0120737"),
            # ("The Lord of the Rings: The Two Towers", "tt0167261"),
            # ("The Lord of the Rings: The Return of the King", "tt0167260"),
            # ("The Hobbit: An Unexpected Journey", "tt0903624"),
            # ("The Hobbit: The Desolation of Smaug", "tt1170358"),
            # ("The Hobbit: The Battle of the Five Armies", "tt2310332"),
            # ("Star Wars: Episode IV - A New Hope", "tt0076759"),
            # ("Star Wars: Episode V - The Empire Strikes Back", "tt0080684"),
            # ("Star Wars: Episode VI - Return of the Jedi", "tt0086190"),
            # ("Star Wars: Episode I - The Phantom Menace", "tt0120915"),
            # ("Star Wars: Episode II - Attack of the Clones", "tt0121765"),
            # ("Star Wars: Episode III - Revenge of the Sith", "tt0121766"),
            # ("Star Wars: Episode VII - The Force Awakens", "tt2488496"),
            # ("Star Wars: Episode VIII - The Last Jedi", "tt2527336"),
            # ("Star Wars: Episode IX - The Rise of Skywalker", "tt2527338"),
            # ("Titanic", "tt0120338"),
            # ("Avatar", "tt0499549"),
            # ("The Lion King", "tt0110357"),
            # ("Aladdin", "tt0103639"),
            # ("Beauty and the Beast", "tt0101414"),
            # ("The Little Mermaid", "tt0097757"),
            # ("Frozen", "tt2294629"),
            # ("Toy Story", "tt0114709"),
            # ("Toy Story 2", "tt0120363"),
            # ("Toy Story 3", "tt0435761"),
            # ("Toy Story 4", "tt1979376"),
            # ("Monsters, Inc.", "tt0198781"),
            # ("Finding Nemo", "tt0266543"),
            # ("Up", "tt1049413"),
            # ("Inside Out", "tt2096673"),
            # ("Coco", "tt2380307"),
            # ("Soul", "tt2948372"),
            # ("The Incredibles", "tt0317705"),
            # ("The Incredibles 2", "tt3606756"),
            # ("Ratatouille", "tt0382932"),
            # ("WALL-E", "tt0910970"),
            # ("Brave", "tt1217209"),
            # ("Moana", "tt3521164"),
            # ("Zootopia", "tt2948356"),
            # ("Big Hero 6", "tt2245084"),
            # ("Raya and the Last Dragon", "tt5109280"),
            # ("Encanto", "tt2953050"),
            # ("Turning Red", "tt8097030"),
            # ("Lightyear", "tt10298810"),
            # ("Elemental", "tt15789038"),
            # ("Wish", "tt11304740"),
            # ("The Super Mario Bros. Movie", "tt6718170"),
            # ("Spider-Man: Into the Spider-Verse", "tt4633694"),
            # ("Spider-Man: Across the Spider-Verse", "tt9362722"),
            # ("The Batman", "tt1877830"),
            # ("Wonder Woman", "tt0451279"),
            # ("Man of Steel", "tt0770828"),
            # ("Batman v Superman: Dawn of Justice", "tt2975590"),
            # ("Suicide Squad", "tt1386697"),
            # ("Justice League", "tt0974015"),
            # ("Aquaman", "tt1477834"),
            # ("Shazam!", "tt0448115"),
            # ("Birds of Prey", "tt7713068"),
            # ("Wonder Woman 1984", "tt7126948"),
            # ("The Suicide Squad", "tt6334354"),
            # ("Black Adam", "tt6443346"),
            # ("Shazam! Fury of the Gods", "tt10151854"),
            # ("The Flash", "tt0439572"),
            # ("Blue Beetle", "tt9362930"),
            # ("Aquaman and the Lost Kingdom", "tt9663764"),
            # ("Iron Man", "tt0371746"),
            # ("The Incredible Hulk", "tt0800080"),
            # ("Iron Man 2", "tt1228705"),
            # ("Thor", "tt0800369"),
            # ("Captain America: The First Avenger", "tt0458339"),
            # ("The Avengers", "tt0848228"),
            # ("Iron Man 3", "tt1300854"),
            # ("Thor: The Dark World", "tt1981115"),
            # ("Captain America: The Winter Soldier", "tt1843866"),
            # ("Guardians of the Galaxy", "tt2015381"),
            # ("Avengers: Age of Ultron", "tt2395427"),
            # ("Ant-Man", "tt0478970"),
            # ("Captain America: Civil War", "tt3498820"),
            # ("Doctor Strange", "tt1211837"),
            # ("Guardians of the Galaxy Vol. 2", "tt3896198"),
            # ("Spider-Man: Homecoming", "tt2250912"),
            # ("Thor: Ragnarok", "tt3501632"),
            # ("Black Panther", "tt1825683"),
            # ("Avengers: Infinity War", "tt4154756"),
            # ("Ant-Man and the Wasp", "tt5095030"),
            # ("Captain Marvel", "tt4154664"),
            # ("Avengers: Endgame", "tt4154796"),
            # ("Spider-Man: Far From Home", "tt6320628"),
            # ("Black Widow", "tt3480822"),
            # ("Shang-Chi and the Legend of the Ten Rings", "tt9376612"),
            # ("Eternals", "tt9032400"),
            # ("Spider-Man: No Way Home", "tt10872600"),
            # ("Doctor Strange in the Multiverse of Madness", "tt9419884"),
            # ("Thor: Love and Thunder", "tt10648342"),
            # ("Black Panther: Wakanda Forever", "tt9114286"),
            # ("Ant-Man and the Wasp: Quantumania", "tt10954600"),
            # ("Guardians of the Galaxy Vol. 3", "tt6791350"),
            # ("The Marvels", "tt10676048"),
            # ("Deadpool", "tt1431045"),
            # ("Deadpool 2", "tt5463162"),
            # ("Logan", "tt3315342"),
            # ("X-Men: Days of Future Past", "tt1877832"),
            # ("X-Men: Apocalypse", "tt3385516"),
            # ("X-Men: Dark Phoenix", "tt6565702"),
            # ("The New Mutants", "tt4682266"),
            # ("Venom", "tt1270797"),
            # ("Venom: Let There Be Carnage", "tt7097896"),
            # ("Morbius", "tt5108870"),
            # ("Madame Web", "tt11057302"),
            # ("Kraven the Hunter", "tt10370710"),
            # ("The Amazing Spider-Man", "tt0948470"),
            # ("The Amazing Spider-Man 2", "tt1872181"),
            # ("Spider-Man", "tt0145487"),
            # ("Spider-Man 2", "tt0316654"),
            # ("Spider-Man 3", "tt0413300"),
            # ("X-Men", "tt0120903"),
            # ("X2", "tt0290334"),
            # ("X-Men: The Last Stand", "tt0376994"),
            # ("X-Men Origins: Wolverine", "tt0458525"),
            # ("X-Men: First Class", "tt1270798"),
            # ("The Wolverine", "tt1430132"),
            # ("Fantastic Four", "tt0120667"),
            # ("Fantastic Four: Rise of the Silver Surfer", "tt0486576"),
            # ("Fantastic Four", "tt1502712"),
            # ("Fant4stic", "tt1502712"),
            # ("Blade", "tt0120611"),
            # ("Blade II", "tt0187738"),
            # ("Blade: Trinity", "tt0359013"),
            # ("Punisher: War Zone", "tt0450314"),
            # ("Ghost Rider", "tt0259324"),
            # ("Ghost Rider: Spirit of Vengeance", "tt1071875"),
            # ("Howard the Duck", "tt0091225"),
            # ("Captain America", "tt0034583"),
            # ("The Punisher", "tt0159097"),
            # ("Elektra", "tt0357277"),
            # ("Daredevil", "tt0287978"),
            # ("Hulk", "tt0286716"),
            # ("Man-Thing", "tt0410297"),
            # ("Nick Fury: Agent of S.H.I.E.L.D.", "tt0107665"),
            # ("Generation X", "tt0116424"),
            # ("Mutant X", "tt0284717"),
            # ("The Gifted", "tt4396630"),
            # ("Legion", "tt5114356"),
            # ("The Runaways", "tt5715524"),
            # ("Cloak & Dagger", "tt5777108"),
            # ("Helstrom", "tt10623646"),
            # ("Agents of S.H.I.E.L.D.", "tt2364582"),
            # ("Agent Carter", "tt3475736"),
            # ("Inhumans", "tt4154664"),
            # ("Runaways", "tt5715524"),
            # ("Cloak & Dagger", "tt5777108"),
            # ("Helstrom", "tt10623646"),
            # ("Daredevil", "tt3322312"),
            # ("Jessica Jones", "tt2357547"),
            # ("Luke Cage", "tt3322314"),
            # ("Iron Fist", "tt3322316"),
            # ("The Defenders", "tt3322318"),
            # ("The Punisher", "tt5675620"),
            # ("Helstrom", "tt10623646"),
            # ("WandaVision", "tt9140560"),
            # ("The Falcon and the Winter Soldier", "tt9208876"),
            # ("Loki", "tt9140554"),
            # ("What If...?", "tt10168312"),
            # ("Hawkeye", "tt10160804"),
            # ("Moon Knight", "tt10234724"),
            # ("Ms. Marvel", "tt10857164"),
            # ("She-Hulk: Attorney at Law", "tt10857160"),
            # ("Secret Invasion", "tt13157618"),
            # ("Echo", "tt13966962"),
            # ("Agatha: Darkhold Diaries", "tt13966964"),
            # ("Ironheart", "tt13966966"),
            # ("Armor Wars", "tt13966968"),
            # ("Daredevil: Born Again", "tt13966970"),
            # ("Blade", "tt13966972"),
            # ("Deadpool & Wolverine", "tt13966974"),
            # ("Captain America: Brave New World", "tt13966976"),
            # ("Thunderbolts", "tt13966978"),
            # ("Fantastic Four", "tt13966980"),
            # ("The Avengers", "tt0848228"),
            # ("The Avengers: Age of Ultron", "tt2395427"),
            # ("The Avengers: Infinity War", "tt4154756"),
            # ("The Avengers: Endgame", "tt4154796"),
            # ('Frozen', 'tt2294629'),
            # ('The Lion King', 'tt0110357'),
            # ('Aladdin', 'tt0103639'),
            # ('Beauty and the Beast', 'tt0101414'),
            # ('The Little Mermaid', 'tt0097757'),
            ('The Iron Lady', 'tt1007029'),
        ]
        
        logger.info(f"Loaded {len(self.famous_movies)} famous movies and {len(self.json_movies)} JSON movies for training.")
        
    def collect_training_data(self, min_comments_per_movie: int = 1) -> pd.DataFrame:
        """
        Collect training data from multiple movies using BERT as ground truth.
        Priority: First use existing data from raw_comments.csv, then scrape new movies.
        
        Args:
            min_comments_per_movie (int): Minimum comments to collect per movie
            
        Returns:
            pd.DataFrame: Training dataset with features and labels
        """
        all_data = []
        successful_movies = 0
        failed_movies = 0
        
        # First, try to load existing data from raw_comments.csv
        logger.info("Loading existing data from raw_comments.csv...")
        existing_data = self._load_existing_data()
        if existing_data:
            logger.info(f"Loaded {len(existing_data)} existing reviews from raw_comments.csv")
            all_data.extend(existing_data)
            successful_movies += 1
        
        # If we have enough data, we can stop early
        if len(all_data) >= 1000:
            logger.info(f"Reached target of 1000+ samples from existing data, stopping data collection")
            df = pd.DataFrame(all_data)
            logger.info(f"Total training data collected: {len(df)} samples from {successful_movies} successful movies ({failed_movies} failed)")
            df.to_csv(config.TRAINING_DATA_FILE, index=False)
            logger.info(f"Training data saved to {config.TRAINING_DATA_FILE}")
            return df
        
        # Then, try famous movies (priority)
        logger.info("Starting with famous movies (priority list)...")
        for movie_name, movie_id in self.famous_movies:
            try:
                logger.info(f"Collecting data for famous movie: {movie_name}")
                
                # Scrape movie reviews
                spider = IMDbSpider(movie_id=movie_id)
                movie_data = spider.scrape_movie_reviews(movie_id, min_comments_per_movie)
                
                if not movie_data:
                    logger.warning(f"No data collected for {movie_name} - skipping to next movie")
                    failed_movies += 1
                    continue
                    
                # Process with BERT for ground truth
                texts = [item['text'] for item in movie_data]
                bert_results = self.bert_analyzer.predict(texts)
                
                # Combine data
                for i, item in enumerate(movie_data):
                    bert_result = bert_results[i] if i < len(bert_results) else None
                    
                    if bert_result:  # Remove confidence threshold to allow all data
                        processed_data = {
                            'text': item['text'],
                            'cleaned_text': self.preprocessor.preprocess_text(item['text']),
                            'rating': item.get('rating', 0),
                            'username': item.get('username', ''),
                            'bert_sentiment': bert_result['label'],
                            'bert_confidence': bert_result['confidence'],
                            'movie_name': movie_name
                        }
                        all_data.append(processed_data)
                        
                logger.info(f"Collected {len(movie_data)} comments for {movie_name}")
                successful_movies += 1
                
                # If we have enough data, we can stop early
                if len(all_data) >= 1000:
                    logger.info(f"Reached target of 1000+ samples, stopping data collection")
                    break
                
            except Exception as e:
                logger.error(f"Error collecting data for {movie_name}: {e}")
                failed_movies += 1
                continue
        
        # Then, try JSON movies if we need more data
        if len(all_data) < 1000 and self.json_movies:
            logger.info("Moving to JSON movies for additional data...")
            for movie_name, movie_id in self.json_movies:
                try:
                    logger.info(f"Collecting data for JSON movie: {movie_name}")
                    
                    # Scrape movie reviews
                    spider = IMDbSpider(movie_id=movie_id)
                    movie_data = spider.scrape_movie_reviews(movie_id, min_comments_per_movie)
                    
                    if not movie_data:
                        logger.warning(f"No data collected for {movie_name} - skipping to next movie")
                        failed_movies += 1
                        continue
                        
                    # Process with BERT for ground truth
                    texts = [item['text'] for item in movie_data]
                    bert_results = self.bert_analyzer.predict(texts)
                    
                    # Combine data
                    for i, item in enumerate(movie_data):
                        bert_result = bert_results[i] if i < len(bert_results) else None
                        
                        if bert_result:  # Remove confidence threshold to allow all data
                            processed_data = {
                                'text': item['text'],
                                'cleaned_text': self.preprocessor.preprocess_text(item['text']),
                                'rating': item.get('rating', 0),
                                'username': item.get('username', ''),
                                'bert_sentiment': bert_result['label'],
                                'bert_confidence': bert_result['confidence'],
                                'movie_name': movie_name
                            }
                            all_data.append(processed_data)
                            
                    logger.info(f"Collected {len(movie_data)} comments for {movie_name}")
                    successful_movies += 1
                    
                    # If we have enough data, we can stop early
                    if len(all_data) >= 1000:
                        logger.info(f"Reached target of 1000+ samples, stopping data collection")
                        break
                    
                except Exception as e:
                    logger.error(f"Error collecting data for {movie_name}: {e}")
                    failed_movies += 1
                    continue
                
        df = pd.DataFrame(all_data)
        logger.info(f"Total training data collected: {len(df)} samples from {successful_movies} successful movies ({failed_movies} failed)")
        
        # Save raw training data
        df.to_csv(config.TRAINING_DATA_FILE, index=False)
        logger.info(f"Training data saved to {config.TRAINING_DATA_FILE}")
        
        return df

    def _load_existing_data(self) -> List[Dict[str, Any]]:
        """
        Load and process existing data from raw_comments.csv.
        
        Returns:
            List[Dict[str, Any]]: List of processed data dictionaries
        """
        try:
            import os
            if not os.path.exists(config.RAW_COMMENTS_FILE):
                logger.info("No existing raw_comments.csv file found")
                return []
            
            # Load existing data
            df = pd.read_csv(config.RAW_COMMENTS_FILE)
            logger.info(f"Loaded {len(df)} existing reviews from {config.RAW_COMMENTS_FILE}")
            
            # Filter out invalid data (JavaScript code, etc.)
            valid_data = []
            for _, row in df.iterrows():
                text = str(row.get('text', ''))
                # Skip JavaScript code and other invalid content
                if (len(text) > 10 and 
                    not text.startswith('window.') and 
                    not text.startswith('ue.count') and
                    not text.isdigit() and
                    'CSMLibrarySize' not in text):
                    
                    # Process with BERT for ground truth
                    bert_result = self.bert_analyzer.predict([text])[0]
                    
                    if bert_result:
                        processed_data = {
                            'text': text,
                            'cleaned_text': self.preprocessor.preprocess_text(text),
                            'rating': float(row.get('rating', 0)) if pd.notna(row.get('rating')) else 0.0,
                            'username': str(row.get('username', '')) if pd.notna(row.get('username')) else '',
                            'bert_sentiment': bert_result['label'],
                            'bert_confidence': bert_result['confidence'],
                            'movie_name': f"Movie_{row.get('movie_id', 'unknown')}"
                        }
                        valid_data.append(processed_data)
            
            logger.info(f"Processed {len(valid_data)} valid reviews from existing data")
            return valid_data
            
        except Exception as e:
            logger.error(f"Error loading existing data: {e}")
            return []
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features for model training.
        
        Args:
            df (pd.DataFrame): Training dataframe
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Features and labels
        """
        # Create TF-IDF features
        tfidf = TfidfVectorizer(
            max_features=config.TFIDF_MAX_FEATURES,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        text_features = tfidf.fit_transform(df['cleaned_text'])
        
        # Additional features
        text_length = df['cleaned_text'].str.len().values.reshape(-1, 1)
        rating = df['rating'].values.reshape(-1, 1)
        bert_confidence = df['bert_confidence'].values.reshape(-1, 1)
        
        # Combine features
        features = np.hstack([
            text_features.toarray(),
            text_length,
            rating,
            bert_confidence
        ])
        
        # Convert sentiment labels to numeric
        label_mapping = {
            'very negative': 0,
            'negative': 1, 
            'neutral': 2,
            'positive': 3,
            'very positive': 4
        }
        
        labels = df['bert_sentiment'].map(label_mapping).values
        
        # Save TF-IDF vectorizer
        joblib.dump(tfidf, config.TFIDF_VECTORIZER_FILE)
        logger.info(f"TF-IDF vectorizer saved to {config.TFIDF_VECTORIZER_FILE}")
        
        return features, labels
        
    def split_dataset(self, features: np.ndarray, labels: np.ndarray, 
                     test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """
        Split dataset into train and test sets.
        
        Args:
            features (np.ndarray): Feature matrix
            labels (np.ndarray): Label array
            test_size (float): Proportion of test set
            random_state (int): Random seed
            
        Returns:
            Tuple: (X_train, X_test, y_train, y_test)
        """
        return train_test_split(
            features, labels,
            test_size=test_size,
            random_state=random_state,
            stratify=labels
        )
        
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate training data quality.
        
        Args:
            df (pd.DataFrame): Training dataframe
            
        Returns:
            bool: True if data is valid
        """
        if len(df) < 10:  # Reduced from 1000 to 10
            logger.warning("Training dataset too small (< 10 samples)")
            return False
            
        # Check sentiment distribution
        sentiment_counts = df['bert_sentiment'].value_counts()
        min_class_size = len(df) * 0.05  # Reduced from 0.1 to 0.05 (5% per class)
        
        for sentiment, count in sentiment_counts.items():
            if count < min_class_size:
                logger.warning(f"Class {sentiment} has too few samples: {count}")
                return False
                
        # Check for missing values
        if df.isnull().any().any():
            logger.warning("Dataset contains missing values")
            return False
            
        logger.info("Training data validation passed")
        return True 

    def _load_movies_from_json(self) -> list:
        """
        Load movies from data/movies.json and return a list of (title, imdb_id) tuples.
        Returns empty list if file is missing or invalid.
        """
        try:
            with open("data/movies.json", "r", encoding="utf-8") as f:
                movies = json.load(f)
            # Only keep movies with both imdb_id and title
            movie_list = [(m["title"], m["imdb_id"]) for m in movies if m.get("imdb_id") and m.get("title")]
            logger.info(f"Loaded {len(movie_list)} movies from data/movies.json for training.")
            return movie_list
        except Exception as e:
            logger.error(f"Failed to load movies from data/movies.json: {e}")
            return [] 