"""
Database models for sentiment analysis system.

This module defines SQLAlchemy models for movies, comments, and predictions
with proper relationships, validation methods, and CRUD operations.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from sqlalchemy import Column, Integer, String, Text, Float, DateTime, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy import create_engine, func
import logging

logger = logging.getLogger(__name__)

Base = declarative_base()


class Movie(Base):
    """Movie model for storing movie information."""
    
    __tablename__ = 'movies'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False, index=True)
    imdb_id = Column(String(20), nullable=False, unique=True, index=True)
    processed_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    comments = relationship("Comment", back_populates="movie", cascade="all, delete-orphan")
    
    def __str__(self) -> str:
        """String representation of Movie."""
        return f"Movie(id={self.id}, name='{self.name}', imdb_id='{self.imdb_id}')"
    
    def __repr__(self) -> str:
        """Detailed string representation of Movie."""
        return f"<Movie(id={self.id}, name='{self.name}', imdb_id='{self.imdb_id}', processed_at='{self.processed_at}')>"
    
    @classmethod
    def create(cls, session, name: str, imdb_id: str) -> 'Movie':
        """Create a new movie record."""
        try:
            movie = cls(name=name, imdb_id=imdb_id)
            session.add(movie)
            session.commit()
            logger.info(f"Created movie: {movie}")
            return movie
        except Exception as e:
            session.rollback()
            logger.error(f"Error creating movie: {e}")
            raise
    
    @classmethod
    def get_by_imdb_id(cls, session, imdb_id: str) -> Optional['Movie']:
        """Get movie by IMDb ID."""
        try:
            return session.query(cls).filter(cls.imdb_id == imdb_id).first()
        except Exception as e:
            logger.error(f"Error getting movie by IMDb ID {imdb_id}: {e}")
            return None
    
    @classmethod
    def get_by_name(cls, session, name: str) -> Optional['Movie']:
        """Get movie by name."""
        try:
            return session.query(cls).filter(cls.name.ilike(f"%{name}%")).first()
        except Exception as e:
            logger.error(f"Error getting movie by name {name}: {e}")
            return None
    
    @classmethod
    def get_all(cls, session) -> List['Movie']:
        """Get all movies."""
        try:
            return session.query(cls).all()
        except Exception as e:
            logger.error(f"Error getting all movies: {e}")
            return []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert movie to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'imdb_id': self.imdb_id,
            'processed_at': self.processed_at.isoformat() if self.processed_at else None
        }


class Comment(Base):
    """Comment model for storing movie review comments."""
    
    __tablename__ = 'comments'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    movie_id = Column(Integer, ForeignKey('movies.id'), nullable=False, index=True)
    text = Column(Text, nullable=False)
    cleaned_text = Column(Text, nullable=True)
    username = Column(String(100), nullable=True)
    rating = Column(Float, nullable=True)
    date = Column(DateTime, nullable=True)
    
    # Relationships
    movie = relationship("Movie", back_populates="comments")
    predictions = relationship("Prediction", back_populates="comment", cascade="all, delete-orphan")
    
    def __str__(self) -> str:
        """String representation of Comment."""
        return f"Comment(id={self.id}, movie_id={self.movie_id}, username='{self.username}')"
    
    def __repr__(self) -> str:
        """Detailed string representation of Comment."""
        return f"<Comment(id={self.id}, movie_id={self.movie_id}, username='{self.username}', rating={self.rating})>"
    
    @classmethod
    def create(cls, session, movie_id: int, text: str, username: str = None, 
               rating: float = None, date: datetime = None, cleaned_text: str = None) -> 'Comment':
        """Create a new comment record."""
        try:
            comment = cls(
                movie_id=movie_id,
                text=text,
                cleaned_text=cleaned_text,
                username=username,
                rating=rating,
                date=date
            )
            session.add(comment)
            session.commit()
            logger.info(f"Created comment: {comment}")
            return comment
        except Exception as e:
            session.rollback()
            logger.error(f"Error creating comment: {e}")
            raise
    
    @classmethod
    def get_by_movie_id(cls, session, movie_id: int) -> List['Comment']:
        """Get all comments for a movie."""
        try:
            return session.query(cls).filter(cls.movie_id == movie_id).all()
        except Exception as e:
            logger.error(f"Error getting comments for movie {movie_id}: {e}")
            return []
    
    @classmethod
    def get_by_id(cls, session, comment_id: int) -> Optional['Comment']:
        """Get comment by ID."""
        try:
            return session.query(cls).filter(cls.id == comment_id).first()
        except Exception as e:
            logger.error(f"Error getting comment {comment_id}: {e}")
            return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert comment to dictionary."""
        return {
            'id': self.id,
            'movie_id': self.movie_id,
            'text': self.text,
            'cleaned_text': self.cleaned_text,
            'username': self.username,
            'rating': self.rating,
            'date': self.date.isoformat() if self.date else None
        }


class Prediction(Base):
    """Prediction model for storing sentiment analysis results."""
    
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    comment_id = Column(Integer, ForeignKey('comments.id'), nullable=False, index=True)
    bert_sentiment = Column(String(20), nullable=True)
    bert_confidence = Column(Float, nullable=True)
    svm_sentiment = Column(String(20), nullable=True)
    svm_confidence = Column(Float, nullable=True)
    final_sentiment = Column(String(20), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    comment = relationship("Comment", back_populates="predictions")
    
    def __str__(self) -> str:
        """String representation of Prediction."""
        return f"Prediction(id={self.id}, comment_id={self.comment_id}, final_sentiment='{self.final_sentiment}')"
    
    def __repr__(self) -> str:
        """Detailed string representation of Prediction."""
        return f"<Prediction(id={self.id}, comment_id={self.comment_id}, final_sentiment='{self.final_sentiment}', bert_conf={self.bert_confidence}, svm_conf={self.svm_confidence})>"
    
    @classmethod
    def create(cls, session, comment_id: int, final_sentiment: str, 
               bert_sentiment: str = None, bert_confidence: float = None,
               svm_sentiment: str = None, svm_confidence: float = None) -> 'Prediction':
        """Create a new prediction record."""
        try:
            prediction = cls(
                comment_id=comment_id,
                bert_sentiment=bert_sentiment,
                bert_confidence=bert_confidence,
                svm_sentiment=svm_sentiment,
                svm_confidence=svm_confidence,
                final_sentiment=final_sentiment
            )
            session.add(prediction)
            session.commit()
            logger.info(f"Created prediction: {prediction}")
            return prediction
        except Exception as e:
            session.rollback()
            logger.error(f"Error creating prediction: {e}")
            raise
    
    @classmethod
    def get_by_comment_id(cls, session, comment_id: int) -> Optional['Prediction']:
        """Get prediction by comment ID."""
        try:
            return session.query(cls).filter(cls.comment_id == comment_id).first()
        except Exception as e:
            logger.error(f"Error getting prediction for comment {comment_id}: {e}")
            return None
    
    @classmethod
    def get_by_movie_id(cls, session, movie_id: int) -> List['Prediction']:
        """Get all predictions for a movie."""
        try:
            return session.query(cls).join(Comment).filter(Comment.movie_id == movie_id).all()
        except Exception as e:
            logger.error(f"Error getting predictions for movie {movie_id}: {e}")
            return []
    
    @classmethod
    def get_sentiment_stats(cls, session, movie_id: int) -> Dict[str, Any]:
        """Get sentiment statistics for a movie."""
        try:
            stats = session.query(
                cls.final_sentiment,
                func.count(cls.id).label('count')
            ).join(Comment).filter(Comment.movie_id == movie_id).group_by(cls.final_sentiment).all()
            
            return {sentiment: count for sentiment, count in stats}
        except Exception as e:
            logger.error(f"Error getting sentiment stats for movie {movie_id}: {e}")
            return {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert prediction to dictionary."""
        return {
            'id': self.id,
            'comment_id': self.comment_id,
            'bert_sentiment': self.bert_sentiment,
            'bert_confidence': self.bert_confidence,
            'svm_sentiment': self.svm_sentiment,
            'svm_confidence': self.svm_confidence,
            'final_sentiment': self.final_sentiment,
            'created_at': self.created_at.isoformat() if self.created_at else None
        } 