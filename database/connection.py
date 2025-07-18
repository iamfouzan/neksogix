"""
Database connection management for sentiment analysis system.

This module handles PostgreSQL connection setup, session management,
and error handling for the database operations.
"""

from typing import Optional, Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError, OperationalError
import logging
import time
from contextlib import contextmanager

from config import config
from .models import Base

logger = logging.getLogger(__name__)


class DatabaseConnection:
    """Database connection manager for PostgreSQL."""
    
    def __init__(self):
        """Initialize database connection manager."""
        self.engine = None
        self.SessionLocal = None
        self._setup_connection()
    
    def _setup_connection(self) -> None:
        """Setup database connection with connection pooling."""
        try:
            db_config = config.get_database_config()
            
            # Create engine with connection pooling
            self.engine = create_engine(
                db_config['url'],
                poolclass=QueuePool,
                pool_size=db_config['pool_size'],
                max_overflow=db_config['max_overflow'],
                pool_pre_ping=True,  # Verify connections before use
                pool_recycle=3600,   # Recycle connections every hour
                echo=db_config['echo']
            )
            
            # Create session factory
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
            
            logger.info("Database connection setup completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup database connection: {e}")
            raise
    
    def create_tables(self) -> bool:
        """Create all database tables."""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")
            return False
    
    def drop_tables(self) -> bool:
        """Drop all database tables."""
        try:
            Base.metadata.drop_all(bind=self.engine)
            logger.info("Database tables dropped successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to drop database tables: {e}")
            return False
    
    def get_session(self) -> Session:
        """Get a new database session."""
        if not self.SessionLocal:
            raise RuntimeError("Database session factory not initialized")
        return self.SessionLocal()
    
    @contextmanager
    def get_session_context(self) -> Generator[Session, None, None]:
        """Context manager for database sessions with automatic cleanup."""
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def test_connection(self) -> bool:
        """Test database connection."""
        try:
            with self.get_session_context() as session:
                session.execute("SELECT 1")
                logger.info("Database connection test successful")
                return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    def get_connection_info(self) -> dict:
        """Get database connection information."""
        if not self.engine:
            return {"status": "Not connected"}
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute("SELECT version()")
                version = result.fetchone()[0]
                
                return {
                    "status": "Connected",
                    "version": version,
                    "pool_size": self.engine.pool.size(),
                    "checked_in": self.engine.pool.checkedin(),
                    "checked_out": self.engine.pool.checkedout()
                }
        except Exception as e:
            return {"status": f"Error: {e}"}
    
    def close(self) -> None:
        """Close database connection."""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connection closed")


# Global database connection instance
db_connection = DatabaseConnection()


def get_db_session() -> Session:
    """Get a database session."""
    return db_connection.get_session()


@contextmanager
def get_db_session_context() -> Generator[Session, None, None]:
    """Context manager for database sessions."""
    with db_connection.get_session_context() as session:
        yield session


def init_database() -> bool:
    """Initialize database with tables."""
    try:
        # Test connection first
        if not db_connection.test_connection():
            logger.error("Database connection test failed")
            return False
        
        # Create tables
        if not db_connection.create_tables():
            logger.error("Failed to create database tables")
            return False
        
        logger.info("Database initialization completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        return False


def cleanup_database() -> None:
    """Cleanup database connections."""
    db_connection.close() 