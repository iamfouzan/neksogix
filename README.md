# 🎬 Movie Sentiment Analysis System

A comprehensive end-to-end sentiment analysis system for movie reviews using advanced AI models including BERT and custom SVM classifiers.

## 🌟 Features

- **Web Scraping**: Automated IMDb review collection using Scrapy
- **NLP Processing**: BERT-based sentiment analysis with Hugging Face Transformers
- **Custom ML Model**: SVM classifier trained on movie review data
- **Database Storage**: PostgreSQL integration with SQLAlchemy ORM
- **Web Interface**: Interactive Streamlit dashboard
- **Automation**: Discord webhook notifications
- **Visualization**: Comprehensive charts and analytics

## 🏗️ Architecture

```
sentiment_classifier/
├── scraper/          # IMDb web scraping
├── nlp/             # BERT sentiment analysis
├── ml/              # Custom SVM model
├── database/        # PostgreSQL models
├── automation/      # Discord webhooks
├── frontend/        # Streamlit interface
├── utils/           # Helper utilities
└── tests/           # Unit tests
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd sentiment_classifier

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Copy the environment template and configure your settings:

```bash
cp env.example .env
```

Edit `.env` with your configuration:

```env
# Database
DATABASE_URL=postgresql://user:password@localhost/sentiment_db

# Discord Webhook
DISCORD_WEBHOOK_URL=your_discord_webhook_url

# Model Settings
HUGGINGFACE_MODEL_NAME=nlptown/bert-base-multilingual-uncased-sentiment
BERT_BATCH_SIZE=16
BERT_MAX_LENGTH=512

# Scraping Settings
SCRAPING_DELAY=2
LOG_LEVEL=INFO
```

### 3. Database Setup

```bash
# Create PostgreSQL database
createdb sentiment_db

# Run migrations
psql -d sentiment_db -f database/migrations.sql
```

### 4. Model Training

```bash
# Train the custom SVM model
python main.py --mode train
```

This will:
- Collect training data from multiple movies
- Train the SVM model with hyperparameter tuning
- Generate evaluation reports and visualizations
- Send Discord notification upon completion

### 5. Run Predictions

```bash
# Analyze a single movie
python main.py --mode predict --movie "The Shawshank Redemption" --reviews 10
```

### 6. Web Interface

```bash
# Launch Streamlit dashboard
streamlit run frontend/streamlit_app.py
```

Visit `http://localhost:8501` to use the interactive interface.

## 📊 Model Performance

### BERT Model
- **Model**: `nlptown/bert-base-multilingual-uncased-sentiment`
- **Accuracy**: ~85-90% on movie reviews
- **Features**: 5-class sentiment classification (very negative to very positive)

### Custom SVM Model
- **Features**: TF-IDF + text length + rating + BERT confidence
- **Performance**: Comparable to BERT with faster inference
- **Ensemble**: Combines both models for optimal results

## 🔧 Usage Examples

### Command Line Interface

```bash
# Training mode
python main.py --mode train

# Prediction mode
python main.py --mode predict --movie "Inception" --reviews 50

# Analyze multiple movies
for movie in "The Godfather" "Pulp Fiction" "Fight Club"; do
    python main.py --mode predict --movie "$movie" --reviews 30
done
```

### Python API

```python
from ml.predictor import SentimentPredictor
from scraper.imdb_spider import ImdbSpider

# Initialize components
predictor = SentimentPredictor()
spider = ImdbSpider()

# Load models
predictor.load_models()

# Scrape and analyze
reviews = spider.scrape_movie_reviews("The Matrix", 50)
predictions = predictor.predict_sentiment([r['text'] for r in reviews])

# Get summary
summary = predictor.get_prediction_summary(predictions)
print(f"Positive: {summary['sentiment_percentages']['positive']:.1f}%")
```

### Database Queries

```python
from database.connection import get_db_session
from database.models import Movie, Comment, Prediction

with get_db_session() as session:
    # Get all analyzed movies
    movies = session.query(Movie).all()
    
    # Get predictions for a movie
    movie = session.query(Movie).filter(Movie.name == "The Shawshank Redemption").first()
    predictions = session.query(Prediction).join(Comment).filter(Comment.movie_id == movie.id).all()
```

## 📈 Data Flow

1. **Input**: Movie name from user
2. **Scraping**: IMDb review collection with pagination
3. **Preprocessing**: Text cleaning and normalization
4. **BERT Analysis**: Sentiment classification with confidence scores
5. **SVM Prediction**: Custom model inference
6. **Ensemble**: Combine predictions using confidence weighting
7. **Storage**: Save results to PostgreSQL
8. **Notification**: Send Discord webhook
9. **Visualization**: Generate charts and reports

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Test specific modules
pytest tests/test_scraper.py
pytest tests/test_nlp.py
pytest tests/test_ml.py
```

## 📁 Project Structure

```
sentiment_classifier/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── config.py                # Configuration management
├── main.py                  # Main application entry point
├── .env.example             # Environment variables template
├── database/                # Database models and connection
│   ├── models.py           # SQLAlchemy models
│   ├── connection.py       # Database connection
│   └── migrations.sql      # Database schema
├── scraper/                # Web scraping components
│   ├── imdb_spider.py     # IMDb spider
│   ├── settings.py        # Scrapy settings
│   └── scrapy.cfg         # Scrapy configuration
├── nlp/                    # Natural language processing
│   ├── preprocessor.py    # Text preprocessing
│   └── sentiment_analyzer.py # BERT sentiment analysis
├── ml/                     # Machine learning components
│   ├── data_preparation.py # Training data preparation
│   ├── trainer.py         # Model training
│   ├── predictor.py       # Prediction utilities
│   └── model_evaluation.py # Model evaluation
├── automation/             # Automation components
│   └── discord_webhook.py # Discord notifications
├── frontend/              # Web interface
│   └── streamlit_app.py  # Streamlit dashboard
├── utils/                 # Utility functions
│   └── helpers.py        # Common utilities
├── tests/                 # Unit tests
│   ├── test_scraper.py   # Scraper tests
│   ├── test_nlp.py       # NLP tests
│   └── test_ml.py        # ML tests
├── models/                # Trained model files
│   ├── sentiment_model.pkl
│   └── tfidf_vectorizer.pkl
├── data/                  # Data files
│   ├── raw_comments.csv
│   ├── training_data.csv
│   └── output.csv
└── logs/                  # Application logs
    └── app.log
```

## 🔍 Troubleshooting

### Common Issues

1. **Models not loading**
   ```bash
   # Ensure models are trained first
   python main.py --mode train
   ```

2. **Database connection errors**
   ```bash
   # Check PostgreSQL is running
   sudo systemctl status postgresql
   
   # Verify connection string in .env
   DATABASE_URL=postgresql://user:password@localhost/sentiment_db
   ```

3. **Scraping blocked by IMDb**
   ```bash
   # Increase delay in config.py
   SCRAPING_DELAY = 3
   
   # Use rotating user agents
   # Already implemented in scraper/settings.py
   ```

4. **Discord webhook not working**
   ```bash
   # Test webhook
   python -c "from automation.discord_webhook import DiscordWebhook; DiscordWebhook().test_webhook()"
   ```

### Performance Optimization

1. **GPU Acceleration**: Install CUDA for faster BERT inference
2. **Database Indexing**: Add indexes for frequently queried columns
3. **Caching**: Implement Redis for repeated movie searches
4. **Batch Processing**: Increase batch size for BERT (configurable)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Run tests: `pytest tests/`
5. Commit changes: `git commit -am 'Add feature'`
6. Push to branch: `git push origin feature-name`
7. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Hugging Face** for the BERT model
- **IMDb** for providing movie review data
- **Scrapy** for web scraping framework
- **Streamlit** for the web interface
- **Discord** for webhook integration

## 📞 Support

For questions or issues:
- Create an issue on GitHub
- Check the logs in `logs/app.log`
- Review the troubleshooting section above

---

**Made with ❤️ for sentiment analysis enthusiasts** 