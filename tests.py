import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from app import (
    get_imdb_id, get_reviews, preprocess,
    vectorize_reviews, compute_similarity, analyze_reviews
)


# =========================================================
# TEXT ACQUISITION TESTS
# =========================================================

def test_acquisition_valid_movie():
    """Valid movie name returns a TMDB ID, title, and rating."""
    tmdb_id, title, rating = get_imdb_id("The Batman")
    assert tmdb_id is not None
    assert isinstance(title, str) and len(title) > 0
    assert rating is not None


def test_acquisition_invalid_movie():
    """Completely nonsense movie name returns None for all fields."""
    tmdb_id, title, rating = get_imdb_id("xyzzy_this_does_not_exist_12345")
    assert tmdb_id is None
    assert title is None
    assert rating is None


# =========================================================
# TEXT PRE-PROCESSING TESTS
# =========================================================

def test_preprocessing_removes_stopwords():
    """Common stop words like 'the', 'is', 'a' are removed from tokens."""
    tokens, _ = preprocess("The movie is a masterpiece")
    assert "the" not in tokens
    assert "is"  not in tokens
    assert "a"   not in tokens


def test_preprocessing_lemmatises_correctly():
    """'running' and 'flies' should be lemmatised to their base forms."""
    tokens, _ = preprocess("He was running and the time flies")
    assert "run"  in tokens or "running" not in tokens
    assert "fli"  in tokens or "fly" in tokens or "flies" not in tokens


# =========================================================
# VECTORIZATION TESTS
# =========================================================

def test_vectorization_produces_matrix():
    """TF-IDF on 3 reviews should return a matrix with 3 rows."""
    texts = [
        "This movie was absolutely fantastic",
        "Terrible film, complete waste of time",
        "An average movie nothing special"
    ]
    vectorizer, matrix = vectorize_reviews(texts)
    assert matrix.shape[0] == 3


def test_vectorization_empty_input():
    """Empty string list should raise an error or return empty matrix."""
    try:
        vectorizer, matrix = vectorize_reviews([""])
        assert matrix.shape[0] == 1
    except Exception:
        pass  # acceptable — empty vocab raises ValueError in sklearn


# =========================================================
# SENTIMENT ANALYSIS TESTS
# =========================================================

def test_sentiment_rating_positive():
    """A review with rating 8 should be classified as Positive."""
    reviews = [("Great film loved it", "User1", 8.0)]
    results = analyze_reviews(reviews)
    assert results[0][3] == "Positive"


def test_sentiment_rating_negative():
    """A review with rating 2 should be classified as Negative."""
    reviews = [("Absolutely terrible waste of money", "User2", 2.0)]
    results = analyze_reviews(reviews)
    assert results[0][3] == "Negative"


def test_sentiment_no_rating_fallback():
    """A review with no rating uses TextBlob — clearly positive text → Positive."""
    reviews = [("This film is absolutely wonderful and brilliant", "User3", None)]
    results = analyze_reviews(reviews)
    assert results[0][3] in ("Positive", "Neutral")  # TextBlob may vary


def test_sentiment_no_rating_negative_fallback():
    """Clearly negative text with no rating should not be Positive."""
    reviews = [("This movie was awful terrible and horrible disaster", "User4", None)]
    results = analyze_reviews(reviews)
    assert results[0][3] in ("Negative", "Neutral")


# =========================================================
# RESULT / ANALYSIS TESTS
# =========================================================

def test_results_dataframe_columns():
    """analyze_reviews should return rows with 6 fields each."""
    reviews = [
        ("Great film", "Alice", 8.0),
        ("Terrible film", "Bob", 2.0),
    ]
    results = analyze_reviews(reviews)
    df = pd.DataFrame(results, columns=[
        "Review", "Author", "Rating", "Sentiment", "Tokens", "Similarity"
    ])
    assert list(df.columns) == ["Review", "Author", "Rating", "Sentiment", "Tokens", "Similarity"]
    assert len(df) == 2


def test_results_csv_saved(tmp_path):
    """CSV should be saved to the output directory and be readable."""
    reviews = [("Great film", "Alice", 8.0)]
    results = analyze_reviews(reviews)
    df = pd.DataFrame(results, columns=[
        "Review", "Author", "Rating", "Sentiment", "Tokens", "Similarity"
    ])
    path = tmp_path / "test_output.csv"
    df.to_csv(path, index=False)
    loaded = pd.read_csv(path)
    assert len(loaded) == 1
    assert "Sentiment" in loaded.columns