import requests
import os
import re
from textblob import TextBlob
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("punkt",        quiet=True)
nltk.download("punkt_tab",    quiet=True)
nltk.download("stopwords",    quiet=True)
nltk.download("wordnet",      quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)

API_KEY = "ef9c7d784836fe362f49a2c768ca82d8"


# -----------------------------
# TMDB: SEARCH MOVIE BY NAME
# -----------------------------
def get_imdb_id(movie_name):
    url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={movie_name}"
    response = requests.get(url)
    data = response.json()
    results = data.get("results", [])

    if not results:
        return None, None, None

    movie = results[0]
    tmdb_id = movie["id"]

    return tmdb_id, movie["title"], movie.get("vote_average")


# -----------------------------
# GET REVIEWS FROM TMDB API (up to 50)
# -----------------------------
def get_reviews(tmdb_id):
    reviews = []
    page = 1

    while len(reviews) < 50 and page <= 5:
        url = f"https://api.themoviedb.org/3/movie/{tmdb_id}/reviews?api_key={API_KEY}&page={page}"
        response = requests.get(url)
        data = response.json()
        results = data.get("results", [])

        if not results:
            break

        for r in results:
            content = r.get("content", "").strip()
            author  = r.get("author", "Unknown")

            rating = None
            author_details = r.get("author_details", {})
            if author_details.get("rating") is not None:
                rating = author_details["rating"]

            if content:
                reviews.append((content, author, rating))

        page += 1

    return reviews[:50]


# -----------------------------
# TEXT PRE-PROCESSING
# -----------------------------
lemmatizer = WordNetLemmatizer()
stop_words  = set(stopwords.words("english"))

def preprocess(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text.lower())
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    pos_tags = pos_tag(tokens)

    def get_wordnet_pos(tag):
        if tag.startswith("V"): return "v"
        if tag.startswith("J"): return "a"
        if tag.startswith("R"): return "r"
        return "n"

    lemmatized = [lemmatizer.lemmatize(word, get_wordnet_pos(tag))
                  for word, tag in pos_tags]

    return lemmatized, pos_tags


# -----------------------------
# VECTORIZATION — TF-IDF
# -----------------------------
def vectorize_reviews(review_texts):
    vectorizer = TfidfVectorizer(max_features=500)
    tfidf_matrix = vectorizer.fit_transform(review_texts)
    return vectorizer, tfidf_matrix


# -----------------------------
# SIMILARITY MEASURE
# -----------------------------
def compute_similarity(tfidf_matrix):
    sim_matrix = cosine_similarity(tfidf_matrix)
    n = sim_matrix.shape[0]
    avg_similarities = []
    for i in range(n):
        others = [sim_matrix[i][j] for j in range(n) if j != i]
        avg_similarities.append(round(sum(others) / len(others), 4) if others else 0.0)
    return avg_similarities


# -----------------------------
# SENTIMENT ANALYSIS
# -----------------------------
def analyze_reviews(reviews):
    results      = []
    review_texts = [content for content, _, _ in reviews]

    _, tfidf_matrix = vectorize_reviews(review_texts)
    similarities    = compute_similarity(tfidf_matrix)

    for i, (content, author, rating) in enumerate(reviews):
        tokens, pos_tags = preprocess(content)

        if rating is not None:
            if rating <= 3:
                sentiment = "Negative"
            elif rating <= 6:
                sentiment = "Neutral"
            else:
                sentiment = "Positive"
        else:
            polarity = TextBlob(content).sentiment.polarity
            if polarity > 0.05:
                sentiment = "Positive"
            elif polarity < -0.05:
                sentiment = "Negative"
            else:
                sentiment = "Neutral"

        rating_str  = f"{rating}/10" if rating is not None else "N/A"
        token_count = len(tokens)
        similarity  = similarities[i]

        results.append([content, author, rating_str, sentiment,
                        token_count, similarity])

    return results


# -----------------------------
# SAVE TEXT SUMMARY → reviews folder
# -----------------------------
def save_text_summary(df, title, tmdb_rating, output_dir="reviews"):
    os.makedirs(output_dir, exist_ok=True)
    safe_title = "".join(c for c in title if c.isalnum() or c in (" ", "-", "_")).strip()

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"{safe_title}_{timestamp}_summary.txt")

    counts = df["Sentiment"].value_counts()

    with open(path, "w", encoding="utf-8") as f:
        f.write("Movie Mood Analyzer — Results Summary\n")
        f.write(f"{'=' * 45}\n")
        f.write(f"Movie        : {title}\n")
        f.write(f"TMDB Rating  : {tmdb_rating}\n")
        f.write(f"Reviews      : {len(df)}\n")
        f.write(f"Generated    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("Sentiment Breakdown\n")
        f.write(f"{'-' * 25}\n")
        for label, count in counts.items():
            pct = round(count / len(df) * 100, 1)
            f.write(f"  {label:<10}: {count} reviews ({pct}%)\n")

        f.write(f"\nAvg Cosine Similarity : {df['Similarity'].mean():.4f}\n")
        f.write(f"Avg Token Count       : {df['Tokens'].mean():.1f}\n\n")

        f.write("Individual Reviews\n")
        f.write(f"{'-' * 45}\n")
        for _, row in df.iterrows():
            f.write(f"Author    : {row['Author']}\n")
            f.write(f"Rating    : {row['Rating']}\n")
            f.write(f"Sentiment : {row['Sentiment']}\n")
            f.write(f"Similarity: {row['Similarity']}\n")
            f.write(f"Review    : {row['Review'][:200]}\n")
            f.write(f"{'-' * 45}\n")

    return path


# -----------------------------
# BUILD EMBEDDED PIE CHART
# -----------------------------
def build_chart(df, movie_title=""):
    counts = df["Sentiment"].value_counts()

    colors = {
        "Positive": "#4CAF50",
        "Negative": "#f44336",
        "Neutral":  "#2196F3"
    }
    chart_colors = [colors.get(label, "#999") for label in counts.index]

    fig, ax = plt.subplots(figsize=(3.6, 3.6), facecolor="#1e1e1e")
    ax.set_facecolor("#1e1e1e")
    ax.pie(counts, labels=counts.index, autopct="%1.1f%%",
           colors=chart_colors, startangle=140,
           textprops={"color": "white", "fontsize": 10})
    ax.set_title(
        movie_title if movie_title else "Sentiment",
        color="white", fontsize=11, pad=10
    )
    fig.tight_layout()

    return fig