"""
Generate movies.pkl from the TMDB dataset.
Run this script if movies.pkl doesn't exist (e.g. after fresh clone).
Usage: python build_model.py
"""
import os
import ast
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datset')


def convert(obj):
    """Extract names from JSON string."""
    try:
        return [i['name'] for i in ast.literal_eval(obj)]
    except (ValueError, TypeError):
        return []


def convert_cast(obj):
    """Extract top 3 cast names."""
    try:
        return [i['name'] for i in ast.literal_eval(obj)[:3]]
    except (ValueError, TypeError):
        return []


def convert_crew(obj):
    """Extract Director name."""
    try:
        crew = ast.literal_eval(obj)
        return [i['name'] for i in crew if i['job'] == 'Director'][:1]
    except (ValueError, TypeError):
        return []


def main():
    movies_path = os.path.join(DATA_DIR, 'tmdb_5000_movies.csv')
    credits_path = os.path.join(DATA_DIR, 'tmdb_5000_credits.csv')

    if not os.path.exists(movies_path) or not os.path.exists(credits_path):
        print("Error: CSV files not found in datset/")
        print("Download TMDB 5000 dataset from Kaggle: tmdb_5000_movies.csv, tmdb_5000_credits.csv")
        return 1

    print("Loading CSVs...")
    movies = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)

    credits = credits[['movie_id', 'cast', 'crew']].rename(columns={'movie_id': 'id'})
    movies = movies.merge(credits, on='id')

    movies = movies[['id', 'title', 'overview', 'cast', 'crew', 'genres', 'keywords']]
    movies = movies.rename(columns={'id': 'movie_id'})
    movies = movies.dropna(subset=['overview'])

    print("Parsing genres, keywords, cast, crew...")
    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast'] = movies['cast'].apply(convert_cast)
    movies['crew'] = movies['crew'].apply(convert_crew)

    movies['overview'] = movies['overview'].apply(lambda x: x.split() if isinstance(x, str) else [])
    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
    movies['tags'] = movies['tags'].apply(lambda x: ' '.join(str(i) for i in x) if isinstance(x, list) else '')
    movies['tags'] = movies['tags'].apply(lambda x: x.lower())
    movies['tags'] = movies['tags'].str.replace(r'\s+', ' ', regex=True)

    new_df = movies[['movie_id', 'title', 'tags']].copy()

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'movies.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(new_df, f)

    print(f"Saved movies.pkl ({len(new_df)} movies) at {output_path}")
    print("You can now run: streamlit run app.py")
    return 0


if __name__ == '__main__':
    exit(main())
