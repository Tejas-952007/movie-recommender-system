# Movie Recommendation System

A content-based movie recommendation system using TF-IDF vectorization and cosine similarity.

## Features

- **Content-based filtering**: Recommends movies based on genres, keywords, cast, crew, and overview
- **TF-IDF vectorization**: Better similarity matching than raw word counts
- **Stemming**: Reduces vocabulary size and improves matching
- **Sparse matrix optimization**: Memory-efficient storage and computation
- **On-demand similarity**: Computes similarity only when needed (no N×N matrix)
- **Persistent model**: Save and load the trained model without recomputing

## Installation

```bash
pip install -r requirements.txt
```

## Dataset

The project uses the TMDB 5000 Movie Dataset:
- `datset/tmdb_5000_movies.csv`
- `datset/tmdb_5000_credits.csv`

## Usage

### Run the Notebook

Open `recommmendor.ipynb` in Jupyter Notebook or Google Colab and run all cells.

### Get Recommendations

```python
# After running the notebook
recommend('Avatar')
```

### Load Saved Model

```python
import pickle
from sklearn.metrics.pairwise import cosine_similarity

with open('movies.pkl', 'rb') as f:
    model = pickle.load(f)

new_df = model['df']
cv = model['vectorizer']
vectors = model['vectors']

# Use the recommend function
def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    sim = cosine_similarity(vectors[movie_index], vectors).flatten()
    top = sorted(enumerate(sim), key=lambda x: x[1], reverse=True)[1:6]
    for i, _ in top:
        print(new_df.iloc[i].title)

recommend('Avatar')
```

## Optimizations

- **Local paths**: Works without Google Drive/Colab
- **TfidfVectorizer**: Better than CountVectorizer for text similarity
- **Sparse matrices**: Reduces memory usage
- **Vectorized operations**: List comprehensions instead of loops
- **On-demand computation**: No full N×N similarity matrix
- **Complete model saving**: Saves vectorizer, vectors, and dataframe

## Requirements

- Python 3.7+
- numpy
- pandas
- scikit-learn
- nltk

## License

MIT
