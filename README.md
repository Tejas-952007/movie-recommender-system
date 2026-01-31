# 🎬 Movie Recommendation System

A content-based movie recommendation engine that suggests similar movies using TF-IDF vectorization and cosine similarity. Built with Python, scikit-learn, and the TMDB 5000 dataset.

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [How It Works](#-how-it-works)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Example Output](#-example-output)
- [Dataset](#-dataset)
- [Optimizations](#-optimizations)
- [Technologies Used](#-technologies-used)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

This project implements a **content-based filtering** approach to recommend movies. Unlike collaborative filtering (which relies on user ratings), this system analyzes movie metadata—genres, keywords, cast, crew, and plot overview—to find semantically similar films.

**Perfect for:** Learning ML pipelines, portfolio projects, or building a foundation for a movie discovery app.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Content-Based Filtering** | Recommends based on movie attributes (genres, cast, crew, keywords, overview) |
| **TF-IDF Vectorization** | Captures term importance for better similarity matching |
| **Porter Stemming** | Reduces word variations (loving → lov) for improved matches |
| **Sparse Matrix** | Memory-efficient storage for large vocabularies |
| **On-Demand Similarity** | Computes only when needed—no full N×N matrix |
| **Model Persistence** | Save & load trained model for instant recommendations |
| **Local & Colab Ready** | Works with local datasets or Google Colab |

---

## 🔬 How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│  1. DATA MERGE                                                   │
│     tmdb_5000_movies.csv + tmdb_5000_credits.csv                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  2. FEATURE EXTRACTION                                           │
│     Parse JSON → Extract: genres, keywords, top 3 cast, director │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  3. TAG CREATION                                                 │
│     Combine: overview + genres + keywords + cast + crew          │
│     → Preprocess: lowercase, remove spaces, stemming             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  4. VECTORIZATION                                                │
│     TfidfVectorizer (max 5000 features, English stop words)      │
│     → Sparse matrix representation                               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  5. RECOMMENDATION                                               │
│     Input movie → Cosine similarity → Top 5 similar movies       │
└─────────────────────────────────────────────────────────────────┘
```

**Algorithm:** Cosine similarity measures the angle between two vectors. Movies with similar tag profiles (genres, cast, themes) have smaller angles and higher similarity scores.

---

## 📁 Project Structure

```
movie_recommendation_system/
├── datset/
│   ├── tmdb_5000_movies.csv      # Movie metadata
│   └── tmdb_5000_credits.csv     # Cast & crew data
├── recommmendor.ipynb            # Main notebook (EDA + training + recommend)
├── movies.pkl                    # Saved model (generated after first run)
├── requirements.txt              # Python dependencies
└── README.md
```

---

## 🛠 Installation

### Prerequisites

- Python 3.7 or higher
- pip

### Steps

1. **Clone the repository**

   ```bash
   git clone https://github.com/Tejas-952007/movie-recommender-system.git
   cd movie-recommender-system
   ```

2. **Create a virtual environment** (recommended)

   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   # or: venv\Scripts\activate   # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data** (required for stemming)

   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Open the notebook
jupyter notebook recommmendor.ipynb
# or use Google Colab and upload the notebook
```

Run all cells in the notebook. The first run will process the dataset and save the model to `movies.pkl`.

---

## 📖 Usage

### Option 1: Use the Notebook

Open `recommmendor.ipynb` and run all cells. Then call:

```python
recommend('Avatar')
recommend('The Dark Knight')
recommend('Inception')
```

### Option 2: Load Saved Model (No Re-training)

```python
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Load model
with open('movies.pkl', 'rb') as f:
    model = pickle.load(f)

df = model['df']
vectorizer = model['vectorizer']
vectors = model['vectors']

# Recommend function
def recommend(movie):
    idx = df[df['title'] == movie].index[0]
    sim = cosine_similarity(vectors[idx], vectors).flatten()
    top = sorted(enumerate(sim), key=lambda x: x[1], reverse=True)[1:6]
    for i, _ in top:
        print(df.iloc[i]['title'])

recommend('Avatar')
```

---

## 📊 Example Output

```
Input: recommend('Avatar')

Output:
Aliens
Falcon Rising
Battle: Los Angeles
Aliens vs Predator: Requiem
Apollo 18
```

---

## 📦 Dataset

| File | Description |
|------|-------------|
| `tmdb_5000_movies.csv` | ~5000 movies with title, overview, genres, keywords, etc. |
| `tmdb_5000_credits.csv` | Cast and crew for each movie |

**Source:** [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) (Kaggle)

---

## ⚡ Optimizations

| Optimization | Benefit |
|--------------|---------|
| **TfidfVectorizer** | Better similarity than raw CountVectorizer |
| **Sparse matrices** | ~10x less memory than dense arrays |
| **On-demand cosine** | No N×N matrix; O(n) per query instead of O(n²) |
| **List comprehensions** | Faster parsing than explicit loops |
| **Stemming before fit** | Smaller vocabulary, better generalization |
| **Full model save** | Instant load—no recomputation |

---

## 🛠 Technologies Used

- **Python 3**
- **pandas** – Data manipulation
- **numpy** – Numerical operations
- **scikit-learn** – TfidfVectorizer, cosine_similarity
- **nltk** – Porter Stemmer for text normalization

---

## 🤝 Contributing

Contributions are welcome! Feel free to:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

**Author:** [Tejas-952007](https://github.com/Tejas-952007)
