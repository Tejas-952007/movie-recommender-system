"""
Movieflix Ultimate - Streamlit Web App
Auto-generates movies.pkl on first run (for deployment).
"""
import streamlit as st
import pickle
import pandas as pd
import requests
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MOVIES_PKL = os.path.join(SCRIPT_DIR, 'movies.pkl')

# Auto-generate movies.pkl if missing (for Streamlit Cloud / deployment)
if not os.path.exists(MOVIES_PKL):
    try:
        import build_model
        build_model.main()
    except Exception:
        pass

# --- 1. PAGE SETUP & CSS ---
st.set_page_config(page_title="Movieflix Ultimate", page_icon="🎬", layout="wide")

st.markdown("""
<style>
    /* 1. Main Dark Theme */
    .stApp {
        background-color: #141414;
        color: #ffffff;
    }

    /* 2. Header & Title Styling */
    h1 {
        color: #E50914;
        font-family: 'Bebas Neue', sans-serif;
        font-size: 3.5rem;
        text-align: center;
        text-shadow: 2px 2px 4px #000000;
        margin-bottom: 0;
    }

    /* 3. Animations */
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }

    .fade-in {
        animation: fadeIn 1.5s ease-out;
    }

    .mood-card {
        background-color: #1f1f1f;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        border: 2px solid transparent;
        transition: all 0.3s ease;
        cursor: pointer;
    }

    .mood-card:hover {
        border: 2px solid #E50914;
        transform: scale(1.05);
        box-shadow: 0 0 15px rgba(229, 9, 20, 0.6);
    }

    /* 4. Section Headers */
    h3 {
        color: #e5e5e5;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: bold;
        padding-left: 10px;
        border-left: 5px solid #E50914;
        margin-top: 30px;
    }

    /* 5. Movie Cards */
    div[data-testid="stImage"] {
        border-radius: 8px;
        transition: transform 0.3s ease;
    }
    div[data-testid="stImage"]:hover {
        transform: scale(1.08);
        cursor: pointer;
        z-index: 10;
        box-shadow: 0 10px 20px rgba(0,0,0,0.8);
    }

    /* 6. Custom Buttons */
    div.stButton > button {
        background-color: #E50914;
        color: white;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
        border-radius: 4px;
        transition: 0.3s;
        width: 100%;
    }
    div.stButton > button:hover {
        background-color: #f40612;
    }
</style>
""", unsafe_allow_html=True)


# --- 2. DATA LOADING ---
import traceback

# --- 2. DATA LOADING ---
@st.cache_resource
def load_data():
    """Load from movies.pkl (dict or DataFrame), or legacy movie_dict.pkl + similarity.pkl."""
    # Removed generic try-except to debug deployment issues
    movies_pkl = os.path.join(SCRIPT_DIR, 'movies.pkl')
    movie_dict_pkl = os.path.join(SCRIPT_DIR, 'movie_dict.pkl')
    similarity_pkl = os.path.join(SCRIPT_DIR, 'similarity.pkl')

    if os.path.exists(movies_pkl):
        with open(movies_pkl, 'rb') as f:
            data = pickle.load(f)

        if isinstance(data, dict) and 'df' in data and 'vectors' in data:
            movies = data['df'].copy()
            vectors = data['vectors']
            from sklearn.metrics.pairwise import cosine_similarity
            similarity = cosine_similarity(vectors)
            return movies, similarity
        elif isinstance(data, pd.DataFrame):
            movies = data.copy()
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            cv = TfidfVectorizer(max_features=5000, stop_words='english')
            vectors = cv.fit_transform(movies['tags'].fillna(''))
            similarity = cosine_similarity(vectors)
            return movies, similarity
    elif os.path.exists(movie_dict_pkl) and os.path.exists(similarity_pkl):
        movies_dict = pickle.load(open(movie_dict_pkl, 'rb'))
        movies = pd.DataFrame(movies_dict)
        similarity = pickle.load(open(similarity_pkl, 'rb'))
        return movies, similarity
    return None, None


try:
    movies, similarity = load_data()
except Exception:
    st.error(f"An error occurred while loading data:\n\n{traceback.format_exc()}")
    st.stop()

if movies is None:
    st.error("⚠️ movies.pkl not found.")
    st.info("Run: `python build_model.py`  (or run all cells in recommmendor.ipynb)")
    st.stop()


# --- 3. HELPER FUNCTIONS ---
def fetch_poster(movie_id):
    try:
        api_key = os.environ.get('TMDB_API_KEY', '8265bd1679663a7ea12ac168da84d2e8')
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US"
        data = requests.get(url, timeout=5.0).json()
        if data.get('poster_path'):
            return "https://image.tmdb.org/t/p/w500/" + data['poster_path']
    except Exception:
        pass
    return "https://via.placeholder.com/500x750?text=No+Image"


def get_genre_movies(genre, count=5):
    if 'tags' not in movies.columns:
        return [], []
    g_lower = genre.lower()
    mask = movies['tags'].apply(lambda x: g_lower in str(x).lower())
    if not mask.any():
        try:
            from nltk.stem.porter import PorterStemmer
            stemmed = ' '.join(PorterStemmer().stem(w) for w in g_lower.split())
            mask = movies['tags'].apply(lambda x: stemmed in str(x).lower())
        except ImportError:
            pass
    filtered = movies[mask]
    if filtered.empty:
        return [], []
    if len(filtered) > count:
        filtered = filtered.sample(n=count, random_state=42)
    names, posters = [], []
    for _, row in filtered.iterrows():
        names.append(row['title'])
        posters.append(fetch_poster(row['movie_id']))
    return names, posters


def recommend(movie_name):
    try:
        idx = movies[movies['title'] == movie_name].index[0]
        distances = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x: x[1])
        names, posters = [], []
        for i in distances[1:6]:
            m_id = movies.iloc[i[0]].movie_id
            names.append(movies.iloc[i[0]].title)
            posters.append(fetch_poster(m_id))
        return names, posters
    except Exception:
        return [], []


# --- 4. SESSION STATE INIT ---
if 'quiz_completed' not in st.session_state:
    st.session_state['quiz_completed'] = False
if 'user_mood' not in st.session_state:
    st.session_state['user_mood'] = 'Neutral'
if 'mood_genre' not in st.session_state:
    st.session_state['mood_genre'] = 'Action'
if 'page' not in st.session_state:
    st.session_state['page'] = 'home'
if 'search_results' not in st.session_state:
    st.session_state['search_results'] = ([], [], "")


# --- 5. THE MOOD QUIZ UI ---
def show_quiz():
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<h1 class='fade-in'>How are you feeling today?</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center; font-size:18px;' class='fade-in'>Let us pick the perfect movie for your mood.</p>",
        unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    moods = [
        {"emoji": "😂", "text": "Laugh", "genre": "Comedy", "color": "#FFD700"},
        {"emoji": "😤", "text": "Pumped", "genre": "Action", "color": "#FF4500"},
        {"emoji": "😨", "text": "Scared", "genre": "Horror", "color": "#800080"},
        {"emoji": "😭", "text": "Emotional", "genre": "Drama", "color": "#1E90FF"},
    ]

    cols = [c1, c2, c3, c4]

    for i, mood in enumerate(moods):
        with cols[i]:
            st.markdown(f"""
            <div class='mood-card fade-in' style='border-top: 5px solid {mood['color']}'>
                <div style='font-size: 50px;'>{mood['emoji']}</div>
                <h2 style='color: white; margin: 0;'>{mood['text']}</h2>
            </div>
            """, unsafe_allow_html=True)

            if st.button(f"Select {mood['text']}", key=f"btn_{i}"):
                st.session_state['user_mood'] = mood['text']
                st.session_state['mood_genre'] = mood['genre']
                st.session_state['quiz_completed'] = True
                st.rerun()

    st.markdown("<br><br>", unsafe_allow_html=True)
    col_skip = st.columns([1, 2, 1])
    with col_skip[1]:
        if st.button("Skip to Home 🏠"):
            st.session_state['quiz_completed'] = True
            st.rerun()


# --- 6. MAIN APP LOGIC ---

if not st.session_state['quiz_completed']:
    show_quiz()

else:
    col_logo, col_search, col_btn, col_home = st.columns([2, 4, 1, 1])
    with col_logo:
        st.markdown("<h1>MOVIEFLIX</h1>", unsafe_allow_html=True)
    with col_search:
        selected_movie = st.selectbox(
            "Search", movies['title'].values, label_visibility="collapsed",
            placeholder="Search movie...")
    with col_btn:
        if st.button("Search 🔍"):
            st.session_state['page'] = 'search'
            names, posters = recommend(selected_movie)
            st.session_state['search_results'] = (names, posters, f"More like '{selected_movie}'")
    with col_home:
        if st.button("Reset Quiz"):
            st.session_state['quiz_completed'] = False
            st.session_state['page'] = 'home'
            st.rerun()

    st.markdown("---")

    if st.session_state['page'] == 'home':

        if st.session_state['user_mood'] != 'Neutral':
            st.subheader(f"✨ Because you want to {st.session_state['user_mood']}...")
            names, posters = get_genre_movies(st.session_state['mood_genre'], count=5)

            c1, c2, c3, c4, c5 = st.columns(5)
            for i, col in enumerate([c1, c2, c3, c4, c5]):
                with col:
                    if i < len(posters):
                        st.image(posters[i], width='stretch')
                        st.caption(names[i])

        genres_to_show = ["Action", "Comedy", "Drama", "Romance"]

        for g in genres_to_show:
            st.subheader(f"{g} Hits")
            names, posters = get_genre_movies(g)
            cols = st.columns(5)
            for i, col in enumerate(cols):
                with col:
                    if i < len(posters):
                        st.image(posters[i], width='stretch')
                        st.caption(names[i])

    elif st.session_state['page'] == 'search':
        names, posters, title = st.session_state['search_results']
        st.subheader(title)
        if names:
            cols = st.columns(5)
            for i, col in enumerate(cols):
                with col:
                    if i < len(posters):
                        st.image(posters[i], width='stretch')
                        st.markdown(f"**{names[i]}**")

        if st.button("⬅️ Back"):
            st.session_state['page'] = 'home'
            st.rerun()

    st.markdown("<br><hr><center style='color:#555'>Movieflix Project © 2024</center>", unsafe_allow_html=True)
