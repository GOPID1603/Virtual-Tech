import streamlit as st
from textblob import TextBlob
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time

def main():
    st.set_page_config(page_title="Advanced AI Dashboard", page_icon="🚀", layout="wide")
    
    st.title("🚀 Advanced AI Dashboard")
    st.markdown("Welcome to the **Virtual Technologies AI Dashboard**. Select a module below to explore the AI models.")
    
    # Custom CSS for modern styling
    st.markdown("""
        <style>
        .stButton>button {
            background-color: #ff4b4b;
            color: white;
            border-radius: 10px;
            transition: all 0.3s;
        }
        .stButton>button:hover {
            transform: scale(1.05);
            background-color: #ff2b2b;
        }
        .metric-card {
            background: rgba(255,255,255,0.05);
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #ff4b4b;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        </style>
    """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🧠 NLP Sentiment Analysis", "🎬 Movie Recommendation Engine"])
    
    # ==========================================
    # TAB 1: SENTIMENT ANALYSIS
    # ==========================================
    with tab1:
        st.header("Sentiment Analysis")
        st.write("Analyze the sentiment of social media posts, product reviews, or any text.")
        
        user_input = st.text_area("Enter text to analyze:", height=150, placeholder="E.g., I absolutely love the stunning graphics in this game!")
        
        if st.button("Analyze Sentiment", key="sent_btn"):
            if user_input.strip():
                with st.spinner('Analyzing textual patterns...'):
                    time.sleep(0.5) # Simulate processing
                    blob = TextBlob(user_input)
                    polarity = blob.sentiment.polarity
                    subjectivity = blob.sentiment.subjectivity
                    
                    if polarity > 0.1:
                        sentiment, color, icon = "Positive", "#2ecc71", "😊"
                    elif polarity < -0.1:
                        sentiment, color, icon = "Negative", "#e74c3c", "😞"
                    else:
                        sentiment, color, icon = "Neutral", "#95a5a6", "😐"
                        
                st.subheader("Analysis Results")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"<div class='metric-card'><h3>Sentiment</h3><span style='color:{color}; font-size:28px;'>{sentiment} {icon}</span></div>", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"<div class='metric-card'><h3>Polarity</h3><span style='font-size:28px;'>{polarity:.2f}</span><br><small>-1.0 to 1.0</small></div>", unsafe_allow_html=True)
                with col3:
                    st.markdown(f"<div class='metric-card'><h3>Subjectivity</h3><span style='font-size:28px;'>{subjectivity:.2f}</span><br><small>0.0 to 1.0</small></div>", unsafe_allow_html=True)
            else:
                st.warning("Please enter some text to analyze.")

    # ==========================================
    # TAB 2: RECOMMENDATION ENGINE
    # ==========================================
    with tab2:
        st.header("Movie Recommendation Engine")
        st.write("Our Content-Based Filtering engine analyzes movie descriptions to find similar recommendations.")
        
        # Setup Data
        movies_data = {
            'MovieID': [1, 2, 3, 4, 5, 6, 7],
            'Title': ['Space Explorer', 'Deep Ocean Mystery', 'Mars Colonization', 'Romantic Sunset', 'Galactic Wars', 'Underwater Love', 'Alien Invasion'],
            'Description': [
                'A sci-fi adventure about astronauts exploring distant galaxies and encountering new life forms.',
                'A thrilling documentary exploring the unknown depths of the ocean and deep sea creatures.',
                'A sci-fi movie detailing the struggles and triumphs of the first human colony on Mars.',
                'A heartwarming romance about two lovers meeting at a beach during a beautiful sunset.',
                'An epic sci-fi battle between the galactic empire and the rebel alliance in space.',
                'A romantic drama about two marine biologists falling in love while studying ocean life.',
                'A sci-fi action movie where humanity fights against a terrifying alien invasion on Earth.'
            ]
        }
        df = pd.DataFrame(movies_data)
        
        # Calculate TF-IDF
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df['Description'])
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        # Select Movie
        selected_movie = st.selectbox("Select a movie you like:", df['Title'].tolist())
        
        if st.button("Get Recommendations", key="rec_btn"):
            with st.spinner('Calculating cosine similarity matrix...'):
                time.sleep(0.5)
                idx = df.index[df['Title'] == selected_movie].tolist()[0]
                sim_scores = list(enumerate(cosine_sim[idx]))
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:4] # Top 3
                movie_indices = [i[0] for i in sim_scores]
                scores = [i[1] for i in sim_scores]
                
            st.subheader(f"Because you liked '{selected_movie}'...")
            
            cols = st.columns(3)
            for i, col in enumerate(cols):
                if i < len(movie_indices):
                    movie_idx = movie_indices[i]
                    with col:
                        st.markdown(f"""
                        <div style='background:rgba(255,255,255,0.05); padding:20px; border-radius:10px; height:100%; border-top:4px solid #3498db;'>
                            <h3 style='color:#3498db;'>{df['Title'].iloc[movie_idx]}</h3>
                            <p style='font-size:14px; opacity:0.8;'>{df['Description'].iloc[movie_idx]}</p>
                            <b>Similarity Match: {(scores[i]*100):.1f}%</b>
                        </div>
                        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
