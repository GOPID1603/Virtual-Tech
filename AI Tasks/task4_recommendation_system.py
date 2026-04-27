import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def main():
    print("=== Virtual Technologies Internship: AI Task 4 - Content-Based Recommendation System ===")
    
    # 1. Create a synthetic dataset of movies and their descriptions
    print("Loading movie database...")
    movies_data = {
        'MovieID': [1, 2, 3, 4, 5, 6, 7],
        'Title': [
            'Space Explorer', 
            'Deep Ocean Mystery', 
            'Mars Colonization', 
            'Romantic Sunset', 
            'Galactic Wars', 
            'Underwater Love', 
            'Alien Invasion'
        ],
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
    
    # 2. Content-Based Filtering using TF-IDF
    print("Processing movie descriptions with TF-IDF Vectorization...")
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['Description'])
    
    # 3. Compute Cosine Similarity Matrix
    print("Computing Cosine Similarity between movies...")
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # 4. Recommendation Function
    def get_recommendations(title, cosine_sim_matrix, df, top_n=5):
        # Get the index of the movie that matches the title
        idx = df.index[df['Title'] == title].tolist()
        if not idx:
            return f"Movie '{title}' not found in database."
        idx = idx[0]
        
        # Get pairwise similarity scores for all movies with that movie
        sim_scores = list(enumerate(cosine_sim_matrix[idx]))
        
        # Sort movies based on similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get the scores of the top_n most similar movies (excluding itself)
        sim_scores = sim_scores[1:top_n+1]
        
        # Get movie indices
        movie_indices = [i[0] for i in sim_scores]
        
        # Return top N most similar movies
        return df[['Title', 'Description']].iloc[movie_indices]

    # 5. Generate Recommendations
    target_movie = 'Space Explorer'
    print(f"\n--- Generating Top Recommendations for '{target_movie}' ---")
    recommendations = get_recommendations(target_movie, cosine_sim, df, top_n=3)
    
    print("\nRecommendations based on content similarity:")
    for i, row in recommendations.iterrows():
        print(f"- {row['Title']}: {row['Description']}")

    target_movie_2 = 'Romantic Sunset'
    print(f"\n--- Generating Top Recommendations for '{target_movie_2}' ---")
    recommendations_2 = get_recommendations(target_movie_2, cosine_sim, df, top_n=2)
    
    print("\nRecommendations based on content similarity:")
    for i, row in recommendations_2.iterrows():
        print(f"- {row['Title']}: {row['Description']}")

    print("\n=== AI Task 4 Completed Successfully ===")

if __name__ == "__main__":
    main()
