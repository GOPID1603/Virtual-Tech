import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def main():
    print("=== CODTECH Internship: Task 4 - Recommendation System ===")
    
    # 1. Create a synthetic User-Item Rating Matrix (Collaborative Filtering)
    # Rows are users, Columns are movies, Values are ratings (0 means unrated)
    print("Creating synthetic user-item rating matrix...")
    ratings_dict = {
        'Action Movie 1': [5, 4, 0, 0, 1],
        'Action Movie 2': [4, 5, 0, 1, 0],
        'Romance Movie 1': [0, 0, 5, 4, 2],
        'Romance Movie 2': [1, 0, 4, 5, 0],
        'Comedy Movie 1': [2, 1, 3, 3, 5]
    }
    users = ['User A', 'User B', 'User C', 'User D', 'User E']
    
    df = pd.DataFrame(ratings_dict, index=users)
    print("\nInitial Rating Matrix:")
    print(df)
    
    # 2. Compute Cosine Similarity between Users
    print("\nComputing User Similarity Matrix (Cosine Similarity)...")
    user_similarity = cosine_similarity(df)
    user_similarity_df = pd.DataFrame(user_similarity, index=users, columns=users)
    print(user_similarity_df.round(2))
    
    # 3. Recommendation Function
    def recommend_movies(user, df_ratings, df_similarity, num_recommendations=2):
        print(f"\n--- Generating Recommendations for {user} ---")
        # Get similar users
        similar_users = df_similarity[user].sort_values(ascending=False).drop(user)
        top_similar_user = similar_users.index[0]
        similarity_score = similar_users.iloc[0]
        
        print(f"Most similar user is {top_similar_user} (Similarity: {similarity_score:.2f})")
        
        # Get movies top_similar_user rated highly (e.g., >= 4) that the target user hasn't rated
        target_user_ratings = df_ratings.loc[user]
        similar_user_ratings = df_ratings.loc[top_similar_user]
        
        unrated_movies = target_user_ratings[target_user_ratings == 0].index
        recommendations = similar_user_ratings[unrated_movies]
        
        # Sort and filter
        recommendations = recommendations[recommendations >= 4].sort_values(ascending=False)
        
        if len(recommendations) == 0:
            print("No new movies to recommend based on the most similar user.")
        else:
            print("Recommended Movies:")
            for movie, rating in recommendations.head(num_recommendations).items():
                print(f"- {movie} (Expected high rating based on {top_similar_user})")
                
    # 4. Generate some recommendations
    # User A likes Action, User C likes Romance. Let's see what happens.
    recommend_movies('User A', df, user_similarity_df)
    recommend_movies('User C', df, user_similarity_df)
    
    print("\n=== Task 4 Completed Successfully ===")

if __name__ == "__main__":
    main()
