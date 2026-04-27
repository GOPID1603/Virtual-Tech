import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def main():
    print("=== CODTECH Internship: Task 2 - Sentiment Analysis with NLP ===")
    
    # 1. Create a synthetic dataset of customer reviews
    print("Generating customer reviews dataset...")
    reviews = [
        "I absolutely loved this product! It works perfectly.",
        "Terrible experience. The quality is so cheap and it broke in a day.",
        "Good value for the price, I'm satisfied.",
        "Worst customer service ever, I will never buy from them again.",
        "Amazing, exceeded my expectations!",
        "Not great. It doesn't look like the picture at all.",
        "Highly recommended! Fast shipping and great quality.",
        "Disappointed. Complete waste of money.",
        "Five stars! Will definitely order more.",
        "Awful. Do not buy this."
    ]
    # 1 for Positive, 0 for Negative
    labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    
    df = pd.DataFrame({'review': reviews, 'sentiment': labels})
    
    # In a real project, we would use a much larger dataset.
    # We will duplicate the dataset a few times to simulate more data for the train_test_split.
    df = pd.concat([df]*10, ignore_index=True)
    
    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.25, random_state=42)
    
    # 3. Preprocessing (TF-IDF Vectorization)
    print("Applying TF-IDF Vectorization...")
    vectorizer = TfidfVectorizer(stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # 4. Modeling (Logistic Regression)
    print("Training Logistic Regression Model...")
    model = LogisticRegression(random_state=42)
    model.fit(X_train_tfidf, y_train)
    
    # 5. Evaluation
    y_pred = model.predict(X_test_tfidf)
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))
    
    # 6. Test with a new review
    new_reviews = [
        "This is the best purchase I have ever made!",
        "I hate this, it is completely useless."
    ]
    new_tfidf = vectorizer.transform(new_reviews)
    predictions = model.predict(new_tfidf)
    
    print("Testing on new custom reviews:")
    for rev, pred in zip(new_reviews, predictions):
        sentiment = "Positive" if pred == 1 else "Negative"
        print(f"Review: '{rev}' => Sentiment: {sentiment}")
        
    print("=== Task 2 Completed Successfully ===\n")

if __name__ == "__main__":
    main()
