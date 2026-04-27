import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import os

def main():
    print("=== Virtual Technologies Internship: Task 1 - Decision Tree Implementation ===")
    
    # 1. Load Data
    print("Loading Breast Cancer dataset...")
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    
    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Build Model
    print("Training Decision Tree Classifier...")
    # Limiting depth so the visualization is readable
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X_train, y_train)
    
    # 4. Evaluate Model
    y_pred = clf.predict(X_test)
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=data.target_names))
    
    # 5. Visualize Tree
    print("\nGenerating tree visualization...")
    os.makedirs('visualizations', exist_ok=True)
    plt.figure(figsize=(20, 10))
    plot_tree(clf, 
              feature_names=data.feature_names, 
              class_names=data.target_names, 
              filled=True, 
              rounded=True, 
              fontsize=10)
    
    viz_path = os.path.join('visualizations', 'decision_tree.png')
    plt.savefig(viz_path)
    print(f"Decision tree visualization saved to {viz_path}")
    print("=== Task 1 Completed Successfully ===\n")

if __name__ == "__main__":
    main()
