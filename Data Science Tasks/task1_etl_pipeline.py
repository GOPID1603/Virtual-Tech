import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import os

def generate_synthetic_data(num_records=1000, output_file='raw_data.csv'):
    """Generates synthetic data to simulate a messy dataset."""
    print("Generating synthetic data...")
    np.random.seed(42)
    
    # Numerical features
    age = np.random.normal(35, 10, num_records)
    income = np.random.normal(60000, 15000, num_records)
    
    # Categorical features
    departments = ['HR', 'Engineering', 'Sales', 'Marketing', 'IT']
    department = np.random.choice(departments, num_records)
    
    cities = ['New York', 'San Francisco', 'Chicago', 'Austin', 'Seattle']
    city = np.random.choice(cities, num_records)
    
    # Target variable (e.g., Performance Score)
    performance = np.random.randint(1, 100, num_records)
    
    df = pd.DataFrame({
        'Age': age,
        'Income': income,
        'Department': department,
        'City': city,
        'Performance': performance
    })
    
    # Inject missing values randomly
    # 10% missing values in Age
    df.loc[df.sample(frac=0.1, random_state=1).index, 'Age'] = np.nan
    # 5% missing values in Department
    df.loc[df.sample(frac=0.05, random_state=2).index, 'Department'] = np.nan
    
    df.to_csv(output_file, index=False)
    print(f"Raw data saved to {output_file}\n")
    return output_file

def main():
    print("=== CODTECH Internship: Task 1 - Data Pipeline Development ===\n")
    
    # 1. Generate Data
    raw_data_path = generate_synthetic_data()
    
    # 2. Extract (Load from CSV)
    print("Extracting data...")
    df = pd.read_csv(raw_data_path)
    print("Data summary before transformation:")
    print(df.info())
    print("\nMissing values before:")
    print(df.isnull().sum())
    print("-" * 50)
    
    # Separate features and target
    X = df.drop('Performance', axis=1)
    y = df['Performance']
    
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Transform (Preprocessing Pipeline)
    print("\nSetting up the preprocessing pipeline...")
    
    # Identify numerical and categorical columns
    numeric_features = ['Age', 'Income']
    categorical_features = ['Department', 'City']
    
    # Create preprocessing steps for numerical data (Impute missing with median, then scale)
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Create preprocessing steps for categorical data (Impute missing with constant, then one-hot encode)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine both transformers into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Fit the preprocessor and transform the training data
    print("Transforming the data...")
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    # Get feature names after transformation for the DataFrame
    cat_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
    feature_names = numeric_features + list(cat_feature_names)
    
    # Convert back to DataFrame for saving
    X_train_transformed_df = pd.DataFrame(X_train_transformed, columns=feature_names)
    
    # 4. Load (Save transformed data)
    transformed_file = 'transformed_data.csv'
    print(f"\nLoading (saving) transformed data to {transformed_file}...")
    X_train_transformed_df.to_csv(transformed_file, index=False)
    
    print("\nData summary after transformation (first 5 rows):")
    print(X_train_transformed_df.head())
    print("\nMissing values after transformation:")
    print(X_train_transformed_df.isnull().sum().sum()) # Should be 0
    print("\n=== Pipeline Execution Completed Successfully ===")

if __name__ == "__main__":
    main()
