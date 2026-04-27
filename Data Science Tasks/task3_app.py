from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model_path = os.path.join('models', 'iris_model.pkl')
try:
    model = joblib.load(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Mapping for class indices to names
class_names = ['Setosa', 'Versicolor', 'Virginica']

@app.route('/')
def home():
    # Serve the HTML frontend
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded.'}), 500
        
    try:
        # Get data from POST request (JSON)
        data = request.get_json()
        
        # Extract features
        features = [
            float(data['sepal_length']),
            float(data['sepal_width']),
            float(data['petal_length']),
            float(data['petal_width'])
        ]
        
        # Make prediction
        prediction_idx = model.predict([features])[0]
        prediction_prob = model.predict_proba([features])[0]
        
        result = {
            'class_index': int(prediction_idx),
            'class_name': class_names[prediction_idx],
            'confidence': float(np.max(prediction_prob))
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
