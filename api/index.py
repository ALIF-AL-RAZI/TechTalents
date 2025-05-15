from flask import Flask, request, jsonify
import os
import pickle
import sys
import importlib.util

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import preprocessing module
from preprocessing import preprocess_text, initialize_nltk

# Initialize Flask app
app = Flask(__name__)

# Initialize NLTK
initialize_nltk()

# Load models
def load_models():
    try:
        # Models should be in the /models directory
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        vectorizer_path = os.path.join(base_dir, 'models', 'tfidf_vectorizer.pkl')
        model_path = os.path.join(base_dir, 'models', 'logistic_regression_model.pkl')
        
        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return vectorizer, model
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None

# Load models globally - this will be cached by Vercel
vectorizer, model = load_models()

@app.route('/', methods=['GET'])
def home():
    return jsonify({"status": "API is running", "message": "Use POST /api/classify to classify text"})

@app.route('/classify', methods=['POST'])
def classify():
    # Check if request is JSON
    if request.is_json:
        data = request.get_json()
        text = data.get('text', '')
    else:
        # Handle form data
        text = request.form.get('text', '')
        
        # If no form data, check if it's in query params
        if not text:
            text = request.args.get('text', '')
    
    # Process and classify text
    if text and vectorizer and model:
        try:
            # Preprocess text
            processed_text = preprocess_text(text)
            
            # Vectorize text
            text_vector = vectorizer.transform([processed_text])
            
            # Make prediction
            prediction = model.predict(text_vector)[0]
            
            # Get prediction probabilities
            probabilities = model.predict_proba(text_vector)[0]
            
            # Create list of class probabilities
            class_probs = []
            for i, class_name in enumerate(model.classes_):
                class_probs.append({
                    'class': class_name,
                    'probability': float(probabilities[i])
                })
            
            # Sort by probability (descending)
            class_probs.sort(key=lambda x: x['probability'], reverse=True)
            
            # Return top 3 classes with probabilities
            return jsonify({
                'success': True,
                'original_text': text,
                'processed_text': processed_text,
                'prediction': prediction,
                'probabilities': class_probs[:3]
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Error processing text: {str(e)}'
            })
    else:
        return jsonify({
            'success': False,
            'error': 'Either no text was provided or models are not loaded.'
        })

# This handler is used for Vercel serverless functions
handler = app