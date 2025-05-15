from flask import Flask, render_template, request, jsonify
import os
import pickle
import nltk
from preprocessing import preprocess_text, initialize_nltk

# Initialize Flask app
app = Flask(__name__)

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')

# Load models
def load_models():
    # Create models directory if not exists
    os.makedirs('models', exist_ok=True)
    
    # Check if models exist, else ask to upload
    vectorizer_path = os.path.join('models', 'tfidf_vectorizer.pkl')
    model_path = os.path.join('models', 'logistic_regression_model.pkl')
    
    if os.path.exists(vectorizer_path) and os.path.exists(model_path):
        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return vectorizer, model
    else:
        print("Models not found. Please place them in the 'models' directory.")
        return None, None

# Initialize NLTK and load models
initialize_nltk()
vectorizer, model = load_models()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    # Get input text from request
    text = request.form.get('text', '')
    
    # Process and classify text
    if text and vectorizer and model:
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
            'prediction': prediction,
            'probabilities': class_probs[:3]
        })
    else:
        return jsonify({
            'success': False,
            'error': 'Either no text was provided or models are not loaded.'
        })

@app.route('/upload', methods=['POST'])
def upload_model():
    # Handle model upload (if needed)
    if 'vectorizer' not in request.files or 'model' not in request.files:
        return jsonify({'success': False, 'error': 'Both vectorizer and model files are required'})
    
    vectorizer_file = request.files['vectorizer']
    model_file = request.files['model']
    
    # Save uploaded files
    os.makedirs('models', exist_ok=True)
    vectorizer_file.save(os.path.join('models', 'tfidf_vectorizer.pkl'))
    model_file.save(os.path.join('models', 'logistic_regression_model.pkl'))
    
    # Reload models
    global vectorizer, model
    vectorizer, model = load_models()
    
    return jsonify({'success': True, 'message': 'Models uploaded successfully'})

if __name__ == "__main__":
    app.run(debug=True)