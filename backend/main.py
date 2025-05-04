"""
Improved FastAPI backend for text classification.
This version correctly handles the model and vectorizer, and includes text preprocessing.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import os
from typing import Dict, List, Any
import re
import string
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

# Download required NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    from nltk.corpus import stopwords
    en_stop = set(stopwords.words('english'))
except Exception as e:
    print(f"Warning: NLTK download issue - {e}")
    en_stop = set()

app = FastAPI(title="Text Classification API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths to model and vectorizer files
MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

# Global variables for model and vectorizer
model = None
vectorizer = None

# Load model and vectorizer when app starts
@app.on_event("startup")
async def load_model_on_startup():
    global model, vectorizer
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            print(f"Model loaded successfully: {type(model)}")
        else:
            print(f"Warning: Model file not found at {MODEL_PATH}")
        
        if os.path.exists(VECTORIZER_PATH):
            vectorizer = joblib.load(VECTORIZER_PATH)
            print(f"Vectorizer loaded successfully: {type(vectorizer)}")
        else:
            print(f"Warning: Vectorizer file not found at {VECTORIZER_PATH}")
    except Exception as e:
        print(f"Error loading model or vectorizer: {e}")

# Text preprocessing function (simplified version of the one in training)
def preprocess_text(text):
    # Ensure text is string
    text = str(text)
    
    # Remove URLs, hashtags, mentions
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"@\w+", "", text)
    
    # Remove punctuations
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    
    # Normalize space
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Tokenize, lowercase, remove stopwords, and lemmatize
    tokens = word_tokenize(text.lower())
    clean_tokens = [lemmatizer.lemmatize(token) for token in tokens 
                   if token.isalpha() and token not in en_stop]
    
    return ' '.join(clean_tokens)

class TextRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    all_probabilities: Dict[str, float]

@app.get("/")
def read_root():
    return {"message": "Text Classification API"}

@app.get("/health")
def health_check():
    if model is None or vectorizer is None:
        components_missing = []
        if model is None:
            components_missing.append("model")
        if vectorizer is None:
            components_missing.append("vectorizer")
        raise HTTPException(
            status_code=503, 
            detail=f"Components not loaded: {', '.join(components_missing)}"
        )
    
    return {
        "status": "ok", 
        "model_type": str(type(model)),
        "vectorizer_type": str(type(vectorizer))
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: TextRequest):
    if model is None or vectorizer is None:
        raise HTTPException(
            status_code=503, 
            detail="Model or vectorizer not loaded"
        )
    
    try:
        # Get and clean text
        text = request.text
        print(f"Received text: '{text[:50]}...'")
        
        cleaned_text = preprocess_text(text)
        print(f"Cleaned text: '{cleaned_text[:50]}...'")
        
        # Vectorize using TF-IDF
        text_tfidf = vectorizer.transform([cleaned_text])
        print(f"Vectorized text shape: {text_tfidf.shape}")
        
        # Get prediction
        prediction = model.predict(text_tfidf)[0]
        
        # Get prediction probabilities
        probabilities = model.predict_proba(text_tfidf)[0]
        predicted_class_idx = probabilities.argmax()
        confidence = float(probabilities[predicted_class_idx])
        
        # Build probabilities dictionary
        all_probs = {
            str(model.classes_[i]): float(probabilities[i]) 
            for i in range(len(model.classes_))
        }
        
        print(f"Prediction: {prediction} with confidence {confidence:.4f}")
        
        return {
            "predicted_class": str(prediction),
            "confidence": confidence,
            "all_probabilities": all_probs
        }
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Prediction error: {str(e)}\n{error_details}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/reload_model")
async def reload_model():
    global model, vectorizer
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
        else:
            raise HTTPException(status_code=404, detail=f"Model file not found at {MODEL_PATH}")
        
        if os.path.exists(VECTORIZER_PATH):
            vectorizer = joblib.load(VECTORIZER_PATH)
        else:
            raise HTTPException(status_code=404, detail=f"Vectorizer file not found at {VECTORIZER_PATH}")
        
        return {"message": "Model and vectorizer reloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reloading components: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)