"""
Script to properly save the trained text classification model and vectorizer.
Run this after training your model to create the correct pickle files.
"""
import joblib
import os
import sys

def save_model_components(model_path, tfidf_path, model, vectorizer):
    """Save model and vectorizer correctly for use with the FastAPI app"""
    print(f"Saving model to: {model_path}")
    joblib.dump(model, model_path)
    
    print(f"Saving TF-IDF vectorizer to: {tfidf_path}")
    joblib.dump(vectorizer, tfidf_path)
    
    print("✅ Model components saved successfully!")

if __name__ == "__main__":
    # Check if we're importing from the training script
    if len(sys.argv) > 1 and sys.argv[1] == "--from-existing":
        # Path to your existing saved model and vectorizer
        existing_model_path = "text_classification_model_T2.pkl"
        existing_vectorizer_path = "tfidf_vectorizer.pkl"
        
        print(f"Loading existing model from: {existing_model_path}")
        model = joblib.load(existing_model_path)
        
        print(f"Loading existing vectorizer from: {existing_vectorizer_path}")
        vectorizer = joblib.load(existing_vectorizer_path)
        
        # Save for FastAPI use
        save_model_components("model.pkl", "vectorizer.pkl", model, vectorizer)
    else:
        print("Please run the training script or use --from-existing flag to save from existing files.")