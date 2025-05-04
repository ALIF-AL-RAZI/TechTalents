"""
Enhanced model debugging script that checks both the model and vectorizer.
This helps ensure both components are correctly loaded and compatible.
"""
import pickle
import joblib
import sys
import os
import inspect
import numpy as np

def inspect_component(component, name):
    """Inspect a model or vectorizer component"""
    print(f"\n=== {name} Information ===")
    print(f"Type: {type(component)}")
    
    # Check if component is a class instance
    if hasattr(component, "__class__"):
        print(f"Class name: {component.__class__.__name__}")
        
        # Check parent classes
        bases = component.__class__.__bases__
        if bases:
            print("Parent classes:")
            for base in bases:
                print(f"  - {base.__name__}")
    
    # List available methods
    methods = [method for method in dir(component) if callable(getattr(component, method)) and not method.startswith("_")]
    
    # Important methods to check
    if name == "Model":
        key_methods = ["predict", "predict_proba", "decision_function", "fit"]
    else:  # Vectorizer
        key_methods = ["transform", "fit_transform", "fit", "get_feature_names_out"]
    
    print("\nKey Methods:")
    for method in key_methods:
        if method in methods:
            print(f"✓ {method}")
            try:
                func = getattr(component, method)
                signature = inspect.signature(func)
                print(f"  Signature: {method}{signature}")
            except:
                print(f"  Signature: [Could not determine]")
        else:
            print(f"✗ {method} [Not found]")
    
    # Check important attributes
    print("\nImportant Attributes:")
    if name == "Model":
        key_attrs = ["classes_", "n_features_in_", "feature_names_in_"]
    else:  # Vectorizer
        key_attrs = ["vocabulary_", "idf_", "stop_words_"]
    
    for attr in key_attrs:
        if hasattr(component, attr):
            value = getattr(component, attr)
            print(f"✓ {attr}")
            
            if attr == "classes_":
                print(f"  Classes: {value}")
                print(f"  Number of classes: {len(value)}")
            elif attr == "vocabulary_":
                vocab_size = len(value) if isinstance(value, dict) else "Unknown"
                print(f"  Vocabulary size: {vocab_size}")
                # Show a few vocabulary items
                if isinstance(value, dict):
                    sample_items = list(value.items())[:5]
                    print(f"  Sample vocabulary items: {sample_items}")
            else:
                # Handle different types of values
                if isinstance(value, np.ndarray):
                    print(f"  Shape: {value.shape}")
                    if value.size < 10:
                        print(f"  Value: {value}")
                    else:
                        print(f"  First few values: {value[:5]}")
                else:
                    print(f"  Value: {value}")
        else:
            print(f"✗ {attr} [Not found]")

def load_and_inspect_components():
    """Load and inspect both model and vectorizer"""
    model_path = "model.pkl"
    vectorizer_path = "vectorizer.pkl"
    
    # Check for model file
    if not os.path.exists(model_path):
        print(f"❌ Model file '{model_path}' not found in current directory!")
        print(f"Current directory: {os.getcwd()}")
        print("Available files:")
        for file in os.listdir():
            print(f"- {file}")
    else:
        print(f"✓ Found model file: {model_path}")
    
    # Check for vectorizer file
    if not os.path.exists(vectorizer_path):
        print(f"❌ Vectorizer file '{vectorizer_path}' not found in current directory!")
    else:
        print(f"✓ Found vectorizer file: {vectorizer_path}")
    
    # Try to load model
    model = None
    try:
        print(f"\nLoading model from: {model_path}")
        model = joblib.load(model_path)
        print("✅ Model loaded successfully!")
        inspect_component(model, "Model")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
    
    # Try to load vectorizer
    vectorizer = None
    try:
        print(f"\nLoading vectorizer from: {vectorizer_path}")
        vectorizer = joblib.load(vectorizer_path)
        print("✅ Vectorizer loaded successfully!")
        inspect_component(vectorizer, "Vectorizer")
    except Exception as e:
        print(f"❌ Failed to load vectorizer: {e}")
        import traceback
        traceback.print_exc()
    
    # Try making a prediction if both components loaded
    if model is not None and vectorizer is not None:
        print("\n=== Testing Prediction Pipeline ===")
        test_text = "This is a test sentence for prediction."
        
        try:
            print(f"Input text: '{test_text}'")
            
            # Try to preprocess text (simplified)
            clean_text = test_text.lower()
            print(f"Preprocessed text: '{clean_text}'")
            
            # Vectorize text
            text_vector = vectorizer.transform([clean_text])
            print(f"Vector shape: {text_vector.shape}")
            
            # Make prediction
            prediction = model.predict(text_vector)
            print(f"Prediction result: {prediction}")
            
            # Get probabilities
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(text_vector)
                print(f"Prediction probabilities shape: {probabilities.shape}")
                print(f"Prediction probabilities: {probabilities}")
                
                # Get class with highest probability
                predicted_class_idx = probabilities.argmax(axis=1)[0]
                predicted_class = model.classes_[predicted_class_idx]
                print(f"Predicted class: {predicted_class}")
                print(f"Probability: {probabilities[0][predicted_class_idx]:.4f}")
            
            print("\n✅ Prediction pipeline works correctly!")
        except Exception as e:
            print(f"❌ Prediction test failed: {e}")
            import traceback
            traceback.print_exc()
            print("\nTo fix this issue, make sure your model and vectorizer are saved correctly "
                  "and that your preprocessing steps match what was used during training.")

if __name__ == "__main__":
    load_and_inspect_components()