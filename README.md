# Text Classification Application

This project is a full-stack text classification application that includes:

1. A machine learning model for text classification
2. A FastAPI backend that serves predictions
3. A React frontend for interacting with the model

## Project Structure

```
project/
├── backend/
│   ├── main.py                  # FastAPI application
│   ├── model_debug.py           # Model debugging tool
│   ├── save_model.py            # Helper to properly save models
│   ├── model.pkl                # Saved classification model
│   ├── vectorizer.pkl           # Saved TF-IDF vectorizer
│   └── requirements.txt         # Python dependencies
└── frontend/
    ├── pages/
    │   └── index.tsx            # React frontend
    └── package.json             # Frontend dependencies
```

## Setup Instructions

### 1. Fix the Model Export

If you're getting an error that the model is a NumPy array instead of a scikit-learn model, run:

```bash
cd backend
python save_model.py --from-existing
```

This will load your existing trained model and vectorizer and save them properly for use with the FastAPI app.

### 2. Debug the Model

To check if your model and vectorizer are correctly saved and loaded:

```bash
cd backend
python model_debug.py
```

This will output information about your model and vectorizer and test if they can be used for predictions.

### 3. Start the Backend

```bash
cd backend
pip install -r requirements.txt  # Install dependencies
uvicorn main:app --reload       # Start the FastAPI server
```

The API will be available at http://localhost:8000

### 4. Start the Frontend

```bash
cd frontend
npm install                      # Install dependencies
npm run dev                      # Start the Next.js development server
```

The frontend will be available at http://localhost:3000

## API Endpoints

- `GET /health` - Check if the API is online and model is loaded
- `POST /predict` - Get predictions for text
  - Request Body: `{ "text": "Your text to classify" }`
- `POST /reload_model` - Reload the model and vectorizer from disk

## Model Information

This application is built to work with a text classification model trained on the following classes:

- Mob Justice
- Law and Order
- Politics
- Islamic Fundamentalism
- International affairs
- Religion
- Corruption
- National Defence
- Diplomacy
- Governance & Policy Reform
- Women Rights
- Sports

The model uses TF-IDF for text vectorization and applies preprocessing steps including:
- Text cleaning (URL/hashtag/emoji removal)
- Tokenization
- Stopword removal
- Lemmatization

## Troubleshooting

### Backend Issues

1. **Model loading errors**:
   - Run `python model_debug.py` to diagnose problems
   - Ensure both `model.pkl` and `vectorizer.pkl` exist in the backend directory
   - Use `save_model.py` to properly export your model

2. **Missing dependencies**:
   - Install all required packages: `pip install fastapi uvicorn scikit-learn nltk joblib`
   - For NLTK resources: `python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"`

### Frontend Issues

1. **API connection errors**:
   - Ensure the backend is running on port 8000
   - Check that CORS is properly configured in the backend

2. **Visualization problems**:
   - Make sure all Tailwind CSS dependencies are installed

## Customization

To adapt this application for different classification tasks:

1. Train your model using your dataset and save it following the same structure
2. Update the preprocessing steps in `main.py` to match your training preprocessing
3. Deploy your new model with the same FastAPI/React infrastructure

## License

This project is licensed under the MIT License - see the LICENSE file for details.