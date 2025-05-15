# Text Classification Application

This project is a full-stack text classification application that includes:
1. A machine learning model for text classification
2. A Flask backend that serves predictions
3. A responsive web interface for user interaction

![Text Classification App Interface](/pic/1.png)

## Project Structure 

```
text-classifier/
├── api/                     # Vercel serverless functions
│   └── index.py             # Main API entry point
├── models/                  # Directory for your model files
│   ├── logistic_regression_model.pkl
│   └── tfidf_vectorizer.pkl
├── static/                  # Static files
│   ├── css/
│   │   └── style.css        # CSS styles
│   └── js/
│       └── script.js        # JavaScript for frontend
├── templates/               # HTML templates
│   └── index.html           # Main page template
├── app.py                   # Flask application (for local development)
├── preprocessing.py         # Text preprocessing functions
├── requirements.txt         # Project dependencies
└── vercel.json              # Vercel configuration
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- pip (Python package manager)
- Virtual environment (recommended)

### Downloading the Project

1. Clone the repository using Git:

```bash
git clone https://github.com/ALIF-AL-RAZI/techtalents2.git
cd text-classifier
```

Alternatively, you can download the ZIP file from the GitHub repository and extract it to your desired location.

### Installation

1. Create and activate a virtual environment (recommended):

```bash
# On Windows
python -m venv venv
./venv/Scripts/Activate.ps1

# On macOS/Linux
python -m venv venv
source venv/bin/activate
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Verify that all models are present in the `models/` directory:
   - `logistic_regression_model.pkl`
   - `tfidf_vectorizer.pkl`

### Running the Application Locally

1. Start the Flask development server:

```bash
python app.py
```

2. Open your web browser and navigate to:

```
http://127.0.0.1:5000
```

You should see the application interface as shown in the screenshot above.

## Using the Application

![Text Classification Demo](/pic/2.png)

1. Enter or paste the text you want to classify in the input box
2. Click the "Classify" button
3. View the classification results that appear below the input form
4. The application will display:
   - The predicted category
   - Confidence score (when available)


## View Machine Learning Code

[View Machine Learning Code](https://nbviewer.org/github/ALIF-AL-RAZI/techtalents2/blob/main/main3.ipynb)


## Machine Learning Image


![Imbalance Data](/pic/3.png)


![Balance Data](/pic/4.png)


![Multinomial Naive Bayes Confusion Matrix](/pic/5.png)


![Bernoulli Naive Bayes Confusion Matrix](/pic/6.png)


![Logistic Regression Confusion Matrix](/pic/7.png)


![LinearSVC Confusion Matrix](/pic/8.png)


![Random Forest Confusion Matrix](/pic/9.png)


![Accuracy Comparison](/pic/10.png)


![Tuned Model Confusion Matrix](/pic/11.png)


![Ensemble Model Confusion Matrix](/pic/12.png)


## Troubleshooting

### Common Issues

1. **Model files not found**: Ensure all pickle files are in the `models/` directory
2. **Dependencies errors**: Make sure you're using the correct Python version and have installed all requirements
3. **Server not starting**: Check if the port is already in use by another application

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


## Acknowledgments

- scikit-learn for the machine learning tools
- Flask for the web framework
- Vercel for hosting capabilities
