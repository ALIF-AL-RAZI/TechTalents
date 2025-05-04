import { useState, useEffect } from 'react';
import Head from 'next/head';

// Define type for API response
interface PredictionResult {
  predicted_class: string;
  confidence: number;
  all_probabilities: Record<string, number>;
}

// Define type for API connection status
type ApiStatus = 'unchecked' | 'online' | 'offline';

export default function Home() {
  const [inputText, setInputText] = useState('');
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [apiStatus, setApiStatus] = useState<ApiStatus>('unchecked');
  const [apiStatusMessage, setApiStatusMessage] = useState('Checking API status...');

  // Check if API is online when component mounts
  useEffect(() => {
    const checkApiStatus = async () => {
      try {
        const response = await fetch('http://localhost:8000/health', { 
          method: 'GET',
          headers: { 'Content-Type': 'application/json' }
        });
        
        if (response.ok) {
          const data = await response.json();
          setApiStatus('online');
          setApiStatusMessage(`API is online. Model: ${data.model_type || 'Unknown'}`);
        } else {
          setApiStatus('offline');
          setApiStatusMessage('API is responding but returned an error');
        }
      } catch (err) {
        setApiStatus('offline');
        setApiStatusMessage('API is offline. Please start the backend server.');
      }
    };
    
    checkApiStatus();
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    // Input validation
    if (!inputText.trim()) {
      setError('Please enter some text to classify');
      return;
    }
    
    // Reset states
    setLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: inputText }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `Server error: ${response.status}`);
      }

      const data = await response.json();
      setPrediction(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An unknown error occurred');
    } finally {
      setLoading(false);
    }
  };

  // Sort probabilities and format them for display
  const sortedProbabilities = prediction?.all_probabilities 
    ? Object.entries(prediction.all_probabilities)
        .sort(([, a], [, b]) => b - a)
        .filter(([, value]) => value > 0.01) // Filter out very low probabilities
    : [];

  return (
    <div className="min-h-screen bg-gray-100 py-6 flex flex-col justify-center sm:py-12">
      <Head>
        <title>Text Classification App</title>
        <meta name="description" content="Text classification using machine learning model" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <div className="relative py-3 sm:max-w-xl sm:mx-auto">
        <div className="relative px-4 py-10 bg-white shadow-lg sm:rounded-3xl sm:p-20">
          <div className="max-w-md mx-auto">
            <div>
              <h1 className="text-2xl font-semibold text-center">Text Classification</h1>
              
              {/* API Status Indicator */}
              <div className={`mt-2 text-sm text-center ${
                apiStatus === 'online' ? 'text-green-600' : 
                apiStatus === 'offline' ? 'text-red-600' : 'text-yellow-600'
              }`}>
                <div className="flex items-center justify-center">
                  <div className={`w-3 h-3 rounded-full mr-2 ${
                    apiStatus === 'online' ? 'bg-green-500' : 
                    apiStatus === 'offline' ? 'bg-red-500' : 'bg-yellow-500'
                  }`}></div>
                  {apiStatusMessage}
                </div>
              </div>
            </div>
            
            <form onSubmit={handleSubmit} className="mt-8 space-y-6">
              <div>
                <label htmlFor="text" className="text-sm font-medium text-gray-700">
                  Enter text to classify:
                </label>
                <div className="mt-1">
                  <textarea
                    id="text"
                    name="text"
                    rows={4}
                    className="shadow-sm focus:ring-indigo-500 focus:border-indigo-500 mt-1 block w-full sm:text-sm border border-gray-300 rounded-md p-2"
                    placeholder="Type your text here..."
                    value={inputText}
                    onChange={(e) => setInputText(e.target.value)}
                    required
                  />
                </div>
              </div>

              <div>
                <button
                  type="submit"
                  disabled={loading || inputText.trim() === '' || apiStatus !== 'online'}
                  className={`w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 ${
                    (loading || inputText.trim() === '' || apiStatus !== 'online') ? 'opacity-50 cursor-not-allowed' : ''
                  }`}
                >
                  {loading ? 'Processing...' : 'Classify Text'}
                </button>
              </div>
            </form>

            {error && (
              <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-md">
                <p className="text-red-600 text-sm">{error}</p>
                {apiStatus === 'offline' && (
                  <p className="text-red-600 text-xs mt-1">
                    Make sure the FastAPI backend is running on port 8000.
                  </p>
                )}
              </div>
            )}

            {prediction && (
              <div className="mt-8 border rounded-md p-4 bg-gray-50">
                <h2 className="text-lg font-medium text-gray-900">Prediction Results</h2>
                <div className="mt-2">
                  <div className="flex items-center bg-indigo-50 p-3 rounded-md border border-indigo-100">
                    <div className="text-indigo-700 text-lg font-semibold flex-grow">
                      {prediction.predicted_class}
                    </div>
                    <div className="text-indigo-600 font-medium">
                      {(prediction.confidence * 100).toFixed(1)}%
                    </div>
                  </div>
                  
                  {sortedProbabilities.length > 0 && (
                    <div className="mt-4">
                      <p className="font-medium text-gray-700 mb-2">All Class Probabilities:</p>
                      <div className="space-y-2">
                        {sortedProbabilities.map(([className, probability]) => (
                          <div key={className} className="bg-white p-2 rounded border border-gray-200">
                            <div className="flex justify-between text-sm">
                              <span>{className}</span>
                              <span className="font-medium">{(probability * 100).toFixed(1)}%</span>
                            </div>
                            <div className="mt-1 w-full bg-gray-200 rounded-full h-1.5">
                              <div 
                                className="bg-indigo-600 h-1.5 rounded-full" 
                                style={{ width: `${probability * 100}%` }}
                              ></div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}