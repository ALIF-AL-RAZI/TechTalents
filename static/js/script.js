document.addEventListener('DOMContentLoaded', function() {
    const classifyBtn = document.getElementById('classify-btn');
    const clearBtn = document.getElementById('clear-btn');
    const inputText = document.getElementById('input-text');
    const loading = document.getElementById('loading');
    const resultsContent = document.getElementById('results-content');
    const errorMessage = document.getElementById('error-message');
    const primaryClass = document.getElementById('primary-class');
    const probabilityContainer = document.getElementById('probability-container');
    
    // Calculate API URL based on environment
    const API_URL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' 
        ? '/classify' 
        : '/api/classify';
    
    classifyBtn.addEventListener('click', function() {
        const text = inputText.value.trim();
        
        if (!text) {
            showError('Please enter some text to classify.');
            return;
        }
        
        // Show loading indicator
        loading.classList.remove('hidden');
        resultsContent.classList.add('hidden');
        errorMessage.classList.add('hidden');
        
        // Create form data
        const formData = new FormData();
        formData.append('text', text);
        
        // Send request to API
        fetch(API_URL, {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            // Hide loading indicator
            loading.classList.add('hidden');
            
            if (data.success) {
                // Display results
                resultsContent.classList.remove('hidden');
                
                // Set primary classification
                primaryClass.textContent = data.prediction;
                
                // Clear previous probability bars
                probabilityContainer.innerHTML = '';
                
                // Create probability bars for top classes
                data.probabilities.forEach(item => {
                    const percentage = (item.probability * 100).toFixed(2);
                    
                    const barDiv = document.createElement('div');
                    barDiv.className = 'probability-bar';
                    
                    const labelSpan = document.createElement('span');
                    labelSpan.className = 'probability-label';
                    labelSpan.textContent = item.class;
                    
                    const valueSpan = document.createElement('span');
                    valueSpan.className = 'probability-value';
                    valueSpan.textContent = `${percentage}%`;
                    
                    const fillDiv = document.createElement('div');
                    fillDiv.className = 'probability-fill';
                    
                    barDiv.appendChild(labelSpan);
                    barDiv.appendChild(valueSpan);
                    barDiv.appendChild(fillDiv);
                    probabilityContainer.appendChild(barDiv);
                    
                    // Animate the fill after a small delay
                    setTimeout(() => {
                        fillDiv.style.width = `${percentage}%`;
                    }, 100);
                });
            } else {
                // Show error message
                showError(data.error || 'An error occurred during classification.');
            }
        })
        .catch(error => {
            // Hide loading indicator and show error
            loading.classList.add('hidden');
            showError('Failed to connect to the server. Please try again.');
            console.error('Error:', error);
        });
    });
    
    clearBtn.addEventListener('click', function() {
        // Clear input text
        inputText.value = '';
        
        // Hide results and errors
        resultsContent.classList.add('hidden');
        errorMessage.classList.add('hidden');
    });
    
    function showError(message) {
        errorMessage.textContent = message;
        errorMessage.classList.remove('hidden');
        resultsContent.classList.add('hidden');
    }
});