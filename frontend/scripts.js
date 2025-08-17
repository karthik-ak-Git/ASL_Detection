// Backend configuration
const API_BASE_URL = window.location.origin; // Use the same origin as the frontend
const API_PREDICT_URL = `${API_BASE_URL}/api/predict`;

const form = document.getElementById('upload-form');
const fileInput = document.getElementById('image');
const resultDiv = document.getElementById('result');
const submitButton = form.querySelector('button[type="submit"]');

// File selection handler
fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        const label = document.querySelector('label[for="image"]');
        label.textContent = `Selected: ${file.name}`;
    }
});

form.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!fileInput.files.length) return;

    const fd = new FormData();
    fd.append('file', fileInput.files[0]);

    // Show loading state
    submitButton.disabled = true;
    submitButton.textContent = 'Predicting...';
    resultDiv.className = 'result-area loading';
    resultDiv.textContent = 'Analyzing your ASL sign...';

    try {
        const res = await fetch(API_PREDICT_URL, { method: 'POST', body: fd });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `HTTP ${res.status}`);
        }
        const data = await res.json();

        // Show success state
        resultDiv.className = 'result-area success';
        resultDiv.innerHTML = `
            <h3>Prediction: ${data.label.toUpperCase()}</h3>
            <p>Confidence: ${(data.confidence * 100).toFixed(2)}%</p>
            <details>
                <summary>View all predictions</summary>
                <ul>
                    ${data.top5.map((item, index) =>
            `<li><strong>${index + 1}.</strong> ${item.label.toUpperCase()}: ${(item.confidence * 100).toFixed(2)}%</li>`
        ).join('')}
                </ul>
            </details>
        `;
    } catch (err) {
        // Show error state
        resultDiv.className = 'result-area error';
        resultDiv.innerHTML = `
            <h3>Error</h3>
            <p>${err.message}</p>
            <p>Please try again with a different image.</p>
        `;
    } finally {
        // Reset button state
        submitButton.disabled = false;
        submitButton.textContent = 'Predict ASL Letter';
    }
});
