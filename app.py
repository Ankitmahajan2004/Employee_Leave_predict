import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request

# IMPORTANT FOR VERCEL: 'app' must be defined globally at top-level
app = Flask(__name__)

# Dynamically resolve model path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'logistic_model.pkl')

# Model loader function
def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, 'rb') as model_file:
                return pickle.load(model_file), None
        except Exception as e:
            return None, f"Model load error: {str(e)}"
    return None, f"File 'logistic_model.pkl' not found at {MODEL_PATH}"

# HTML Interface Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ML Prediction Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --accent-color: #38bdf8;
            --accent-hover: #0284c7;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --border-color: #334155;
            --success-color: #10b981;
            --error-color: #ef4444;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
        }

        body {
            background-color: var(--bg-color);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }

        .container {
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5);
            width: 100%;
            max-width: 480px;
            padding: 32px;
        }

        .header {
            text-align: center;
            margin-bottom: 28px;
        }

        .header h1 {
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--text-main);
            margin-bottom: 8px;
        }

        .header p {
            font-size: 0.875rem;
            color: var(--text-muted);
        }

        .form-group {
            margin-bottom: 20px;
        }

        .form-group label {
            display: block;
            font-size: 0.875rem;
            font-weight: 600;
            margin-bottom: 8px;
            color: var(--text-main);
        }

        .form-control {
            width: 100%;
            padding: 12px 16px;
            border-radius: 8px;
            border: 1px solid var(--border-color);
            background-color: #0f172a;
            color: var(--text-main);
            font-size: 0.95rem;
        }

        .form-control:focus {
            outline: none;
            border-color: var(--accent-color);
            box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.2);
        }

        .btn-submit {
            width: 100%;
            padding: 12px;
            border: none;
            border-radius: 8px;
            background-color: var(--accent-color);
            color: #0f172a;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            margin-top: 10px;
        }

        .btn-submit:hover {
            background-color: var(--accent-hover);
        }

        .result-card {
            margin-top: 28px;
            padding: 16px;
            border-radius: 8px;
            background-color: rgba(16, 185, 129, 0.1);
            border: 1px solid var(--success-color);
            text-align: center;
        }

        .result-card.error {
            background-color: rgba(239, 68, 68, 0.1);
            border-color: var(--error-color);
        }

        .result-card h3 {
            font-size: 0.875rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 4px;
        }

        .result-card .prediction-value {
            font-size: 1.1rem;
            font-weight: 700;
            color: var(--success-color);
            word-break: break-word;
        }

        .result-card.error .prediction-value {
            color: var(--error-color);
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Logistic Regression Model</h1>
            <p>Enter feature inputs to run inference</p>
        </div>

        <form action="/predict" method="POST">
            <div class="form-group">
                <label for="feature_input">Feature Inputs (Comma-separated)</label>
                <input 
                    type="text" 
                    id="feature_input" 
                    name="features" 
                    class="form-control" 
                    placeholder="e.g., 2.5, 1.0, 0.5, 3.2" 
                    value="{{ raw_input if raw_input else '' }}"
                    required
                >
            </div>

            <button type="submit" class="btn-submit">Predict</button>
        </form>

        {% if prediction is not none %}
        <div class="result-card {{ 'error' if is_error else '' }}">
            <h3>{{ "ERROR" if is_error else "PREDICTION RESULT" }}</h3>
            <div class="prediction-value">{{ prediction }}</div>
            {% if probability is not none %}
                <p style="font-size: 0.85rem; color: var(--text-muted); margin-top: 6px;">
                    Confidence: {{ probability }}%
                </p>
            {% endif %}
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def home():
    model, error = load_model()
    if error:
        return render_template_string(HTML_TEMPLATE, prediction=error, is_error=True, probability=None)
    return render_template_string(HTML_TEMPLATE, prediction=None, is_error=False, probability=None)

@app.route('/predict', methods=['POST'])
def predict():
    model, error = load_model()
    raw_input = request.form.get('features', '')

    if error:
        return render_template_string(HTML_TEMPLATE, prediction=error, is_error=True, probability=None, raw_input=raw_input)

    try:
        feature_list = [float(x.strip()) for x in raw_input.split(',') if x.strip()]
        if not feature_list:
            raise ValueError("Empty input list")

        features = np.array([feature_list])
        prediction_val = model.predict(features)[0]

        probability_val = None
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(features)[0]
            probability_val = round(float(np.max(probs)) * 100, 2)

        return render_template_string(
            HTML_TEMPLATE, 
            prediction=f"Result: {prediction_val}", 
            probability=probability_val, 
            raw_input=raw_input, 
            is_error=False
        )

    except ValueError:
        return render_template_string(
            HTML_TEMPLATE, 
            prediction="Input Error: Enter numbers separated by commas.", 
            raw_input=raw_input, 
            is_error=True, 
            probability=None
        )
    except Exception as e:
        return render_template_string(
            HTML_TEMPLATE, 
            prediction=f"Inference Error: {str(e)}", 
            raw_input=raw_input, 
            is_error=True, 
            probability=None
        )

# Required explicit WSGI reference for Vercel
app = app

if __name__ == "__main__":
    app.run(debug=True, port=5000)
