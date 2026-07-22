import pickle
import numpy as np
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Load the logistic regression model
try:
    with open("logistic_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

# Embedded HTML Template with inline CSS layout
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
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5), 0 8px 10px -6px rgba(0, 0, 0, 0.5);
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
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
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
            transition: background-color 0.2s ease, transform 0.1s ease;
            margin-top: 10px;
        }

        .btn-submit:hover {
            background-color: var(--accent-hover);
        }

        .btn-submit:active {
            transform: scale(0.98);
        }

        .result-card {
            margin-top: 28px;
            padding: 16px;
            border-radius: 8px;
            background-color: rgba(16, 185, 129, 0.1);
            border: 1px solid var(--success-color);
            text-align: center;
        }

        .result-card h3 {
            font-size: 0.875rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 4px;
        }

        .result-card .prediction-value {
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--success-color);
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
        <div class="result-card">
            <h3>Prediction Result</h3>
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


@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_TEMPLATE, prediction=None)


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template_string(
            HTML_TEMPLATE,
            prediction="Error: Model file 'logistic_model.pkl' not loaded.",
        )

    try:
        raw_input = request.form.get("features", "")
        # Parse comma-separated inputs into floats
        feature_list = [float(x.strip()) for x in raw_input.split(",") if x.strip()]
        features = np.array([feature_list])

        # Run model inference
        prediction_val = model.predict(features)[0]

        # Calculate prediction probability if supported
        probability_val = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(features)[0]
            probability_val = round(np.max(probs) * 100, 2)

        return render_template_string(
            HTML_TEMPLATE,
            prediction=prediction_val,
            probability=probability_val,
            raw_input=raw_input,
        )

    except Exception as e:
        return render_template_string(
            HTML_TEMPLATE,
            prediction=f"Input Error: Ensure numbers are comma-separated.",
            raw_input=request.form.get("features", ""),
        )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
