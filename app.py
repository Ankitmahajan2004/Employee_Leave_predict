@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template_string(HTML_TEMPLATE, prediction="Error: Model file 'logistic_model.pkl' not loaded.")
    
    try:
        raw_input = request.form.get('features', '')
        
        # String ला Float Numbers मध्ये कन्व्हर्ट करा
        feature_list = [float(x.strip()) for x in raw_input.split(',') if x.strip()]
        
        if not feature_list:
            raise ValueError("कृपया योग्य नंबर्स प्रविष्ट करा (Enter valid numbers).")
            
        features = np.array([feature_list])
        
        # Model Prediction
        prediction_val = model.predict(features)[0]
        
        probability_val = None
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(features)[0]
            probability_val = round(np.max(probs) * 100, 2)
            
        return render_template_string(
            HTML_TEMPLATE, 
            prediction=f"Result: {prediction_val}", 
            probability=probability_val,
            raw_input=raw_input
        )
        
    except ValueError as ve:
        # जर इनपुट नंबर नसेल किंवा रिकामा असेल
        return render_template_string(
            HTML_TEMPLATE, 
            prediction="Input Error: योग्य नंबर स्वल्पविरामाने (comma) अलग करून टाका.", 
            raw_input=request.form.get('features', '')
        )
    except Exception as e:
        # जर Model Shape / Feature count मॅच होत नसेल, तर नक्की काय Error आहे ते दाखवेल
        return render_template_string(
            HTML_TEMPLATE, 
            prediction=f"Model Error: {str(e)}", 
            raw_input=request.form.get('features', '')
        )
