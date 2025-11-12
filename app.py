from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load the trained model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "heartsense_pipeline.pkl")

try:
    pipeline = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully!")
except FileNotFoundError:
    print("❌ Model not found. Please train the model first by running train_model.py")
    pipeline = None

@app.route('/')
def home():
    """Render the home page with the prediction form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    if pipeline is None:
        return jsonify({
            'error': 'Model not loaded. Please train the model first.'
        }), 500
    
    try:
        # Get form data
        data = {
            'age': int(request.form['age']),
            'sex': int(request.form['sex']),
            'cp': int(request.form['cp']),
            'trestbps': int(request.form['trestbps']),
            'chol': int(request.form['chol']),
            'fbs': int(request.form['fbs']),
            'restecg': int(request.form['restecg']),
            'thalach': int(request.form['thalach']),
            'exang': int(request.form['exang']),
            'oldpeak': float(request.form['oldpeak']),
            'slope': int(request.form['slope']),
            'ca': int(request.form['ca']),
            'thal': int(request.form['thal'])
        }
        
        # Create DataFrame with the same column order as training
        df = pd.DataFrame([data])
        
        # Make prediction
        prediction = pipeline.predict(df)[0]
        probability = pipeline.predict_proba(df)[0]
        
        # Prepare response
        result = {
            'prediction': int(prediction),
            'probability_no_disease': float(probability[0] * 100),
            'probability_disease': float(probability[1] * 100),
            'risk_level': get_risk_level(probability[1])
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

def get_risk_level(prob):
    """Determine risk level based on probability"""
    if prob < 0.3:
        return 'Low'
    elif prob < 0.6:
        return 'Moderate'
    elif prob < 0.8:
        return 'High'
    else:
        return 'Very High'

@app.route('/about')
def about():
    """About page with model information"""
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)