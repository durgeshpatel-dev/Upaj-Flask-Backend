from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return jsonify({
        'message': 'Upaj ML Backend API',
        'status': 'running',
        'version': '1.0.0',
        'endpoints': ['/health', '/predict', '/predict-disease']
    })

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'message': 'API is running perfectly!'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        crop = data.get('Crop', 'RICE')
        
        # Mock predictions based on crop
        predictions = {
            'RICE': 3500,
            'WHEAT': 4200,
            'GROUNDNUT': 2800,
            'MAIZE': 3800,
            'SUGARCANE': 65000
        }
        
        return jsonify({
            'prediction': predictions.get(crop, 3000),
            'crop': crop,
            'confidence': 0.85,
            'status': 'success',
            'message': 'Yield prediction completed'
        })
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/predict-disease', methods=['POST'])
def predict_disease():
    try:
        diseases = ['Healthy', 'Leaf_Rust', 'Common_rust', 'Northern_Leaf_Blight']
        import random
        
        return jsonify({
            'predicted_class': random.choice(diseases),
            'confidence': round(random.uniform(0.7, 0.95), 2),
            'status': 'success',
            'message': 'Disease prediction completed'
        })
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)