from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import traceback
from datetime import datetime

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB limit
CORS(app)

# Global variables for model status
model_status = {
    'yield_model': False,
    'disease_model': False,
    'startup_time': datetime.now().isoformat()
}

# Configuration data
valid_soil_types = ['Sandy', 'Alluvial', 'Black', 'Red-Yellow', 'Red', 'Loamy']
valid_crops = ['RICE', 'GROUNDNUT', 'WHEAT', 'MAIZE', 'SUGARCANE']
valid_states = ['Chhattisgarh', 'Madhya Pradesh', 'West Bengal', 'Bihar', 'Jharkhand', 'Orissa', 'Gujarat']
predictable_crops = ['RICE', 'GROUNDNUT']

# Mock prediction data for when models aren't available
mock_predictions = {
    'RICE': {'yield': 3500, 'confidence': 0.85},
    'GROUNDNUT': {'yield': 2800, 'confidence': 0.82},
    'WHEAT': {'yield': 4200, 'confidence': 0.88},
    'MAIZE': {'yield': 3800, 'confidence': 0.86},
    'SUGARCANE': {'yield': 65000, 'confidence': 0.90}
}

disease_classes = [
    'Corn___Common_rust', 'Corn___Healthy', 'Corn___Northern_Leaf_Blight',
    'Cotton___diseased_cotton_leaf', 'Cotton___fresh_cotton_leaf',
    'Rice___Bacterial_leaf_blight', 'Rice___Brown_spot', 'Rice___Healthy',
    'Wheat___Healthy', 'Wheat___Leaf_Rust'
]

@app.route('/', methods=['GET'])
def root():
    """Root endpoint with API information"""
    return jsonify({
        'message': 'Upaj Flask ML Backend API',
        'version': '1.0.0',
        'status': 'running',
        'platform': 'Vercel Serverless',
        'startup_time': model_status['startup_time'],
        'endpoints': {
            'health': '/health',
            'predict_yield': '/predict',
            'predict_disease': '/predict-disease',
            'test': '/test'
        },
        'supported_crops': valid_crops,
        'supported_states': valid_states,
        'soil_types': valid_soil_types
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'platform': 'Vercel Serverless',
        'python_version': f"{os.sys.version}",
        'environment': 'production'
    })

@app.route('/predict', methods=['POST'])
def predict_yield():
    """Crop yield prediction endpoint"""
    try:
        print(f"\n=== YIELD PREDICTION REQUEST ===")
        
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        
        data = request.get_json()
        print(f"Received data: {data}")
        
        # Required fields validation
        required_fields = ['Year', 'State', 'District', 'Area_1000_ha', 'Crop', 'Avg_Rainfall_mm', 'Avg_Temp_C', 'Soil_Type']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({
                'error': 'Missing required fields',
                'missing_fields': missing_fields,
                'required_fields': required_fields
            }), 400
        
        # Validate crop
        if data['Crop'] not in valid_crops:
            return jsonify({
                'error': f"Invalid crop. Supported crops: {valid_crops}"
            }), 400
        
        # Validate state
        if data['State'] not in valid_states:
            return jsonify({
                'error': f"Invalid state. Supported states: {valid_states}"
            }), 400
        
        # Validate soil type
        if data['Soil_Type'] not in valid_soil_types:
            return jsonify({
                'error': f"Invalid soil type. Supported types: {valid_soil_types}"
            }), 400
        
        # Use mock data (serverless-friendly)
        mock_data = mock_predictions.get(data['Crop'], {'yield': 3000, 'confidence': 0.80})
        result = {
            'prediction': mock_data['yield'],
            'confidence': mock_data['confidence'],
            'model_type': 'mock_serverless',
            'status': 'success',
            'input_data': data,
            'timestamp': datetime.now().isoformat(),
            'units': 'kg/hectare'
        }
        
        print(f"Prediction result: {result}")
        return jsonify(result)
        
    except Exception as e:
        error_msg = str(e)
        print(f"ERROR in yield prediction: {error_msg}")
        print(f"Traceback: {traceback.format_exc()}")
        
        return jsonify({
            'error': 'Internal server error during prediction',
            'message': error_msg,
            'status': 'error',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/predict-disease', methods=['POST'])
def predict_disease():
    """Crop disease prediction endpoint"""
    try:
        print(f"\n=== DISEASE PREDICTION REQUEST ===")
        
        if 'image' not in request.files:
            return jsonify({'error': 'No image file uploaded'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Validate file type
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        file_ext = os.path.splitext(file.filename.lower())[1]
        
        if file_ext not in allowed_extensions:
            return jsonify({
                'error': f'Invalid file type. Allowed: {list(allowed_extensions)}'
            }), 400
        
        print(f"Processing image: {file.filename}")
        
        # Use mock data (serverless-friendly)
        import random
        predicted_class = random.choice(disease_classes)
        result = {
            'predicted_class': predicted_class,
            'confidence': round(random.uniform(0.7, 0.95), 2),
            'model_type': 'mock_serverless',
            'status': 'success',
            'filename': file.filename,
            'timestamp': datetime.now().isoformat(),
            'all_classes': disease_classes
        }
        
        print(f"Disease prediction result: {result}")
        return jsonify(result)
        
    except Exception as e:
        error_msg = str(e)
        print(f"ERROR in disease prediction: {error_msg}")
        print(f"Traceback: {traceback.format_exc()}")
        
        return jsonify({
            'error': 'Internal server error during disease prediction',
            'message': error_msg,
            'status': 'error',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/test', methods=['GET', 'POST'])
def test_endpoint():
    """Test endpoint for API validation"""
    if request.method == 'GET':
        return jsonify({
            'message': 'Test endpoint working on Vercel!',
            'method': 'GET',
            'timestamp': datetime.now().isoformat(),
            'status': 'success',
            'platform': 'Vercel Serverless',
            'test_data': {
                'sample_yield_request': {
                    'Year': 2024,
                    'State': 'Bihar',
                    'District': 'Patna',
                    'Area_1000_ha': 100,
                    'Crop': 'RICE',
                    'Avg_Rainfall_mm': 1000,
                    'Avg_Temp_C': 26.5,
                    'Soil_Type': 'Alluvial'
                }
            }
        })
    else:
        data = request.get_json() if request.is_json else {}
        return jsonify({
            'message': 'POST test successful',
            'received_data': data,
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        })

# Error handlers
@app.errorhandler(400)
def bad_request(error):
    return jsonify({
        'error': 'Bad Request',
        'message': 'Invalid request format',
        'status': 'error',
        'timestamp': datetime.now().isoformat()
    }), 400

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Not Found',
        'message': 'Endpoint not found',
        'status': 'error',
        'available_endpoints': ['/', '/health', '/predict', '/predict-disease', '/test'],
        'timestamp': datetime.now().isoformat()
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal Server Error',
        'message': 'Something went wrong on the server',
        'status': 'error',
        'timestamp': datetime.now().isoformat()
    }), 500

# Vercel serverless handler
def handler(request):
    """Vercel serverless handler"""
    return app(request.environ, request.start_response)

# For local development
if __name__ == '__main__':
    print(f"\n{'='*50}")
    print("🚀 STARTING UPAJ FLASK ML BACKEND (LOCAL)")
    print(f"{'='*50}")
    app.run(debug=True, host='0.0.0.0', port=5000)