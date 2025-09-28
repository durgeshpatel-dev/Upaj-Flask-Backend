from flask import Flask, request, jsonify, send_file
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

def init_models():
    """Initialize ML models with proper error handling"""
    global model_status
    
    print("🔧 Initializing models...")
    
    # Try to load yield prediction model
    try:
        import joblib
        model = joblib.load('rf_model.joblib')
        model_status['yield_model'] = True
        print("✅ Yield prediction model loaded successfully")
    except Exception as e:
        print(f"⚠️  Yield model not available (using mock data): {str(e)}")
        model_status['yield_model'] = False
    
    # Try to load disease prediction model
    try:
        import tensorflow as tf
        disease_model = tf.keras.models.load_model('india_crop_disease_model.h5')
        model_status['disease_model'] = True
        print("✅ Disease prediction model loaded successfully")
    except Exception as e:
        print(f"⚠️  Disease model not available (using mock data): {str(e)}")
        model_status['disease_model'] = False

@app.route('/', methods=['GET'])
def root():
    """Root endpoint with API information"""
    return jsonify({
        'message': 'Upaj Flask ML Backend API',
        'version': '1.0.0',
        'status': 'running',
        'startup_time': model_status['startup_time'],
        'models': {
            'yield_prediction': model_status['yield_model'],
            'disease_detection': model_status['disease_model']
        },
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
        'models': {
            'yield_model': 'loaded' if model_status['yield_model'] else 'mock_mode',
            'disease_model': 'loaded' if model_status['disease_model'] else 'mock_mode'
        },
        'python_version': f"{os.sys.version}",
        'environment': os.environ.get('ENVIRONMENT', 'production')
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
        
        # Get prediction (real or mock)
        if model_status['yield_model']:
            try:
                import joblib
                import pandas as pd
                
                model = joblib.load('rf_model.joblib')
                df = pd.DataFrame([data])
                prediction = model.predict(df)[0]
                confidence = 0.85  # Real model confidence would be calculated
                
                result = {
                    'prediction': float(prediction),
                    'confidence': confidence,
                    'model_type': 'random_forest',
                    'status': 'success'
                }
            except Exception as e:
                print(f"Model prediction failed: {str(e)}")
                # Fallback to mock
                mock_data = mock_predictions.get(data['Crop'], {'yield': 3000, 'confidence': 0.80})
                result = {
                    'prediction': mock_data['yield'],
                    'confidence': mock_data['confidence'],
                    'model_type': 'fallback_mock',
                    'status': 'success'
                }
        else:
            # Use mock data
            mock_data = mock_predictions.get(data['Crop'], {'yield': 3000, 'confidence': 0.80})
            result = {
                'prediction': mock_data['yield'],
                'confidence': mock_data['confidence'],
                'model_type': 'mock',
                'status': 'success'
            }
        
        result.update({
            'input_data': data,
            'timestamp': datetime.now().isoformat(),
            'units': 'kg/hectare'
        })
        
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
        
        # Get prediction (real or mock)
        if model_status['disease_model']:
            try:
                import tensorflow as tf
                from PIL import Image
                import numpy as np
                
                # Load and preprocess image
                img = Image.open(file).convert('RGB')
                img = img.resize((224, 224))  # Standard size for most models
                img_array = np.array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                # Load model and predict
                disease_model = tf.keras.models.load_model('india_crop_disease_model.h5')
                predictions = disease_model.predict(img_array)
                predicted_class_idx = np.argmax(predictions[0])
                confidence = float(predictions[0][predicted_class_idx])
                
                predicted_class = disease_classes[predicted_class_idx] if predicted_class_idx < len(disease_classes) else 'Unknown'
                
                result = {
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'model_type': 'cnn',
                    'status': 'success'
                }
            except Exception as e:
                print(f"Model prediction failed: {str(e)}")
                # Fallback to mock
                import random
                predicted_class = random.choice(disease_classes)
                result = {
                    'predicted_class': predicted_class,
                    'confidence': round(random.uniform(0.7, 0.95), 2),
                    'model_type': 'fallback_mock',
                    'status': 'success'
                }
        else:
            # Use mock data
            import random
            predicted_class = random.choice(disease_classes)
            result = {
                'predicted_class': predicted_class,
                'confidence': round(random.uniform(0.7, 0.95), 2),
                'model_type': 'mock',
                'status': 'success'
            }
        
        result.update({
            'filename': file.filename,
            'timestamp': datetime.now().isoformat(),
            'all_classes': disease_classes
        })
        
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
            'message': 'Test endpoint working!',
            'method': 'GET',
            'timestamp': datetime.now().isoformat(),
            'status': 'success',
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

# Initialize models on startup
init_models()

if __name__ == '__main__':
    print(f"\n{'='*50}")
    print("🚀 STARTING UPAJ FLASK ML BACKEND")
    print(f"{'='*50}")
    print(f"Models loaded:")
    print(f"  Yield Prediction: {'✅' if model_status['yield_model'] else '⚠️  (mock mode)'}")
    print(f"  Disease Detection: {'✅' if model_status['disease_model'] else '⚠️  (mock mode)'}")
    print(f"{'='*50}")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)