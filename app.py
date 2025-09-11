from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib 
import tensorflow as tf
from PIL import Image, ImageDraw
import io
import os
import cloudinary
import cloudinary.uploader
from datetime import datetime

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024  # 1 MB limit
CORS(app)

# Existing yield prediction model
RMSE = 351.1136
required_columns = ['Year', 'State', 'District', 'Area_1000_ha', 'Crop', 'Avg_Rainfall_mm', 'Avg_Temp_C', 'Soil_Type']
valid_soil_types = ['Sandy', 'Alluvial', 'Black', 'Red-Yellow', 'Red', 'Loamy']
valid_crops = ['RICE', 'GROUNDNUT', 'WHEAT', 'MAIZE', 'SUGARCANE']
valid_states = ['Chhattisgarh', 'Madhya Pradesh', 'West Bengal', 'Bihar', 'Jharkhand', 'Orissa', 'Gujarat']
predictable_crops = ['RICE', 'GROUNDNUT']

try:
    model = joblib.load('rf_model.joblib')
    print("✅ Yield prediction model loaded successfully")
    print(f"   Model type: {type(model).__name__}")
    print(f"   RMSE: {RMSE}")
except Exception as e:
    print(f"❌ Error loading yield model: {e}")
    model = None

print(f"📋 Valid configurations:")
print(f"   Valid crops: {valid_crops}")
print(f"   Predictable crops: {predictable_crops}")
print(f"   Valid states: {valid_states}")
print(f"   Valid soil types: {valid_soil_types}")
print(f"   Required columns: {required_columns}")

# Disease prediction model and paths
MODEL_PATH = 'india_crop_disease_model.h5'  # Replace with your actual model file name
TRAIN_DIR = 'dataset/train'
LOCAL_UPLOADS_DIR = 'uploads'
LOCAL_PREDICTIONS_DIR = 'predictions'

print(f"📁 Upload directories: uploads={LOCAL_UPLOADS_DIR}, predictions={LOCAL_PREDICTIONS_DIR}")
print(f"☁️  Cloudinary configured: {'Yes' if cloudinary.config().cloud_name != 'YOUR_CLOUDINARY_CLOUD_NAME' else 'No (using placeholder)'}")
print(f"🚀 Starting Flask app on http://0.0.0.0:5000")
cloudinary.config(
    cloud_name='dadlkmjr2',
    api_key='865267912265755',
    api_secret='nogalXpY9Z3XQuA91ytVhEw-LsY'
)

# Create local directories
os.makedirs(LOCAL_UPLOADS_DIR, exist_ok=True)
os.makedirs(LOCAL_PREDICTIONS_DIR, exist_ok=True)

# Load disease prediction model
try:
    disease_model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Disease prediction model loaded successfully")
    print(f"   Model path: {MODEL_PATH}")
    print(f"   Model summary: {disease_model.summary()}")
except Exception as e:
    print(f"❌ Error loading disease model: {e}")
    disease_model = None

# Get class names for disease prediction
if os.path.exists(TRAIN_DIR):
    class_names = sorted(os.listdir(TRAIN_DIR))
else:
    class_names = [
        'Corn___Common_rust', 'Corn___Healthy', 'Corn___Northern_Leaf_Blight',
        'Cotton___diseased_cotton_leaf', 'Cotton___fresh_cotton_leaf',
        'Rice___Bacterial_leaf_blight', 'Rice___Brown_spot', 'Rice___Healthy',
        'Wheat___Healthy', 'Wheat___Leaf_Rust'
    ]

optimal_ranges = {
    'RICE': {
        'Avg_Rainfall_mm': (800, 1200, 'Adjust irrigation to 800–1200 mm for optimal rice yield.'),
        'Avg_Temp_C': (26, 27.5, 'Plant in cooler periods (26–27.5°C) to avoid heat stress.'),
        'Area_1000_ha': (50, 200, 'Scale up area (>50,000 ha) for stable yields.'),
        'Soil_Type': (['Alluvial', 'Black'], 'Use Alluvial/Black soil or add organic matter for fertility.'),
        'NPK': ((80, 120, 40, 60, 40, 50), 'Apply N: 80–120 kg/ha, P: 40–60 kg/ha, K: 40–50 kg/ha.')
    },
    'GROUNDNUT': {
        'Avg_Rainfall_mm': (650, 800, 'Maintain 650–800 mm rainfall to avoid waterlogging.'),
        'Avg_Temp_C': (27, 28, 'Use shade nets or plant earlier to maintain 27–28°C.'),
        'Area_1000_ha': (100, 270, 'Scale up area (>100,000 ha) for stable yields.'),
        'Soil_Type': (['Sandy', 'Red'], 'Use Sandy/Red soil or add organic matter for drainage.'),
        'NPK': ((20, 40, 40, 60, 20, 30), 'Apply N: 20–40 kg/ha, P: 40–60 kg/ha, K: 20–30 kg/ha.')
    },
    'WHEAT': {
        'Avg_Rainfall_mm': (400, 750, 'Ensure 400–750 mm rainfall or irrigation for wheat.'),
        'Avg_Temp_C': (15, 25, 'Plant in cool season (15–25°C) to optimize growth.'),
        'Area_1000_ha': (50, 200, 'Scale up area (>50,000 ha) for stable yields.'),
        'Soil_Type': (['Loamy', 'Alluvial', 'Black'], 'Use Loamy/Alluvial/Black soil or add organic matter.'),
        'NPK': ((100, 120, 50, 60, 40, 50), 'Apply N: 100–120 kg/ha, P: 50–60 kg/ha, K: 40–50 kg/ha.')
    },
    'MAIZE': {
        'Avg_Rainfall_mm': (500, 800, 'Maintain 500–800 mm rainfall for maize.'),
        'Avg_Temp_C': (21, 30, 'Plant in 21–30°C for optimal growth.'),
        'Area_1000_ha': (20, 200, 'Scale up area (>20,000 ha) for stable yields.'),
        'Soil_Type': (['Sandy', 'Loamy', 'Alluvial'], 'Use Sandy/Loamy/Alluvial soil or add organic matter.'),
        'NPK': ((120, 150, 60, 80, 40, 60), 'Apply N: 120–150 kg/ha, P: 60–80 kg/ha, K: 40–60 kg/ha.')
    },
    'SUGARCANE': {
        'Avg_Rainfall_mm': (1000, 1500, 'Ensure 1000–1500 mm rainfall or irrigation for sugarcane.'),
        'Avg_Temp_C': (20, 35, 'Maintain 20–35°C for optimal growth.'),
        'Area_1000_ha': (10, 100, 'Scale up area (>10,000 ha) for stable yields.'),
        'Soil_Type': (['Alluvial', 'Red-Yellow', 'Black'], 'Use Alluvial/Red-Yellow/Black soil or add organic matter.'),
        'NPK': ((150, 200, 60, 80, 60, 80), 'Apply N: 150–200 kg/ha, P: 60–80 kg/ha, K: 60–80 kg/ha.')
    }
}

def get_recommendations(input_data):
    crop = input_data['Crop'][0]
    if crop not in optimal_ranges:
        return ["Invalid crop. No recommendations available."]
    recs = []
    for feature, value in optimal_ranges[crop].items():
        if feature == 'Soil_Type':
            valid_list, advice = value
            if input_data[feature][0] not in valid_list:
                recs.append(f"Soil_Type: Current={input_data[feature][0]}. {advice}")
        elif feature == 'NPK':
            npk_vals, advice = value
            recs.append(f"Fertilizer: {advice}")
        else:
            min_val, max_val, advice = value
            val = input_data[feature][0]
            if val < min_val or val > max_val:
                recs.append(f"{feature}: Current={val}. {advice}")
    if not recs:
        recs.append(f"All inputs are optimal for {crop.lower()} yield.")
    return recs

# Disease prediction helper functions
def preprocess_image(image):
    """Preprocess image for disease prediction model"""
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def save_to_cloudinary(file, folder):
    """Save image to Cloudinary and return URL"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    public_id = f"{folder}/{timestamp}_{file.filename}"
    response = cloudinary.uploader.upload(
        file,
        folder=folder,
        resource_type='image',
        public_id=public_id
    )
    return response['secure_url']

@app.route('/predict', methods=['POST'])
def predict():
    """Yield prediction endpoint (existing functionality)"""
    print(f"\n=== YIELD PREDICTION REQUEST ===")
    print(f"Request Method: {request.method}")
    print(f"Request Headers: {dict(request.headers)}")
    print(f"Content-Type: {request.content_type}")

    try:
        data = request.json
        print(f"Incoming JSON data: {data}")
        print(f"Data types: { {k: type(v).__name__ for k, v in data.items()} }")

        # Validate input
        if not all(col in data for col in required_columns):
            missing = [col for col in required_columns if col not in data]
            print(f"Missing fields: {missing}")
            response = {'error': 'Missing required fields'}
            print(f"Response: {response}")
            return jsonify(response), 400

        if data['Soil_Type'] not in valid_soil_types:
            print(f"Invalid Soil_Type: {data['Soil_Type']}, valid: {valid_soil_types}")
            response = {'error': f"Invalid Soil_Type. Use: {valid_soil_types}"}
            print(f"Response: {response}")
            return jsonify(response), 400

        if data['Crop'] not in valid_crops:
            print(f"Invalid Crop: {data['Crop']}, valid: {valid_crops}")
            response = {'error': f"Invalid Crop. Use: {valid_crops}"}
            print(f"Response: {response}")
            return jsonify(response), 400

        if data['State'] not in valid_states:
            print(f"Invalid State: {data['State']}, valid: {valid_states}")
            response = {'error': f"Invalid State. Use: {valid_states}"}
            print(f"Response: {response}")
            return jsonify(response), 400

        # Create input DataFrame
        print(f"Creating DataFrame with data: {data}")
        input_df = pd.DataFrame([data])
        print(f"Original DataFrame:\n{input_df}")

        input_df['Year'] = input_df['Year'].astype(int)
        input_df['Area_1000_ha'] = input_df['Area_1000_ha'].astype(float)
        input_df['Avg_Rainfall_mm'] = input_df['Avg_Rainfall_mm'].astype(float)
        input_df['Avg_Temp_C'] = input_df['Avg_Temp_C'].astype(float)
        print(f"Converted DataFrame:\n{input_df}")

        # Generate recommendations
        recommendations = get_recommendations(input_df)
        print(f"Generated recommendations: {recommendations}")

        # Predict if crop is supported, else return recommendations only
        if data['Crop'] in predictable_crops:
            print(f"Predicting for crop: {data['Crop']}")
            pred = model.predict(input_df)[0]
            print(f"Raw prediction: {pred}")
            ci_lower = round(pred - 1.96 * RMSE, 2)
            ci_upper = round(pred + 1.96 * RMSE, 2)
            print(f"Confidence interval: [{ci_lower}, {ci_upper}]")

            response = {
                'yield_kg_ha': round(pred, 2),
                'confidence_interval_95': [ci_lower, ci_upper],
                'recommendations': recommendations
            }
            print(f"Final response: {response}")
            return jsonify(response)
        else:
            print(f"Crop {data['Crop']} not in predictable crops: {predictable_crops}")
            response = {
                'yield_kg_ha': None,
                'confidence_interval_95': None,
                'recommendations': recommendations,
                'message': f"Predictions unavailable for {data['Crop']}. Model trained on {predictable_crops}."
            }
            print(f"Final response: {response}")
            return jsonify(response)

    except Exception as e:
        print(f"Exception in yield prediction: {str(e)}")
        print(f"Exception type: {type(e).__name__}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        response = {'error': str(e)}
        print(f"Error response: {response}")
        return jsonify(response), 400

@app.route('/predict-disease', methods=['POST'])
def predict_disease():
    """Disease prediction endpoint (new functionality)"""
    print(f"\n=== DISEASE PREDICTION REQUEST ===")
    print(f"Request Method: {request.method}")
    print(f"Request Headers: {dict(request.headers)}")
    print(f"Content-Type: {request.content_type}")
    print(f"Files in request: {list(request.files.keys())}")

    if disease_model is None:
        print("ERROR: Disease model not loaded")
        return jsonify({'error': 'Disease prediction model not available'}), 500

    if 'image' not in request.files:
        print("ERROR: No image file in request")
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    print(f"Uploaded file: {file.filename}")
    print(f"File content type: {file.content_type}")
    print(f"File size: {len(file.read())} bytes")
    file.seek(0)  # Reset file pointer

    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        print(f"ERROR: Invalid file format: {file.filename}")
        return jsonify({'error': 'Invalid file format. Use JPG, JPEG, or PNG'}), 400

    try:
        print("Starting image processing...")

        # Save uploaded image to Cloudinary
        file.seek(0)
        try:
            print("Attempting to save to Cloudinary...")
            uploaded_url = save_to_cloudinary(file, 'uploads')
            print(f"Successfully saved to Cloudinary: {uploaded_url}")
        except Exception as e:
            print(f"Cloudinary upload failed: {str(e)}, falling back to local storage")
            # Fallback to local storage
            file.seek(0)
            uploaded_path = os.path.join(LOCAL_UPLOADS_DIR, file.filename)
            file.save(uploaded_path)
            uploaded_url = f"/uploads/{file.filename}"
            print(f"Saved locally: {uploaded_path}")

        # Load and preprocess image
        print("Loading and preprocessing image...")
        file.seek(0)
        img = Image.open(file).convert('RGB')
        print(f"Image loaded: size={img.size}, mode={img.mode}")

        img_array = preprocess_image(img)
        print(f"Image preprocessed: shape={img_array.shape}, dtype={img_array.dtype}")

        # Predict
        print("Running model prediction...")
        predictions = disease_model.predict(img_array)
        print(f"Raw predictions: {predictions}")
        print(f"Predictions shape: {predictions.shape}")

        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = class_names[predicted_class_idx]
        confidence = float(predictions[0][predicted_class_idx])

        print(f"Predicted class index: {predicted_class_idx}")
        print(f"Predicted class: {predicted_class}")
        print(f"Confidence: {confidence}")

        # Parse crop and status
        try:
            crop, status = predicted_class.split('___')
            status = status.replace('_', ' ')
            print(f"Parsed crop: {crop}, status: {status}")
        except Exception as e:
            print(f"Failed to parse class name '{predicted_class}': {str(e)}")
            crop, status = 'Unknown', predicted_class

        # Annotate image
        print("Annotating image...")
        draw = ImageDraw.Draw(img)
        annotation_text = f"{crop}: {status} ({confidence:.2%})"
        draw.text((10, 10), annotation_text, fill=(255, 0, 0))
        print(f"Annotation added: {annotation_text}")

        # Save annotated image to in-memory buffer
        output_buffer = io.BytesIO()
        img.save(output_buffer, format='JPEG')
        output_buffer.seek(0)
        print(f"Annotated image saved to buffer: {len(output_buffer.getvalue())} bytes")

        # Save annotated image to Cloudinary
        try:
            print("Attempting to save annotated image to Cloudinary...")
            predicted_url = save_to_cloudinary(output_buffer, 'predictions')
            print(f"Annotated image saved to Cloudinary: {predicted_url}")
        except Exception as e:
            print(f"Cloudinary save failed: {str(e)}, falling back to local storage")
            # Fallback to local storage
            predicted_filename = f"pred_{file.filename}"
            predicted_path = os.path.join(LOCAL_PREDICTIONS_DIR, predicted_filename)
            img.save(predicted_path)
            predicted_url = f"/predictions/{predicted_filename}"
            print(f"Annotated image saved locally: {predicted_path}")

        # Return JSON response
        response = {
            'crop': crop,
            'status': status,
            'confidence': confidence,
            'uploaded_image_url': uploaded_url,
            'predicted_image_url': predicted_url
        }
        print(f"Final response: {response}")
        return jsonify(response)

    except Exception as e:
        print(f"Exception in disease prediction: {str(e)}")
        print(f"Exception type: {type(e).__name__}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        response = {'error': str(e)}
        print(f"Error response: {response}")
        return jsonify(response), 400

# Optional: Serve local images (for fallback)
@app.route('/uploads/<filename>')
def serve_uploaded(filename):
    """Serve uploaded images from local storage"""
    try:
        return send_file(os.path.join(LOCAL_UPLOADS_DIR, filename))
    except FileNotFoundError:
        return jsonify({'error': 'File not found'}), 404

@app.route('/predictions/<filename>')
def serve_predicted(filename):
    """Serve predicted images from local storage"""
    try:
        return send_file(os.path.join(LOCAL_PREDICTIONS_DIR, filename))
    except FileNotFoundError:
        return jsonify({'error': 'File not found'}), 404

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    print(f"\n=== HEALTH CHECK REQUEST ===")
    print(f"Request Method: {request.method}")
    print(f"Request Headers: {dict(request.headers)}")

    response = {
        'status': 'healthy',
        'yield_model': 'loaded',
        'disease_model': 'loaded' if disease_model else 'not available'
    }
    print(f"Health check response: {response}")
    return jsonify(response)

@app.route('/test')
def test_page():
    """Serve the test HTML page"""
    print(f"\n=== TEST PAGE REQUEST ===")
    print(f"Request Method: {request.method}")
    print(f"Request Headers: {dict(request.headers)}")
    print("Serving test.html template")
    return render_template('test.html')

if __name__ == '__main__':
    print(f"\n{'='*50}")
    print("🚀 STARTING FLASK APPLICATION")
    print(f"{'='*50}")
    app.run(debug=True, host='0.0.0.0', port=5000)