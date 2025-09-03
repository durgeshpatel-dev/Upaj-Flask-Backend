from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib 

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024  # 1 MB limit
CORS(app)

model = joblib.load('rf_model.joblib')
RMSE = 351.1136  # From Random Forest model
required_columns = ['Year', 'State', 'District', 'Area_1000_ha', 'Crop', 'Avg_Rainfall_mm', 'Avg_Temp_C', 'Soil_Type']
valid_soil_types = ['Sandy', 'Alluvial', 'Black', 'Red-Yellow', 'Red', 'Loamy']
valid_crops = ['RICE', 'GROUNDNUT', 'WHEAT', 'MAIZE', 'SUGARCANE']
valid_states = ['Chhattisgarh', 'Madhya Pradesh', 'West Bengal', 'Bihar', 'Jharkhand', 'Orissa', 'Gujarat']
predictable_crops = ['RICE', 'GROUNDNUT']  # Crops model is trained on

# Optimal ranges for crops (dataset for RICE/GROUNDNUT, literature for others)
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

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        print(f"Incoming data: {data}")
        print(f"Data types: { {k: type(v).__name__ for k, v in data.items()} }")  # Log types of each field
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
        input_df = pd.DataFrame([data])
        input_df['Year'] = input_df['Year'].astype(int)
        input_df['Area_1000_ha'] = input_df['Area_1000_ha'].astype(float)
        input_df['Avg_Rainfall_mm'] = input_df['Avg_Rainfall_mm'].astype(float)
        input_df['Avg_Temp_C'] = input_df['Avg_Temp_C'].astype(float)
        
        # Generate recommendations
        recommendations = get_recommendations(input_df)
        
        # Predict if crop is supported, else return recommendations only
        if data['Crop'] in predictable_crops:
            pred = model.predict(input_df)[0]
            ci_lower = round(pred - 1.96 * RMSE, 2)
            ci_upper = round(pred + 1.96 * RMSE, 2)
            response = {
                'yield_kg_ha': round(pred, 2),
                'confidence_interval_95': [ci_lower, ci_upper],
                'recommendations': recommendations
            }
            print(f"Response: {response}")
            return jsonify(response)
        else:
            response = {
                'yield_kg_ha': None,
                'confidence_interval_95': None,
                'recommendations': recommendations,
                'message': f"Predictions unavailable for {data['Crop']}. Model trained on {predictable_crops}."
            }
            print(f"Response: {response}")
            return jsonify(response)

    except Exception as e:
        print(f"Exception: {str(e)}")
        response = {'error': str(e)}
        print(f"Response: {response}")
        return jsonify(response), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)