# Test Configuration for Upaj Flask Backend

## API Endpoints

### 1. Health Check
```bash
curl -X GET https://your-app.onrender.com/health
```

### 2. Root Information
```bash
curl -X GET https://your-app.onrender.com/
```

### 3. Test Endpoint
```bash
curl -X GET https://your-app.onrender.com/test
```

### 4. Yield Prediction
```bash
curl -X POST https://your-app.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Year": 2024,
    "State": "Bihar",
    "District": "Patna",
    "Area_1000_ha": 100,
    "Crop": "RICE",
    "Avg_Rainfall_mm": 1000,
    "Avg_Temp_C": 26.5,
    "Soil_Type": "Alluvial"
  }'
```

### 5. Disease Prediction (with image upload)
```bash
curl -X POST https://your-app.onrender.com/predict-disease \
  -F "image=@/path/to/your/plant_image.jpg"
```

## Expected Responses

All endpoints return JSON with proper error handling and fallback modes.

## Features

✅ **Graceful Fallbacks**: Works with or without ML models
✅ **Error Handling**: Comprehensive error responses  
✅ **Input Validation**: Validates all input parameters
✅ **Mock Data**: Provides realistic mock predictions when models unavailable
✅ **Production Ready**: Proper logging, timeouts, and workers
✅ **Cross-Origin**: CORS enabled for frontend integration