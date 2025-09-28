# Upaj Flask Backend - Vercel Deployment

## 🚀 Deployed on Vercel Serverless

This Flask backend is optimized for Vercel's serverless platform.

### API Endpoints

- **GET /** - API information and available endpoints
- **GET /health** - Health check endpoint
- **POST /predict** - Crop yield prediction
- **POST /predict-disease** - Disease detection from image upload
- **GET /test** - Test endpoint

### Sample Usage

#### Yield Prediction
```bash
curl -X POST https://your-app.vercel.app/predict \
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

#### Disease Prediction
```bash
curl -X POST https://your-app.vercel.app/predict-disease \
  -F "image=@plant_image.jpg"
```

### Features

✅ **Serverless Architecture** - Auto-scaling and cost-effective
✅ **Mock Data Support** - Works without heavy ML models
✅ **CORS Enabled** - Ready for frontend integration
✅ **Error Handling** - Comprehensive error responses
✅ **Input Validation** - Validates all parameters
✅ **Fast Deployment** - Deploys in seconds

### Development

For local development:
```bash
python api/index.py
```

The app will run on http://localhost:5000