from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    response = {
        'status': 'healthy',
        'message': 'Flask API is running!',
        'python_version': '3.13.4',
        'available_endpoints': ['/health', '/api/test']
    }
    print(f"Health check response: {response}")
    return jsonify(response)

@app.route('/api/test', methods=['GET', 'POST'])
def test_endpoint():
    """Test endpoint for basic functionality"""
    if request.method == 'GET':
        return jsonify({
            'message': 'API is working!',
            'method': 'GET',
            'status': 'success'
        })
    else:
        data = request.get_json() if request.is_json else {}
        return jsonify({
            'message': 'POST request received',
            'received_data': data,
            'status': 'success'
        })

@app.route('/', methods=['GET'])
def root():
    """Root endpoint"""
    return jsonify({
        'message': 'Upaj Flask Backend API',
        'version': '1.0.0',
        'status': 'running',
        'endpoints': {
            'health': '/health',
            'test': '/api/test'
        }
    })

if __name__ == '__main__':
    print(f"\n{'='*50}")
    print("🚀 STARTING FLASK APPLICATION (BASIC VERSION)")
    print(f"{'='*50}")
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)