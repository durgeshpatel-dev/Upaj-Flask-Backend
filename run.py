#!/usr/bin/env python3
"""
Production startup script for Upaj Flask Backend
Handles graceful fallbacks and proper error handling
"""

import os
import sys

def main():
    print("🚀 Starting Upaj Flask ML Backend...")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    # Set environment variables
    os.environ.setdefault('FLASK_ENV', 'production')
    os.environ.setdefault('FLASK_APP', 'app_production.py')
    
    # Import and run the Flask app
    try:
        from app_production import app
        port = int(os.environ.get('PORT', 5000))
        app.run(host='0.0.0.0', port=port, debug=False)
    except Exception as e:
        print(f"❌ Failed to start application: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()