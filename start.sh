#!/bin/bash
echo "� Setting up Python environment..."
python --version

echo "📦 Installing dependencies with pip cache..."
pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt

echo "� Starting Flask application..."
exec gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120