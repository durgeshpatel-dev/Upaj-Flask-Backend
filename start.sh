#!/bin/bash
echo "🚀 Starting Flask ML Backend..."
echo "📁 Current directory: $(pwd)"
echo "📂 Files available:"
ls -la
echo "🐍 Python version:"
python --version
echo "📦 Installing dependencies..."
pip install -r requirements.txt
echo "🔥 Starting application..."
python app.py