#!/bin/bash
# GraphGuard Single-Command Startup Script

echo "Starting GraphGuard Deployment Pipeline..."

# 1. Install pinned dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# 2. Run Smoke Test
echo "Executing Smoke Test..."
python scripts/smoke_test.py
if [ $? -ne 0 ]; then
    echo "Smoke test failed. Aborting startup."
    exit 1
fi

# 3. Launch the Application
echo "Launching Streamlit App..."
streamlit run app/main.py