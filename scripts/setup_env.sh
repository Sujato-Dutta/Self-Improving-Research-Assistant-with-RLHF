#!/bin/bash
# Setup script for Linux / HPC environment
echo "========================================================"
echo "Setting up Self-Improving Research Assistant Environment"
echo "========================================================"

if [ ! -d "venv" ]; then
    echo "Creating virtual environment 'venv'..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing unpinned dependencies via pip..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Environment setup complete!"
