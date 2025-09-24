#!/bin/bash
# Setup script for the development environment

set -e

echo "Setting up development environment..."

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip -q

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt -q

echo "Setup complete!"
echo ""
echo "To run Python scripts: ./shell python script.py"
echo "To run tests: ./test"
echo "To activate venv manually: source venv/bin/activate"
