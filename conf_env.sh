#!/bin/bash

# Update the package list
sudo apt update

# Install required packages
sudo apt install -y nano screen python3-dev python3-venv git

# Create a Python virtual environment named "venv"
python3 -m venv venv

# Clone the SPAM-Optimizer repository
git clone https://github.com/ffuhu/SPAM-Optimizer/

# Navigate into the cloned repository directory
cd SPAM-Optimizer || exit

# Activate the virtual environment
source ../venv/bin/activate

# Install the required Python packages
pip install -r requirements_working.txt

echo "Setup complete. Virtual environment 'venv' is activated."
