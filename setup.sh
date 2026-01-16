#!/bin/bash

# Update conda
##echo "Updating conda..."
##conda update -n base -c defaults conda
# Create conda environment
echo "Creating conda environment SE3Suite..."
#conda create --name SE3Suite python=3.8;
# Activate conda environment
echo "Activating conda environment SE3Suite..."
#conda activate SE3Suite;
# Install PyTorch
echo "Installing PyTorch with CUDA support..."
pip install torch --index-url https://download.pytorch.org/whl/cu118;
# Install e3nn
echo "Installing e3nn..."
pip install e3nn;
# Install Biopython
echo "Installing Biopython..."
pip install biopython;
# Install pandas
echo "Installing pandas..."
pip install pandas;
# Install matplotlib
echo "Installing matplotlib..."
pip install matplotlib;
# Install tqdm
echo "Installing tqdm..."
pip install tqdm;
# Install plotly
echo "Installing plotly..."
pip install plotly;
echo "Setup completed."
