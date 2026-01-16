#!/bin/bash

# Script to generate inference dataset from PDB files
# This script should be run from the AntibodyDocking/data/ directory

### run on cgpu camden
#sbatch --cluster=amarelc
#SBATCH --partition=cgpu              # Partition (job queue)
#SBATCH --nodes=1                   # Number of nodes you require
#SBATCH --ntasks=1                   # Total # of tasks across all nodes
#SBATCH --cpus-per-task=12            # Cores per task (>1 if multithread tasks)
#SBATCH --gres=gpu:1                  # Number of GPUs
#SBATCH --mem=72G                 # Real memory (RAM) required (MB), 0 is the whole-node memory
#SBATCH --time=3-00:00:00           # Total run time limit (HH:MM:SS)

mkdir -p slurm_log_shriya/
# Add the parent directory of models to the Python path
export PYTHONPATH="${PYTHONPATH}:../"
pwd

# Configuration - modify these paths as needed
# Use relative paths (script should be run from AntibodyDocking/data/ directory)
CSV_FILE="test_pdbs/mappings_example.csv"
PDB_DIR="test_pdbs/"
OUTPUT_DIR="test_pdbs/"
OUTPUT_NAME="test_2example_inference.pkl"


echo "=========================================="
echo "SE3Bind Inference Dataset Generation"
echo "=========================================="
echo "CSV file: $CSV_FILE"
echo "PDB directory: $PDB_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Output filename: $OUTPUT_NAME"
echo "=========================================="
echo ""

# Check if files/directories exist
if [ ! -f "$CSV_FILE" ]; then
    echo "Error: CSV file not found: $CSV_FILE"
    exit 1
fi

if [ ! -d "$PDB_DIR" ]; then
    echo "Error: PDB directory not found: $PDB_DIR"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run the dataset generation script
python Inference_dataset_generation.py "$CSV_FILE" "$PDB_DIR" "$OUTPUT_DIR" "$OUTPUT_NAME"

echo ""
echo "=========================================="
echo "Dataset generation complete!"
echo "Output saved to: ${OUTPUT_DIR}${OUTPUT_NAME}"
echo "=========================================="
