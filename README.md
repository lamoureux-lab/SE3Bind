## SE3Bind: SE(3)-equivariant model for antibody-antigen binding affinity prediction

## Installation

### Requirements
- **Anaconda** (required)
- **Python 3.8**
- **CUDA-enabled GPU** (recommended for training)

### Setup Environment

#### 1. Create conda environment
```bash
conda create --name SE3Bind python=3.8
conda activate SE3Bind
```

#### 2. Install dependencies
**Option A:** Run the setup script
```bash
bash setup.sh
```

**Option B:** Install manually
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install e3nn
pip install biopython
pip install pandas
pip install matplotlib
pip install tqdm
pip install plotly
pip install mrcfile
```

---

## Project Structure

```
AntibodyDocking/
├── src/                          # Source code
│   ├── train_BS_T0.py           # Task 0 (Re-Docking) training
│   ├── train_DC_T1.py           # Task 1 (Binding affinity Joint training model) training
│   ├── TrainerT0.py             #  Re-docking model trainer class
│   ├── TrainerT1.py             # Binding affinity Joint training model trainer class
│   ├── TrainerWrapper.py        # Wrapper for both trainers
│   ├── TorchDockingFFT.py       # FFT-based docking
│   ├── ProcessCoords.py         # Coordinate processing
│   ├── Rotations.py             # Rotation handling
│   ├── UtilityFunctions.py      # Utility functions
│   ├── PlotterT0.py             # Task 0 (Re-Docking) visualization
│   ├── PlotterT1.py             # Task 1 (Binding affinity Joint training model) visualization
│   ├── configT0.txt             # Task 0 (Re-Docking) model only configuration
│   └── configT1.txt             # Task 1 (Binding affinity Joint training model)l configuration
├── models/                       # Model architectures
│   ├── model_sampling.py        # Sampling model
│   ├── model_docking.py         # Docking model
│   └── model_interaction.py     # Interaction/affinity model
├── data/                         # Dataset scripts and data
├── Figs/                         # Output visualizations
└── Log/                          # Training logs and saved models
```

---

## Usage

### Inference with SE3Bind

#### Step 1: Prepare Your Data

Create a CSV file (e.g., `mappings_example.csv`) with your PDB files and chain mappings:

```csv
filename,antibody_chains,antigen_chains
5mev.pdb,H;L,A
3iu3.pdb,A;B,K
1s78.pdb,H;L,A,B
```

- **filename**: Name of your PDB file
- **antibody_chains**: Comma-separated chain IDs for antibody (e.g., 'H,L')
- **antigen_chains**: Comma-separated chain IDs for antigen (e.g., 'A')

#### Step 2: Generate Inference Dataset

Run the dataset generation script:

```bash
cd SE3Bind/data/
python Inference_dataset_generation.py <csv_file> <pdb_directory> <output_path> <output_name>
```

**Or use the provided bash script:**
```bash
cd SE3Bind/data/
bash run_inference_dataset_gen.sh
```

**Example:**
```bash
python Inference_dataset_generation.py mappings_example.csv ./pdb_files/ ./datasets/ inference_dataset.pkl
```


**Arguments:**
- `csv_file`: Path to your CSV mapping file
- `pdb_directory`: Directory containing your PDB files
- `output_path`: Where to save the output dataset
- `output_name`: Name for the output pickle file

This will:
1. Read your CSV file
2. Split each PDB into antibody and antigen chains
3. Generate voxelized representations (75³ grid at 2Å resolution)
4. Save as a pickle file ready for inference

#### Step 3: Run Inference

Configure your inference settings in `src/inference_config.txt`:

```bash
# Key settings to modify:
testset = ../data/datasets/inference_dataset.pkl    # Your generated dataset
resume_epoch = 1000                                  # Epoch of trained model to load
```

Run inference using the command-line interface:

```bash
cd AntibodyDocking/src/
python train_DC_T1.py --mode evaluate --config inference_config.txt
```

**Command-line options:**
- `--mode`: Operation mode (`train`, `evaluate`, or `resume`)
- `--config`: Path to config file (default: `configT1.txt`)
- `--testset`: Path to dataset (overrides config file)
- `--epoch`: Epoch number for evaluation (overrides config file)

**Examples:**

```bash
# Evaluate with config file settings
python train_DC_T1.py --mode evaluate --config inference_config.txt

# Evaluate with CLI overrides
python train_DC_T1.py --mode evaluate --config inference_config.txt \
    --experiment my_custom_run --testset ../data/datasets/my_data.pkl --epoch 1000
```

The predictions will be saved in `Log/losses/<experiment_name>/` with binding affinity (ΔG) predictions.

See [data/README_Inference.md](data/README_Inference.md) for detailed documentation on the inference dataset generation process.

---

### Training Models

#### Re-Docking task
Predicts the Re-docking pose of antibody-antigen complexes.

**(not recommended without GPU):**
```bash
python train_BS_T0.py
```

Configuration: Edit `src/configT0.txt` to set model parameters.

#### Binding Affinity (FI) Prediction  
Predicts binding free energy (ΔG) from docked complexes.
```bash
cd AntibodyDocking/src/
bash run_DC_T1.sh
```

Configuration: Edit `src/configT1.txt` to set model parameters.

### Model Configuration

Key parameters in config files:
- `box_dim`: Base grid dimension (default: 50)
- `padded_dim`: Padded grid dimension (default: 100)
- `resolution_in_angstroms`: Voxel resolution (default: 2.0)
- `learning_rate`: Optimizer learning rate
- `train_epochs`: Number of training epochs
- `eval_freq`: Evaluation frequency
- `docked_complex`: Use docked complex features (True/False)
- `zero_feature`: Use zero-feature ablation (True/False)

### Monitoring Training

Training logs are saved in:
- `src/slurm_log/` (cluster jobs)
- `Log/losses/` (loss values)
- `Log/saved_models/` (model checkpoints)

---

## Inputs

### Data Format
- **Training/Testing datasets:** `.pkl` files containing:
  - Receptor (antibody) volumes
  - Ligand (antigen) volumes  
  - Ground truth rotations and translations
  - Atomic coordinates
  - Binding affinity values
  - Structure IDs and cluster information

Paths are configured in `configT0.txt` and `configT1.txt`.

---

## Outputs

### Generated Directories

#### `Figs/`
Visualization outputs:
- **`FI_correlation_plots/`** - ΔF vs ΔG correlation plots
- **`FI_loss_plots/`** - Training loss curves
- **`FI_RMSD_distribution_plots/`** - RMSD distributions
- **`Coordinate_RMSD/`** - 3D docking pose visualizations (HTML)
- **`CorrelationFFTvolumes/`** - Energy grid visualizations
- **`Feature_volumes/`** - Feature map visualizations (.html and .map files)
- **`Input_volumes/`** - Input volume visualizations

#### `Log/`
Training artifacts:
- **`losses/`** - Loss and RMSD log files (.txt)
- **`saved_models/`** - Model checkpoints (.th)

### Output File Types
- `.html` - Interactive 3D plots (open in web browser)
- `.map` - MRC density maps (open in PyMOL/ChimeraX)
- `.txt` - Training logs and metrics
- `.th` - PyTorch model checkpoints

---
<!-- Citation -->
