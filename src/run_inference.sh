#!/bin/bash

### Inference job for SE3Bind
### run on cgpu camden
#sbatch --cluster=amarelc
#SBATCH --partition=cgpu              # Partition (job queue)
#SBATCH --nodes=1                   # Number of nodes you require
#SBATCH --ntasks=1                   # Total # of tasks across all nodes
#SBATCH --cpus-per-task=12            # Cores per task (>1 if multithread tasks)
#SBATCH --gres=gpu:1                  # Number of GPUs
#SBATCH --mem=72G                 # Real memory (RAM) required (MB), 0 is the whole-node memory
#SBATCH --time=3-00:00:00           # Total run time limit (HH:MM:SS)

### more shared flags
##mkdir -p slurm_log/;
#SBATCH --output=slurm_log_shriya/slurm.%N.%j.%x.out
#SBATCH --error=slurm_log_shriya/slurm.%N.%j.%x.err
#SBATCH --export=ALL

mkdir -p slurm_log_shriya/

# Add the parent directory of models to the Python path
export PYTHONPATH="${PYTHONPATH}:../"
pwd

###############################################################################
# CLI Arguments - Modify these as needed
###############################################################################
# --mode              : Operation mode (required)
#                       Options: 'train', 'evaluate', 'resume'
# --config            : Path to config file
#                       Default: configT1.txt
# --experiment        : Experiment name (overrides config file)
#                       Example: my_inference_run
# --testset           : Path to test/inference dataset (overrides config file)
#                       Example: ../data/datasets/inference_dataset.pkl
# --epoch             : Epoch number for evaluation (overrides config file)
#                       Example: 1000
# --resume-epoch      : Epoch to resume training from (overrides config file)
#                       Example: 500
###############################################################################

# OPTION 1: Basic inference - uses settings from inference_config.txt
srun -N1 -n1 python train_DC_T1.py \
    --mode evaluate \
    --config inference_config.txt \
    --testset ../data/test_pdbs/test_2example_inference.pkl

# OPTION 2: Inference with epoch override
# srun -N1 -n1 python train_DC_T1.py \
#     --mode evaluate \
#     --config inference_config.txt \
#     --epoch 1000

# OPTION 3: Inference with custom testset file
# srun -N1 -n1 python train_DC_T1.py \
#     --mode evaluate \
#     --config inference_config.txt \
#     --testset ../data/datasets/my_custom_testset.pkl

# OPTION 4: Full CLI override - all arguments specified
# srun -N1 -n1 python train_DC_T1.py \
#     --mode evaluate \
#     --config inference_config.txt \
#     --experiment my_custom_inference \
#     --testset ../data/datasets/my_inference_dataset.pkl \
#     --epoch 1000

# OPTION 5: Train a new model
# srun -N1 -n1 python train_DC_T1.py \
#     --mode train \
#     --config configT1.txt

# OPTION 6: Resume training from checkpoint
# srun -N1 -n1 python train_DC_T1.py \
#     --mode resume \
#     --config configT1.txt \
#     --resume-epoch 500

# OPTION 7: Resume training with experiment override
# srun -N1 -n1 python train_DC_T1.py \
#     --mode resume \
#     --config configT1.txt \
#     --experiment my_experiment_continued \
#     --resume-epoch 500
