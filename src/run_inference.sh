#!/bin/bash

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
