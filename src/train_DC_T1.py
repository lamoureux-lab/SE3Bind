import argparse
import random

import numpy as np
import torch

from PlotterT1 import PlotterT1
from Rotations import Rotations
from TorchDataLoader import get_docking_stream
from TrainerWrapper import TrainerWrapperT1
from UtilityFunctions import UtilityFunctions

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Evaluate SE3Bind binding affinity prediction model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Train a new model
  python train_DC_T1.py --mode train --config configT1.txt
  
  # Evaluate a trained model
  python train_DC_T1.py --mode evaluate --config inference_config.txt --experiment my_experiment --epoch 1000
  
  # Resume training from checkpoint
  python train_DC_T1.py --mode resume --config configT1.txt --resume-epoch 500
        ''')
    
    parser.add_argument('--mode', type=str, required=True, 
                       choices=['train', 'evaluate', 'resume'],
                       help='Mode: train (train new model), evaluate (evaluate trained model), or resume (resume training)')
    parser.add_argument('--config', type=str, default='configT1.txt',
                       help='Path to config file (default: configT1.txt)')
    parser.add_argument('--experiment', type=str, default=None,
                       help='Experiment name (overrides config file)')
    parser.add_argument('--testset', type=str, default=None,
                       help='Path to test/inference dataset (overrides config file)')
    parser.add_argument('--epoch', type=int, default=None,
                       help='Epoch number for evaluation (overrides config file)')
    parser.add_argument('--resume-epoch', type=int, default=None,
                       help='Epoch to resume training from (overrides config file)')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("SE3Bind Training Script")
    print("="*80)
    print(f"Mode: {args.mode}")
    print(f"Config file: {args.config}")
    if args.experiment:
        print(f"Experiment (CLI override): {args.experiment}")
    if args.testset:
        print(f"Test dataset (CLI override): {args.testset}")
    if args.epoch:
        print(f"Evaluation epoch (CLI override): {args.epoch}")
    if args.resume_epoch:
        print(f"Resume from epoch (CLI override): {args.resume_epoch}")
    print("="*80 + "\n")
    #### initialization of random seeds
    random_seed = 42
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    ######################
    U = UtilityFunctions(device=None, dtype=None)

    file_path = args.config
    config_dict = U.read_config(file_path)
    pretrain_experiment = config_dict.get('pretrain_experiment', None)
    pretrain_epoch_loaded = config_dict.get('pretrain_epoch_loaded', None)
    training_case = config_dict.get('training_case', None)
    interaction_learning_rate = float(config_dict.get('interaction_learning_rate', 0.1))
    JT = config_dict.get('joint_training', False)

    docked_complex = bool(config_dict.get('docked_complex', False))
    zero_feature = bool(config_dict.get('zero_feature', False))
    trainset = config_dict.get('trainset', None)
    validset = config_dict.get('validset', None)
    testset = config_dict.get('testset', None)
    targetset = config_dict.get('targetset', None)
    train_max_size = int(config_dict.get('train_max_size', 10))
    test_max_size = int(config_dict.get('test_max_size', 10))
    valid_max_size = int(config_dict.get('test_max_size', 13))
    experiment = str(config_dict.get('experiment', None))
    learning_rate = float(config_dict.get('learning_rate', 0.1))
    train_epochs = int(config_dict.get('train_epochs', 10))
    eval_freq = int(config_dict.get('eval_freq', 5))
    train_shuffle = bool(config_dict.get('train_shuffle', False))
    resolution_in_angstroms = float(config_dict.get('resolution_in_angstroms', None))
    box_dim = int(config_dict.get('box_dim', None))
    padded_dim = int(config_dict.get('padded_dim', None))
    inference = bool(config_dict.get('inference', False))

    # Override config with CLI arguments
    if args.experiment:
        experiment = args.experiment
    if args.testset:
        testset = args.testset
    if args.epoch:
        resume_epoch = args.epoch
    elif args.resume_epoch:
        resume_epoch = args.resume_epoch
    else:
        resume_epoch = int(config_dict.get('resume_epoch', train_epochs))
    
    # Set mode flags based on CLI argument (overrides config file)
    train_model = (args.mode == 'train')
    evaluate_model = (args.mode == 'evaluate')
    resume_training = (args.mode == 'resume')

    ######################
    print('\nConfig file parameters:')
    for k, v in config_dict.items():
        print(f"  {k}: {v}")
    
    print('\nActive settings after CLI overrides:')
    print(f"  Mode: {args.mode}")
    print(f"  Experiment: {experiment}")
    print(f"  Train model: {train_model}")
    print(f"  Evaluate model: {evaluate_model}")
    print(f"  Resume training: {resume_training}")
    if evaluate_model or resume_training:
        print(f"  Epoch: {resume_epoch}")
    if testset:
        print(f"  Test dataset: {testset}")

    dtype = torch.float32
    print("dtype:", dtype)
    print(f'\n\n\n\n')

    path_pretrain = 'Log/saved_models/'+ pretrain_experiment \
                    +'/'+pretrain_experiment + pretrain_epoch_loaded +'.th'
    print('pretrain to be loaded\n', path_pretrain)

    ######################
    ## Set up data streams conditionally based on mode
    train_stream = None
    valid_stream = None
    test_stream = None
    
    if train_model or resume_training:
        # Training and resuming need train and valid streams
        print('Loading training and validation datasets...')
        train_stream = get_docking_stream(trainset, docked_complex=docked_complex, max_size=train_max_size, shuffle=train_shuffle, clustering=False)
        valid_stream = get_docking_stream(validset, docked_complex=docked_complex, max_size=valid_max_size, shuffle=train_shuffle, clustering=False)
    
    if evaluate_model:
        # Evaluation only needs test stream
        print('Loading test/inference dataset...')
        test_stream = get_docking_stream(testset, docked_complex=docked_complex, max_size=test_max_size, shuffle=False, clustering=False)

    if targetset:
        target_stream = get_docking_stream(targetset, max_size=None, shuffle=False, clustering=False)
        print('Target stream loaded')

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    ################################################################
    # so3_angles = Rotations(angle_inc=20, device=device, dtype=dtype).rotation_mats
    ################################################################
    ### Initialize trainers for specified tasks

    if train_model:
        ### train
        TRAIN = TrainerWrapperT1(
            device=device,
            dtype=dtype,
            docked_complex=docked_complex,
            zero_feature=zero_feature,
            experiment=experiment,
            learning_rate=learning_rate,
            interaction_learning_rate=interaction_learning_rate,
            path_pretrain=path_pretrain,
            JT=JT,
            training_case=training_case,
            train_model=train_model,
            train_epochs=train_epochs + 1,
            resolution_in_angstroms=resolution_in_angstroms,
            box_dim=box_dim,
            padded_dim=padded_dim,
            trainer_debug=False,
            fft_debug=False,
            eval_freq=eval_freq,
            inference=inference,
        )
        TRAIN.call_trainer(
            train_stream=train_stream,
            valid_stream=valid_stream,
            test_stream=None)

    if evaluate_model:
        ### evaluate
        evaluate_epoch = resume_epoch
        EVALUATE = TrainerWrapperT1(
            device=device,
            dtype=dtype,
            docked_complex=docked_complex,
            zero_feature=zero_feature,
            experiment=experiment,
            path_pretrain=path_pretrain,
            learning_rate=learning_rate,
            interaction_learning_rate=interaction_learning_rate,
            JT=JT,
            training_case=training_case,
            evaluate_model=evaluate_model,
            resume_epoch=evaluate_epoch,
            train_epochs=1,
            resolution_in_angstroms=resolution_in_angstroms,
            box_dim=box_dim,
            padded_dim=padded_dim,
            trainer_debug=False,
            fft_debug=False,
            inference=inference,
        )
        EVALUATE.call_trainer(
            train_stream=None,
            valid_stream=None,
            test_stream=test_stream)

    if resume_training:
        ### resume training
        RESUMER = TrainerWrapperT1(
            device=device,
            dtype=dtype,
            docked_complex=docked_complex,
            zero_feature=zero_feature,
            experiment=experiment,
            learning_rate=learning_rate,
            interaction_learning_rate=interaction_learning_rate,
            path_pretrain=path_pretrain,
            JT=JT,
            training_case=training_case,
            resume_training=resume_training,
            resume_epoch=resume_epoch,
            train_epochs=train_epochs + 1,
            eval_freq=eval_freq,
            resolution_in_angstroms=resolution_in_angstroms,
            box_dim=box_dim,
            padded_dim=padded_dim,
            trainer_debug=False,
            fft_debug=False,
            inference=inference,
        )
        RESUMER.call_trainer(
            train_stream=train_stream,
            valid_stream=valid_stream,
            test_stream=None)


    #### Plot loss and RMSDs from current experiment
    show = True
    plot_epoch = train_epochs
    Plotter = PlotterT1(experiment=experiment,
                        logfile_savepath='Log/losses/' + experiment + '/')
    ylim = None

    Plotter.plot_loss(ylim=ylim, show=show)

    plot_streams = {
        'Training': False,
        'Validation': False,
        'Testing': True
    }

    Plotter.plot_rmsd_distribution(
        plot_streams=plot_streams,
        plot_epoch=plot_epoch,
        show=show)

    ## plot deltaG correlation with for inference test dataset 
    Plotter.plot_inference_deltaG_correlation(
            plot_training=True,
            plot_valid=False,
            plot_testing=True,
            plot_all_combined=False,
            plot_crystal=False,
            plot_homology=False, 
            plot_epoch=plot_epoch,
            show=False,
            save=True,
            valid_crystal=False,
            valid_homology=False)