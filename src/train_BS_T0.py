import numpy as np
import torch
import random
from TorchDataLoader import get_docking_stream
from Rotations import Rotations
from PlotterT0 import PlotterT0
from TrainerWrapper import TrainerWrapperT0
from UtilityFunctions import UtilityFunctions

if __name__ == "__main__":
    #### initialization of random seeds
    random_seed = 42
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    ######################
    U = UtilityFunctions(device=None, dtype=None)

    file_path = 'configT0.txt'
    config_dict = U.read_config(file_path)
    docked_complex = bool(config_dict.get('docked_complex', False))
    zero_feature = bool(config_dict.get('zero_feature', False))
    trainset = config_dict.get('trainset', None)
    validset = config_dict.get('validset', None)
    testset = config_dict.get('testset', None)
    targetset = config_dict.get('targetset', None)
    train_max_size = int(config_dict.get('train_max_size', 10))
    test_max_size = int(config_dict.get('test_max_size', 10))
    experiment = str(config_dict.get('experiment', None))
    learning_rate = float(config_dict.get('learning_rate', 0.1))
    train_epochs = int(config_dict.get('train_epochs', 10))
    eval_freq = int(config_dict.get('eval_freq', 5))
    train_shuffle = bool(config_dict.get('train_shuffle', False))
    resolution_in_angstroms = float(config_dict.get('resolution_in_angstroms', None))
    box_dim = int(config_dict.get('box_dim', None))
    padded_dim = int(config_dict.get('padded_dim', None))

    train_model = bool(config_dict.get('train_model', False))
    evaluate_model = bool(config_dict.get('evaluate_model', False))
    resume_training = bool(config_dict.get('resume_training', False))
    resume_epoch = int(config_dict.get('resume_epoch', train_epochs))

    ######################
    print('Config file parameters set:')
    for k, v in config_dict.items():
        print(k, v)

    dtype = torch.float32
    print("dtype:", dtype)
    print(f'\n\n\n\n')
    ######################
    ## Set up data streams
    train_stream = get_docking_stream(trainset, docked_complex=docked_complex, max_size=train_max_size, shuffle=train_shuffle, clustering=False)
    valid_stream = None
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
        TRAIN = TrainerWrapperT0(
            device=device,
            dtype=dtype,
            docked_complex=docked_complex,
            zero_feature=zero_feature,
            experiment=experiment,
            learning_rate=learning_rate,
            train_model=train_model,
            train_epochs=train_epochs + 1,
            resolution_in_angstroms=resolution_in_angstroms,
            box_dim=box_dim,
            padded_dim=padded_dim,
            # so3_angles=so3_angles,
            trainer_debug=False,
            fft_debug=False,
            eval_freq=eval_freq
        )
        TRAIN.call_trainer(
            train_stream=train_stream,
            valid_stream=None,
            test_stream=test_stream)

    if evaluate_model:
        ### evaluate
        evaluate_epoch = resume_epoch
        EVALUATE = TrainerWrapperT0(
            device=device,
            dtype=dtype,
            docked_complex=docked_complex,
            zero_feature=zero_feature,
            experiment=experiment,
            evaluate_model=evaluate_model,
            resume_epoch=evaluate_epoch,
            train_epochs=1,
            resolution_in_angstroms=resolution_in_angstroms,
            box_dim=box_dim,
            padded_dim=padded_dim,
        #     so3_angles=so3_angles,
            trainer_debug=False,
            fft_debug=False,
        )
        EVALUATE.call_trainer(
            train_stream=None,
            valid_stream=valid_stream,
            test_stream=None)

    if resume_training:
        ### resume training
        RESUMER = TrainerWrapperT0(
            device=device,
            dtype=dtype,
            docked_complex=docked_complex,
            zero_feature=zero_feature,
            experiment=experiment,
            learning_rate=learning_rate,
            resume_training=resume_training,
            resume_epoch=resume_epoch,
            train_epochs=train_epochs + 1,
            eval_freq=eval_freq,
            resolution_in_angstroms=resolution_in_angstroms,
            box_dim=box_dim,
            padded_dim=padded_dim,
            # so3_angles=so3_angles,
            trainer_debug=False,
            fft_debug=False,
        )
        RESUMER.call_trainer(
            train_stream=train_stream,
            valid_stream=None,
            test_stream=test_stream)


    #### Plot loss and RMSDs from current experiment
    show = True
    plot_epoch = train_epochs
    Plotter = PlotterT0(experiment=experiment,
                        logfile_savepath='Log/losses/' + experiment + '/')
    ylim = None
    Plotter.plot_loss(ylim=ylim, show=show)
    
    plot_streams = {
        'Training': True,
        'Validation': False,
        'Testing': True
    }
    Plotter.plot_rmsd_distribution(
        plot_streams=plot_streams,
        plot_epoch=plot_epoch,
        show=show)
