from PlotterT1 import PlotterT1
from UtilityFunctions import UtilityFunctions

if __name__ == "__main__":

    ######################
    U = UtilityFunctions(device=None, dtype=None)

    file_path = 'configT1.txt'
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

    train_model = bool(config_dict.get('train_model', False))
    evaluate_model = bool(config_dict.get('evaluate_model', False))
    resume_training = bool(config_dict.get('resume_training', False))
    resume_epoch = int(config_dict.get('resume_epoch', train_epochs))

    ######################
    print('Config file parameters set:')
    for k, v in config_dict.items():
        print(k, v)

    print('\n\n\n\n')

    path_pretrain = 'Log/saved_models/' + pretrain_experiment \
                    + '/' + pretrain_experiment + pretrain_epoch_loaded + '.th'
    print('pretrain to be loaded\n', path_pretrain)


    #### Plot loss and RMSDs from current experiment
    show = False
    plot_epoch = train_epochs
    Plotter = PlotterT1(experiment=experiment,
                        logfile_savepath='Log/losses/' + experiment + '/')
    ylim = None

    print("current experiment", experiment)
    Plotter.plot_loss(ylim=ylim, show=show)

    plot_streams = {
        'Training': True,
        'Validation': False,
        'Testing': True
    }

    plot_streams = {
        'Training': True,
        'Validation': True,
        'Testing': False
    }

    # Plotter.plot_rmsd_distribution(
    #     plot_streams=plot_streams,
    #     plot_epoch=plot_epoch,
    #     show=show)

    if experiment == 'SE3Bind_exp41_B_JT_L2loss_absDeltaE0_zeroFeatL1_wReg6sum_refModel_4L3s2v_200ep' \
        or experiment == 'SE3Bind_exp42_B_JT_L2loss_NOE0_nozeroFeatL1_refModel_2s2v'\
        or experiment=="SE3Bind_exp43_B_wreg1e3_JT_L2loss_NOE0_nozeroFeatL1_refModel_2s2v"\
        or experiment=='SE3Bind_exp44_B_L1_L2loss_NOE0_nozeroFeatL1_refModel_2s2v'\
        or experiment=='SE3Bind_exp50_B_JT_L2loss_absDeltaE0_zeroFeat_wReg6sum_refModel_4L3s2v_200ep_asign' \
        or experiment=='SE3Bind_exp51_B_JT_L2losszeroFeatL1_wReg3sum_convZerofeat_plr4_ilr2_refModel_4L3s2v_200ep_asign':
    
        Plotter.replace_median_deltaG_withavg(plot_epoch=plot_epoch, trainset=True)

        # Plotter.replace_median_deltaG_withavg(plot_epoch=plot_epoch, validset=True) ## not need for epoch 3500.



    # # plotting all combined correlation in single plot
    Plotter.plot_deltaF_vs_deltaG_correlation(
            plot_training=True,
            plot_valid=False,
            plot_testing=False,
            plot_all_combined=True,
            plot_crystal=True,
            plot_homology=True, 
            plot_epoch=plot_epoch,
            show=False,
            save=True,
            valid_crystal=False,
            valid_homology=False)


    experiments_list = [
        "SE3Bind_exp29_B_JT_zeroFeatL1_wReg3sum_convZerofeat_plr4_ilr2_refModel_4L3s2v_200ep_asign",
        "SE3Bind_exp40_B_JT_absDeltaE0_zeroFeatL1_wReg6sum_refModel_4L3s2v_200ep_asign",
        "SE3Bind_exp39_B_JT_NOE0_nozeroFeatL1_wReg3sum_convZerofeat_plr4_ilr2_refModel_2s2v_200ep_asign",
        "SE3Bind_exp51_B_JT_L2losszeroFeatL1_wReg3sum_convZerofeat_plr4_ilr2_refModel_4L3s2v_200ep_asign",
        "SE3Bind_exp50_B_JT_L2loss_absDeltaE0_zeroFeat_wReg6sum_refModel_4L3s2v_200ep_asign"
        "SE3Bind_exp42_B_JT_L2loss_NOE0_nozeroFeatL1_refModel_2s2v", #no deltaE0 with L2 loss

        "SE3Bind_exp33_B_JT_zeroFeatL1_wReg3sum_convZerofeat_plr4_ilr2_refModel_5L3s2v_200ep_asign",
        "SE3Bind_exp34_B_JT_zeroFeatL1_wReg3sum_convZerofeat_plr4_ilr2_refModel_3L3s2v_200ep_asign",
        "SE3Bind_exp45_B_absf0_L1loss_3L3s2v",
        "SE3Bind_exp46_B_noE0_L1loss_3L3s2v",

    ]

    exp_short_names_list = [
        "SE3Bind",
        "F_0 > 0",
         "F_0=0",
        "L1=L2",
        "F_0>0, L1=L2",
        "F_0=0, L1=L2",

        "More Layers",
        "Fewer Layers",
        "Fewer Layers,F_0 > 0",
        "Fewer Layers,F_0 = 0",
    
    ]



    # # # ## combining all metrics from Train, crystal and homology valid set across every 10 epoch.
    for exp in experiments_list:
        print(f"\n{'='*80}")
        print(f"Processing experiment: {exp}")
        print(f"{'='*80}\n")
        
        plotter = PlotterT1(experiment=exp,
                                logfile_savepath='Log/losses/' + exp + '/')
        
        plotter.get_all_epochs_metrics(
                    plot_epoch=train_epochs,
                    evaluate_epoch=resume_epoch,
                    plot_training=True,
                    plot_valid = False,
                    show=False,
                    save=True,
                    valid_crystal=False,
                    valid_homology=False)
        
        plotter.get_all_epochs_metrics(
                    plot_epoch=train_epochs,
                    evaluate_epoch=resume_epoch,
                    plot_training=False,
                    plot_valid = True,
                    show=False,
                    save=True,
                    valid_crystal=True,
                    valid_homology=False)
        
        plotter.get_all_epochs_metrics(
                    plot_epoch=train_epochs,
                    evaluate_epoch=resume_epoch,
                    plot_training=False,
                    plot_valid = True,
                    show=False,
                    save=True,
                    valid_crystal=False,
                    valid_homology=True)


    ## plot various metrics for all epoch and train and both valid set (crystal and homology);using csv file created by get_all_epochs_metrics
    # Plotter.plot_all_epochs_metrics(plot_metrics=False,
    #                                 save=True)
    
    # Plotter.plot_F_loss_allcombined(experiments_list,exp_short_names_list, ylim=None, show=True, save=False)
