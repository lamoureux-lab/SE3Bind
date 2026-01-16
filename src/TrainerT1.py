import os

import mrcfile
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from PlotterT1 import PlotterT1
from ProcessCoords import ProcessCoords
from Rotations import Rotations
from UtilityFunctions import UtilityFunctions


class TrainerT1:
    def __init__(self,
                 device,
                 dtype,
                 docked_complex=False,
                 zero_feature = False,
                 experiment=None,
                 dockingFFT=None,
                 sampling_model=None,
                 model_optimizer=None,
                 interaction_model=None,
                 interaction_optimizer=None,
                 resolution_in_angstroms=2.0,
                 eval_freq=10,
                 training_case='scratch',
                 path_pretrain=None,
                 target_checking=False,
                 so3_angles=None,
                 debug=False,
                 inference=False
                 ):
        """
        Initialize trainer for IP task models, paths, and class instances.

        :param dockingFFT: dockingFFT initialized to match dimensions of current sampling scheme
        :param cur_model: the current docking model initialized outside the trainer
        :param cur_optimizer: the optimizer initialized outside the trainer
        :param cur_experiment: current experiment name
        :param debug: set to `True` to check model parameter gradients
        :param plotting: create plots or not
        :param inference: set to `True` for inference mode (no training, eval with out provided deltaG)

        """
        self.device = device
        self.dtype = dtype

        self.target_checking = target_checking
        self.inference = inference

        self.dockingFFT = dockingFFT
        self.sampling_model = sampling_model.to(self.device, dtype=self.dtype)
        self.optimizer = model_optimizer

        self.interaction_model = interaction_model
        self.interaction_optimizer = interaction_optimizer

        self.experiment = experiment

        self.training_case = training_case
        self.path_pretrain = path_pretrain
        self.param_to_freeze = None
        self.set_docking_model_state()
        self.freeze_weights()

        self.resolution_in_angstroms = resolution_in_angstroms

        self.debug = debug
        self.eval_freq = eval_freq
        self.save_freq = self.eval_freq
        self.plot_freq = self.save_freq

        self.model_savepath = 'Log/saved_models/' + self.experiment + '/'
        os.makedirs(self.model_savepath, exist_ok=True)
        self.logfile_savepath = 'Log/losses/' + self.experiment + '/'
        os.makedirs(self.logfile_savepath, exist_ok=True)

        # self.log_header = 'Epoch\tLoss\tavgRMSD\n'
        # self.log_format = '%d\t%f\t%f\n'
        self.log_header = f"{'Epoch'}\t{'Loss'}\t{'F_loss'}\t{'CE_loss'}\t{'Feature0_reg_loss'}\t{'deltaF'}\t{'RMSD'}" \
                            f"\n"

        self.F_log_header = f"{'Example'}\t{'Loss'}\t{'F_loss'}\t{'CE_loss'}\t{'Feature0_reg_loss'}\t{'deltaF'}\t{'deltaG'}\t{'RMSD'}" \
                            f"\t{'clusterID'}\t{'structureIDs'}" \
                            f"\n"

        self.U = UtilityFunctions(device=self.device, dtype=self.dtype)
        self.box_dim = dockingFFT.box_dim
        self.padded_dim = dockingFFT.padded_dim
        self.padded_dim_center_3index = (self.padded_dim / 2,) * 3

        self.so3_angles = so3_angles
        self.R = Rotations(device=self.device, dtype=self.dtype)
        self.rot_grid_path = '../data/saved_angular_grids/'
        adj_threshold = 0.95
        angle_inc = 20
        param_string = self.rot_grid_path + 'so3_angular_grid_threshold' + str(adj_threshold) + 'angle_inc' + str(
            angle_inc)
        pkl_dict = param_string + '.pkl'
        self.so3_adjacency_dict = self.U.read_pkl(pkl_dict)

        self.P = ProcessCoords(dim=self.padded_dim, resolution_in_angstroms=self.resolution_in_angstroms,
                               device=self.device, dtype=self.dtype)
        self.atom_keys = "CNOS"

        self.CE_loss = torch.nn.CrossEntropyLoss()
        self.L1_loss = torch.nn.L1Loss()
        self.L2_loss = torch.nn.MSELoss()

        self.Plotter = PlotterT1(experiment=self.experiment,
                            logfile_savepath='Log/losses/' + self.experiment + '/')
        self.docked_complex = docked_complex
        self.zero_feature = zero_feature

   
        self.wReg = 1e-3
        self.zero_value = torch.zeros(1).squeeze().to(device=self.device, dtype=self.dtype)

        self.zeros_box = torch.zeros(self.box_dim, self.box_dim, self.box_dim).squeeze().to(device=self.device, 
                                                                                            dtype=self.dtype)

    @staticmethod
    def calc_RMSD(groundtruth_coords_antigen, predicted_coords_antigen, atom_keys="CNOS"):
        '''
        Calculate the RMSD between groundtruth and predicted ligand (antigen) coordinates
        Args:
            groundtruth_coords_antigen:
            predicted_coords_antigen:
            atom_keys:

        Returns:

        '''
        lengths = []
        rmsd_list = []
        for key in atom_keys:
            rmsd_list.append(
                (((groundtruth_coords_antigen[key] - predicted_coords_antigen[key]) ** 2).mean() ** 0.5).mean())
            # print(len(rmsd_dict[key].index))
            lengths.append(len(groundtruth_coords_antigen[key].index))

        rmsd_arr = np.array(rmsd_list)
        # # print(lengths)

        RMSD = rmsd_arr.mean()

        return RMSD

    def preprocess_coordinate_data(self, list_of_AG_coord_dicts, list_of_AB_coord_dicts, gt_txyz_3x1):
        """

        Args:
            list_of_AG_coord_dicts:
            list_of_AB_coord_dicts:
            gt_txyz_3x1:

        Returns:
            target_flatindex:
            df_AB_coord_dfs:
            df_AG_coord_dfs:
        """
        ### processing dataset atomic coordinates used in RMSD calculation
        ### encode flattened target index for CE loss
        target_flatindex = self.dockingFFT.encode_transform(gt_txyz_3x1)

        ### check for sulfurs
        def parse_dict_of_df_by_atom_channels(list_of_coord_dicts):
            dict_coord_dfs = {}
            cur_idx = 0
            for current_dict in list_of_coord_dicts:
                current_atom_shape = current_dict.shape
                if len(current_atom_shape) > 2:
                    current_dict = current_dict.squeeze()
                    current_atom_shape = current_dict.shape
                if len(current_atom_shape) < 2:
                    current_dict = current_dict.view(1, 3)
                    # current_atom_shape = current_dict.shape
                df_coord_dfs = pd.DataFrame(current_dict, columns=['x', 'y', 'z'])
                dict_coord_dfs[self.atom_keys[cur_idx]] = df_coord_dfs
                cur_idx += 1

            return dict_coord_dfs

        dict_AB_coord_dfs = parse_dict_of_df_by_atom_channels(list_of_AB_coord_dicts)
        dict_AG_coord_dfs = parse_dict_of_df_by_atom_channels(list_of_AG_coord_dicts)

        return target_flatindex, dict_AB_coord_dfs, dict_AG_coord_dfs

    def train_model(self, train_epochs, train_stream=None, valid_stream=None, test_stream=None,
                    resume_training=False,
                    resume_epoch=0):
        """
        Train model for specified number of epochs and data streams.

        :param train_epochs:  number of epoch to train
        :param train_stream: training set data stream
        :param valid_stream: valid set data stream
        :param test_stream: test set data stream
        :param resume_training: resume training from a loaded model state or train fresh model
        :param resume_epoch: epoch to load model and resume training
        """
        self.train_epochs = train_epochs

        ### Continue training on existing model?
        start_epoch = self.resume_training_or_not(resume_training, 
                                                  resume_epoch,
                                                  train_stream=train_stream,
                                                  valid_stream=valid_stream,
                                                  test_stream=test_stream
                                                  )
        num_epochs = start_epoch + train_epochs
        self.num_epochs = num_epochs

        for epoch in range(start_epoch, num_epochs):
            checkpoint_dict = {
                'epoch': epoch,
                'state_dict': self.sampling_model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }
            interaction_checkpoint_dict = {
                'epoch': epoch,
                'state_dict': self.interaction_model.state_dict(),
                'optimizer': self.interaction_optimizer.state_dict(),
            }
            if train_stream:
                ### Training epoch
                stream_name = 'TRAINset'
                self.run_epoch(train_stream, epoch, training=True, stream_name=stream_name)

                self.Plotter.plot_loss(show=False)
                plot_streams = {
                    'Training': True,
                    'Validation': False,
                    'Testing': False
                }
                ## uncomment if want to plot while training
                # self.Plotter.plot_rmsd_distribution(
                #     plot_streams=plot_streams,
                #     plot_epoch=epoch,
                #     show=False)

                # self.Plotter.plot_deltaF_vs_deltaG_correlation(
                #                     plot_training=True,
                #                     plot_epoch=epoch)

                #### saving model while training
                if epoch % self.save_freq == 0 and epoch > 0:
                    model_savefile = self.model_savepath + 'docking_' + self.experiment + str(epoch) + '.th'
                    self.save_checkpoint(checkpoint_dict, model_savefile, self.sampling_model)
                    print('saving model ' + model_savefile)

                    interaction_savepath = self.model_savepath + self.experiment + str(epoch) + '.th'
                    self.save_checkpoint(interaction_checkpoint_dict, interaction_savepath, self.interaction_model)
                    print('saving interaction model ' + interaction_savepath)

            ### Evaluation epoch(s)
            if epoch % self.eval_freq == 0 and epoch > 0:
                print(f'Evaluating model at epoch {epoch}')
                if valid_stream:
                    stream_name = 'VALIDset'
                    self.run_epoch(valid_stream, epoch, training=False, stream_name=stream_name)

                    plot_streams = {
                        'Training': True,
                        'Validation': True,
                        'Testing': False
                    }

                    ## uncomment if want to plot while training             
                    # self.Plotter.plot_rmsd_distribution(
                    #     plot_streams=plot_streams,
                    #     plot_epoch=epoch
                    # )

                    # self.Plotter.plot_deltaF_vs_deltaG_correlation(
                    #     plot_valid =True,
                    #     plot_epoch=epoch
                    # )

                if test_stream:
                    stream_name = 'TESTset'
                    self.run_epoch(test_stream, epoch, training=False, stream_name=stream_name)
                    plot_streams = {
                            'Training': False,
                            'Validation': False,
                            'Testing': True
                        }
                    # self.Plotter.plot_rmsd_distribution(
                    #     plot_streams=plot_streams,
                    #     plot_epoch=epoch
                    # )
                    # self.Plotter.plot_deltaF_vs_deltaG_correlation(
                    #     plot_testing=True,
                    #     plot_epoch=epoch
                    # )

    def run_epoch(self, data_stream, epoch, training=False, stream_name=None):
        """
        Run the model for an epoch.

        :param data_stream: input data stream
        :param epoch: current epoch number
        :param training: set to `True` for training, `False` for evalutation.
        :param stream_name: name of the data stream
        """
        stream_loss = []
        pos_idx = torch.tensor([0])

        if stream_name == 'VALIDset':
            prefix = 'BM_sab_HM'

            rmsd_logfile = self.logfile_savepath + 'log_RMSDs' + prefix +stream_name + '_epoch' + str(epoch) + self.experiment + '.txt'
            F_logfile = self.logfile_savepath + 'log_F_' + prefix+stream_name + '_epoch' + str(epoch) + self.experiment + '.txt'
            loss_logfile = self.logfile_savepath + 'log_loss_' + prefix+stream_name + '_' + self.experiment + '.txt'
        
        else: #crystal structures
            rmsd_logfile = self.logfile_savepath + 'log_RMSDs' +stream_name + '_epoch' + str(epoch) + self.experiment + '.txt'
            F_logfile = self.logfile_savepath + 'log_F_' + stream_name + '_epoch' + str(epoch) + self.experiment + '.txt'
            loss_logfile = self.logfile_savepath + 'log_loss_' + stream_name + '_' + self.experiment + '.txt'

        if stream_name == 'TESTset':
            rmsd_logfile = self.logfile_savepath + 'log_RMSDs' + stream_name + '_epoch' + str(epoch) + self.experiment + '.txt'
            F_logfile = self.logfile_savepath + 'log_F_' + stream_name + '_epoch' + str(epoch) + self.experiment + '.txt'
            loss_logfile = self.logfile_savepath + 'log_loss_' + stream_name + '_' + self.experiment + '.txt'

        if not os.path.exists(rmsd_logfile):
            with open(rmsd_logfile, 'w') as fout:
                fout.close()

        if not os.path.exists(F_logfile):
            with open(F_logfile, 'w') as fout:
                fout.write(self.F_log_header)
                fout.close()

        for data in tqdm(data_stream):
            train_output = [
                self.run_model(data=data, pos_idx=pos_idx, training=training, epoch=epoch)]
            
            stream_loss.append(train_output)
            
            if stream_name == "TEST_set":
                with open(F_logfile, 'a') as log_file:
                    loss, F_loss, CE_loss, feature0_reg_loss, deltaF, RMSD = train_output[0]
                    log_file.write(
                        f"{pos_idx.item()}\t{loss}\t{F_loss}\t{CE_loss}\t{feature0_reg_loss}\t{deltaF}\t{RMSD}"
                        f"\t{self.cluster_id}\t{self.structure_ids}"
                        f"\n"
                    )
                    log_file.close()
            else:
                with open(F_logfile, 'a') as log_file:
                    loss, F_loss, CE_loss, feature0_reg_loss, deltaF, RMSD = train_output[0]
                    deltaG_median = self.deltaG_stack.median() if self.deltaG_stack.numel() > 0 else 0.0
                    log_file.write(
                        f"{pos_idx.item()}\t{loss}\t{F_loss}\t{CE_loss}\t{feature0_reg_loss}\t{deltaF}\t{deltaG_median}\t{RMSD}"
                        f"\t{self.cluster_id}\t{self.structure_ids}"
                        f"\n"
                    )
                    log_file.close()

            pos_idx += 1

        avg_loss_output = np.average(stream_loss, axis=0)[0, :]

        avg_loss, avg_F_loss, avg_CE_loss, feature0_reg_loss, avg_deltaF, avg_RMSD = avg_loss_output
        print('\nEpoch', epoch, stream_name, ':',
              'loss', avg_loss,
              'F_loss', avg_F_loss,
              'CE_loss', avg_CE_loss,
              'Feature0_reg_loss', feature0_reg_loss,
              'deltaF', avg_deltaF,
              'RMSD', avg_RMSD
              )
        with open(loss_logfile, 'a') as log_file:
            log_file.write(
                f"{epoch}\t{avg_loss}\t{avg_F_loss}\t{avg_CE_loss}\t{feature0_reg_loss}\t{avg_deltaF}\t{avg_RMSD}\n"
            )

    def run_model(self, data, pos_idx, training=False, epoch=0):
        """
        Run a model iteration on the current example.

        :param data: training example
        :param pos_idx: current example position index
        :param training: set to `True` for training, `False` for evalutation.
        :param stream_name: data stream name
        :param epoch: epoch count used in plotting
        :return: `loss` and `rmsd`
        """
        if self.docked_complex:
            receptor, ligand, \
            docked_complex_volume,  \
            gt_rot_4x4, gt_txyz_4x4, \
            list_of_AB_coord_dicts, list_of_AG_coord_dicts, \
            deltaGlist, cluster_id, structure_ids = data
            docked_complex_volume = docked_complex_volume.to(device=self.device, dtype=self.dtype)
        else:
            receptor, ligand, gt_rot_4x4, gt_txyz_4x4, \
            list_of_AB_coord_dicts, list_of_AG_coord_dicts, \
            deltaGlist, cluster_id, structure_ids = data

        receptor = receptor.to(device=self.device, dtype=self.dtype)
        ligand = ligand.to(device=self.device, dtype=self.dtype)

        gt_rot_4x4 = gt_rot_4x4.to(device=self.device, dtype=self.dtype).squeeze()
        gt_txyz_4x4 = gt_txyz_4x4.to(device=self.device, dtype=self.dtype).squeeze()
        

        gt_txyz_3x1 = gt_txyz_4x4[:3, 3]

        target_flatindex, dict_AB_coord_dfs, dict_AG_coord_dfs = self.preprocess_coordinate_data(list_of_AG_coord_dicts,
                                                                                             list_of_AG_coord_dicts,
                                                                                             gt_txyz_3x1)
        if training:
            self.sampling_model.train()
            self.interaction_model.train()
        else:
            self.sampling_model.eval()
            self.interaction_model.eval()

        if self.docked_complex:
            energy_grid, pred_rotation_4x4, pred_txyz, rec_feat, lig_feat, docked_complex_feat,a_coef_list = self.sampling_model(
                receptor, ligand,
                docked_complex_volume=docked_complex_volume,
                angle=gt_rot_4x4,
                )

            if self.zero_feature is True:
                receptor_feat_0 = rec_feat[:, 0, :, :, :]
                ligand_feat_0 = lig_feat[:, 0, :, :, :]
                docked_complex_feat_0 = docked_complex_feat[:, 0, :, :, :]
            else:
                receptor_feat_0 = rec_feat
                ligand_feat_0 = lig_feat
                docked_complex_feat_0 = docked_complex_feat

            deltaF, log_N_0, delta_E_0 = self.interaction_model(energy_grid=energy_grid,
                                                 docked_complex_feat_0=docked_complex_feat_0,
                                                 receptor_feat_0=receptor_feat_0, ligand_feat_0=ligand_feat_0,
                                                )
        else:
            energy_grid, pred_rotation_4x4, pred_txyz, rec_feat, lig_feat, = self.sampling_model(receptor, ligand,
                                                                                                angle=gt_rot_4x4,
                                                                                                )
            deltaF, log_N_0, delta_E_0 = self.interaction_model(energy_grid=energy_grid,
                                                 docked_complex_feat_0=None,
                                                 receptor_feat_0=None, ligand_feat_0=None,
                                                )
        ## translation vector only RMSD
        gt_txyz_mod = gt_txyz_3x1.clone()
        pred_txyz_mod = pred_txyz.clone()
        rmsd_out = (((gt_txyz_mod - pred_txyz_mod) ** 2).mean() ** 0.5).detach().cpu().numpy()


        if self.debug:
            ## full atomic coordinate RMSD
            with torch.no_grad():
                rmsd_out = self.coord_rmsd(dict_AB_coord_dfs, dict_AG_coord_dfs,
                                           gt_rot_4x4, pred_txyz,
                                           gt_rot_4x4, gt_txyz_4x4
                                           )

                print('pred_txyz', pred_txyz)
                print('gt_txyz_4x4', gt_txyz_4x4[:3, 3])
                print('RMSD coords', rmsd_out)

        ## Model plotting during final epoch evaluation or target structure checking
        if not training and epoch == self.num_epochs - 1 or self.target_checking:
            rmsd_out = self.coord_rmsd(dict_AB_coord_dfs, dict_AG_coord_dfs,
                                       pred_rotation_4x4, pred_txyz,
                                       gt_rot_4x4=gt_rot_4x4, gt_txyz_4x4=gt_txyz_4x4,
                                       cluster_id=cluster_id, structure_ids=structure_ids,
                                       pos_idx=pos_idx.item(),
                                       epoch=epoch,
                                       )

            self.plot_correlation_energy(energy_grid=energy_grid, pos_idx=pos_idx, epoch=epoch,
                                         rmsd_out=rmsd_out)

            # self.plot_input_volumes(receptor=receptor, ligand=ligand, pos_idx=pos_idx, epoch=epoch,
            #                         rmsd_out=rmsd_out)
            # print('structure_ids', structure_ids)

            ## plot features volumes for summed rec and lig
            # self.plot_feature_volumes(
            #     rec_feat=rec_feat,
            #     lig_feat=lig_feat,
            #     epoch=epoch, pos_idx=pos_idx, rmsd_out=rmsd_out,
            #     show_plot=False)

            ### plotting scalar features ####
            # validset_type = 'crystal'
            # validset_type='Homology'
            
            # self.plot_scalar_feature(
            #                 rec_feat,
            #                 lig_feat,
            #                 docked_complex_feat,
            #                 epoch,
            #                 pos_idx,
            #                 rmsd_out,
            #                 structure_ids,
            #                 a_coef_list=a_coef_list,
            #                 type_feat='scalar_1',
            #                 validset_type=validset_type,
            #                 show_plot=False,
            #                 save=True)
            
            # self.plot_scalar_feature(
            #                 rec_feat,
            #                 lig_feat,
            #                 docked_complex_feat,
            #                 epoch,
            #                 pos_idx,
            #                 rmsd_out,
            #                 structure_ids,
            #                 a_coef_list=a_coef_list,
            #                  validset_type=validset_type,
            #                 type_feat='scalar_2',
            #                 show_plot=False,
            #                 save=True)
            
            
            # self.plot_scalar_feature(
            #                 rec_feat,
            #                 lig_feat,
            #                 docked_complex_feat,
            #                 epoch,
            #                 pos_idx,
            #                 rmsd_out,
            #                 structure_ids,
            #                 a_coef_list=a_coef_list,
            #                 type_feat='scalar_0',
            #                 validset_type=validset_type,
            #                 show_plot=False,
            #                 save=True  )
            
            # self.plot_scalar_feature(
            #                 rec_feat,
            #                 lig_feat,
            #                 docked_complex_feat,
            #                 epoch,
            #                 pos_idx,
            #                 rmsd_out,
            #                 structure_ids,
            #                 a_coef_list=a_coef_list,
            #                 type_feat='sum', 
            #                 validset_type=validset_type,
            #                 show_plot=False,
            #                 save=True)
            
            # self.plot_scalar_feature(
            #                 rec_feat,
            #                 lig_feat,
            #                 docked_complex_feat,
            #                 epoch,
            #                 pos_idx,
            #                 rmsd_out,
            #                 structure_ids,
            #                 a_coef_list=a_coef_list,
            #                 type_feat='diff',
            #                 validset_type=validset_type,
            #                 show_plot=False,
            #                 save=True)

        ## Prepare deltaG_stack for both training and evaluation
        if self.inference:  ## Inference mode (no deltaG provided)
            deltaG_stack = torch.zeros(len(deltaGlist), device=self.device, dtype=self.dtype)
        elif training: ##### For training/testing data (values are DeltaG)
            deltaG_stack = torch.stack([-i for i in deltaGlist], dim=0).to(device=self.device, dtype=self.dtype)
        elif not training and not self.inference and epoch == self.num_epochs - 1:  ## For valid set (values are already negative, (-DeltatG))
            deltaG_stack = torch.stack([i for i in deltaGlist], dim=0).to(device=self.device, dtype=self.dtype)
        
        ## declare "globally" within class for file logging
        self.deltaG_stack = deltaG_stack
        self.cluster_id = cluster_id
        self.structure_ids = structure_ids

        if training:
            #### Loss functions
            loss = torch.zeros(1, requires_grad=True).to(device=self.device, dtype=self.dtype) 

            ## L1 loss ##
            L1_loss_list = [self.L1_loss(deltaF, deltaG) for deltaG in deltaG_stack]
            L1_loss_tensor = torch.stack(L1_loss_list)  
            F_loss = torch.median(L1_loss_tensor)
            loss += F_loss

            ### L2 loss ##
            # L2_loss_list = [self.L2_loss(deltaF, deltaG) for deltaG in deltaG_stack]
            # F_loss = sum(L2_loss_list) / len(L2_loss_list)
            # loss += F_loss

            if self.zero_feature is True:
                reg_loss = torch.sum(torch.abs(docked_complex_feat_0)) \
                            + torch.sum(torch.abs(receptor_feat_0)) \
                            + torch.sum(torch.abs(ligand_feat_0))
                feature0_reg_loss = self.wReg * reg_loss
                loss += feature0_reg_loss
            else:
                feature0_reg_loss = torch.zeros(1, requires_grad=True).to(device=self.device, dtype=self.dtype)  

            if self.JT:
                CE_loss = self.CE_loss(-energy_grid.flatten().unsqueeze(0), target_flatindex.unsqueeze(0))
                loss += CE_loss

            else:
                CE_loss = torch.zeros(1).to(device=self.device, dtype=self.dtype)
        else:
            loss = torch.zeros(1).to(device=self.device, dtype=self.dtype)
            F_loss = torch.zeros(1).to(device=self.device, dtype=self.dtype)
            CE_loss = torch.zeros(1).to(device=self.device, dtype=self.dtype)
            feature0_reg_loss = torch.zeros(1).to(device=self.device, dtype=self.dtype)
            deltaF = deltaF

        if training:
            self.sampling_model.zero_grad()
            self.interaction_model.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()
            self.interaction_optimizer.step()

        return loss.item(), F_loss.item(), CE_loss.item(), feature0_reg_loss.item(), deltaF.item(), rmsd_out, 

    def coord_rmsd(self, dict_AB_coord_dfs, dict_AG_coord_dfs,
                   pred_rotation_4x4, pred_txyz,
                   gt_rot_4x4, gt_txyz_4x4,
                   cluster_id=None, structure_ids=None,
                   pos_idx=0, epoch=0):

        dict_coords_AB_corner, _ = self.P.translate_coords_to_center(dict_AB_coord_dfs, desired_center=[0, 0, 0])
        dict_coords_AB_boxcenter, _ = self.P.translate_coords_to_center(dict_coords_AB_corner,
                                                                        desired_center=self.padded_dim_center_3index)

        #### corner->rotate->center RMSD
        dict_coords_corner, _ = self.P.translate_coords_to_center(dict_AG_coord_dfs, desired_center=[0, 0, 0])
        dict_coords_pred_rot_corner = self.P.transform_coords(dict_coords_corner, pred_rotation_4x4)
        dict_coords_pred_rot_boxcenter, _ = self.P.translate_coords_to_center(dict_coords_pred_rot_corner,
                                                                              desired_center=self.padded_dim_center_3index)
        pred_4x4_txyz = self.U.build_4x4_transform_mat(translation=pred_txyz.unsqueeze(0).t())
        dict_coords_pred_rot_trans = self.P.transform_coords(dict_coords_pred_rot_boxcenter, pred_4x4_txyz)

        dict_coords_corner, _ = self.P.translate_coords_to_center(dict_AG_coord_dfs, desired_center=[0, 0, 0])
        dict_coords_gt_rot_corner = self.P.transform_coords(dict_coords_corner, gt_rot_4x4)
        dict_coords_gt_rot_boxcenter, _ = self.P.translate_coords_to_center(dict_coords_gt_rot_corner,
                                                                            desired_center=self.padded_dim_center_3index)
        dict_coords_gt_rot_trans = self.P.transform_coords(dict_coords_gt_rot_boxcenter, gt_txyz_4x4)

        if self.debug:
            print('pred_txyz', pred_txyz)
            print('gt_txyz_4x4', gt_txyz_4x4[:3, 3])

        if self.target_checking:
            fig = self.P.plot_complex_scatter(list_of_dicts=[
                dict_coords_AB_boxcenter,
                dict_coords_pred_rot_trans,
            ],
                colors=[
                    {'C': 'grey', 'N': 'blue', 'O': 'black', 'S': 'yellow'},
                    {'C': 'green', 'N': 'blue', 'O': 'black', 'S': 'yellow'},
                ],
                opacities=[
                    {'C': 0.1, 'N': 0.1, 'O': 0.1, 'S': 0.1},
                    {'C': 0.5, 'N': 0.1, 'O': 0.1, 'S': 0.1},
                ],
                show=False)

            rmsd_out = 0.101010101  ## placeholder value
            target_filename = cluster_id[0][:8]  ### will give target name
            AG_ID = '_'.join(structure_ids[0].split('_')[3:5])

            # print(cluster_id, structure_ids)
            # print(target_filename, AG_ID)

            save_folder = 'Figs/Coordinate_RMSD/Target_Checking/' + self.experiment
            os.makedirs(save_folder, exist_ok=True)
            save_html_path = save_folder + '/' + \
                             'Target_ep' + str(epoch) + \
                             target_filename + AG_ID + \
                             '_ex' + str(pos_idx) + '_' + ".html"
            # + self.experiment + ".html"
        else:
            fig = self.P.plot_complex_scatter(list_of_dicts=[
                dict_coords_AB_boxcenter,
                dict_coords_gt_rot_trans,
                dict_coords_pred_rot_trans,
            ],
                colors=[
                    {'C': 'grey', 'N': 'blue', 'O': 'black', 'S': 'yellow'},
                    {'C': 'red', 'N': 'blue', 'O': 'black', 'S': 'yellow'},
                    {'C': 'green', 'N': 'blue', 'O': 'black', 'S': 'yellow'},
                ],
                opacities=[
                    {'C': 0.1, 'N': 0.1, 'O': 0.1, 'S': 0.1},
                    {'C': 0.5, 'N': 0.1, 'O': 0.1, 'S': 0.1},
                    {'C': 0.5, 'N': 0.1, 'O': 0.1, 'S': 0.1},
                ],
                show=False)

            rmsd_out = self.calc_RMSD(dict_coords_gt_rot_trans, dict_coords_pred_rot_trans, atom_keys=self.atom_keys)
            save_folder = 'Figs/Coordinate_RMSD/' + self.experiment
            os.makedirs(save_folder, exist_ok=True)
            save_html_path = save_folder + '/' + \
                             'ABcoords_epoch' + str(epoch) + '_ex' + str(pos_idx) + "_rmsd" + str(rmsd_out)[
                                                                                              :4] + self.experiment + ".html"


        fig.write_html(save_html_path)

        return rmsd_out
    
    def plot_scalar_feature(self,
                            rec_feat,
                            lig_feat,
                            docked_feature,
                            epoch,
                            pos_idx,
                            rmsd_out,
                            structure_id,
                            type_feat,
                            a_coef_list,
                            validset_type,
                            show_plot=False,
                            save=False):
        """
        Plot scalar feature maps for receptor and ligand."""

        print("Plotting scalar features!!")
        print("Validset type,", validset_type)
        print('struct_id', structure_id)


        rec_feat = rec_feat.detach().cpu().squeeze(dim=0) # remove batch dim, (dim=0)
        lig_feat = lig_feat.detach().cpu().squeeze(dim=0)
        docked_feature = docked_feature.detach().cpu().squeeze(dim=0)

        ## spliting for 1+2s ONLY!
        if self.zero_feature is True:
                receptor_feat_0 = rec_feat[ 0, :, :, :]
                ligand_feat_0 = lig_feat[0, :, :, :]
                docked_feature_0 =  docked_feature[0, :, :, :]

                # receptor_feat_0_posonly = torch.abs(receptor_feat_0)
                # ligand_feat_0_posonly = torch.abs(ligand_feat_0)
                # docked_feature_0_posonly = torch.abs(docked_feature_0)

                rec_scalar_1 = rec_feat[1, :, :, :]
                lig_scalar_1 = lig_feat[1, :, :, :]

                rec_scalar_2 = rec_feat[2, :, :, :]
                lig_scalar_2 = lig_feat[2, :, :, :] # [2:]=vector feats

               
        else:
            rec_scalar_1 = rec_feat[0, :, :, :].detach().cpu()
            lig_scalar_1 = lig_feat[0, :, :, :].detach().cpu()

            rec_scalar_2 = rec_feat[1, :, :, :].detach().cpu()
            lig_scalar_2 = lig_feat[1, :, :, :].detach().cpu()

        if type_feat == 'scalar_0':
            plot_rec = receptor_feat_0 
            plot_lig = ligand_feat_0
            plot_docked = docked_feature_0

        if type_feat == 'scalar_1':
            plot_rec = rec_scalar_1 #* torch.sqrt(rec_feat1_acoef)
            plot_lig = lig_scalar_1 #* torch.sqrt(lig_feat1_acoef)

        if type_feat == 'scalar_2':
            plot_rec = rec_scalar_2 #* torch.sqrt(rec_feat2_acoef)
            plot_lig = lig_scalar_2 #* torch.sqrt(lig_feat2_acoef)

        #Sum and diff use scalar feat 2 and 3
        if type_feat == 'sum':
            rec_scalar_sum = torch.add(rec_scalar_1, rec_scalar_2) 
            lig_scalar_sum = torch.add(lig_scalar_1, lig_scalar_2) 

            plot_rec = rec_scalar_sum
            plot_lig = lig_scalar_sum

        if type_feat == 'diff':
            rec_scalar_diff = torch.sub(rec_scalar_1, rec_scalar_2)  
            lig_scalar_diff = torch.sub(lig_scalar_1, lig_scalar_2)

            plot_rec = rec_scalar_diff
            plot_lig = lig_scalar_diff


        # ###### 2D slice ######
        # i = 0  # Index of the slice
        # rec_scalar1_slice = rec_scalars1[i, :, :] # slice along the first dimension
        # print('rec_slice_2d shape', rec_scalar1_slice.shape)
        # rec_scalar1_slice_np = rec_scalar1_slice.numpy()
        # print("rec_slice_2d_np", rec_scalar1_slice_np.shape)
        #
        # plt.figure(figsize=(10, 8))
        # sns.heatmap(rec_scalar1_slice_np, cmap="viridis", cbar=True)
        # plt.title(f"2D Heatmap of Slice along Dim 0 slice {i};")
        # plt.xlabel("X-axis")
        # plt.ylabel("Y-axis")
        #### Save 2D slices ###
        # output_filename = save_folder + '/'+ f'rec_sc1_2Dslice_epoch{str(epoch)}_example{str(pos_idx.item())}_rmsd{str(rmsd_out)[:6]}.png'
        # plt.savefig(output_filename, dpi=300, bbox_inches="tight")

        save_folder = 'Figs/Feature_volumes/' + self.experiment
        os.makedirs(save_folder, exist_ok=True)
        print("saving scalars feat map files to folder::", save_folder)

        ## plot using plotly
        # feature_volume_plot = self.P.plot_complex_volume(
        #     list_of_volumes=
        #     [
        #         torch.add(rec_scalar_1, rec_scalar_2),
        #         torch.add(lig_scalar_1, lig_scalar_2),
                
        #     ],
        #     isomins=[
        #         None,
        #         None,

        #     ],
        #     opacities=[
        #         0.25,
        #         0.25,

        #     ],
        #     surface_counts=[
        #         8,
        #         8,
        #     ],
        #     colorscale=[
        #         'Picnic',
        #         'Picnic',
        #     ],
        #     labels=[
        #         'Receptor feature summed',
        #         'Ligand feature summed',
        #     ],
        #     show=show_plot
        # )

        # save_html_path = save_folder + '/' + \
        #                  'epoch' + str(epoch) + "sum"+'.html'
        # feature_volume_plot.write_html(save_html_path)
         
        if save:
            plot_suffix = type_feat
            # plot_suffix = f'{type_feat}_abs

            if type_feat == 'scalar_0':
                mrc_pred = mrcfile.new(
                    save_folder + '/' + f'{validset_type}_{plot_suffix}_rec_epoch{str(epoch)}_exp{str(pos_idx.item())}_rmsd{str(rmsd_out)[:6]}_feats.map',
                    overwrite=True)
                mrc_pred.set_data(plot_rec.transpose(0, 2).numpy().astype(np.float32))
                mrc_pred.voxel_size = self.resolution_in_angstroms
               
                mrc_pred = mrcfile.new(
                    save_folder + '/' + f'{validset_type}_{plot_suffix}_lig_epoch{str(epoch)}_exp{str(pos_idx.item())}_rmsd{str(rmsd_out)[:6]}_feats.map',
                    overwrite=True)
                mrc_pred.set_data(plot_lig.transpose(0, 2).numpy().astype(np.float32))
                mrc_pred.voxel_size = self.resolution_in_angstroms

                mrc_pred = mrcfile.new(
                    save_folder + '/' + f'{validset_type}_{plot_suffix}_complex_epoch{str(epoch)}_exp{str(pos_idx.item())}_rmsd{str(rmsd_out)[:6]}_feats.map',
                    overwrite=True)

                mrc_pred.set_data(plot_docked.transpose(0, 2).numpy().astype(np.float32))
                mrc_pred.voxel_size = self.resolution_in_angstroms

            else:
                mrc_pred = mrcfile.new(
                    save_folder + '/' + f'{validset_type}_{plot_suffix}_rec_epoch{str(epoch)}_exp{str(pos_idx.item())}_rmsd{str(rmsd_out)[:6]}_feats.map',
                    overwrite=True)
                mrc_pred.set_data(plot_rec.transpose(0, 2).numpy().astype(np.float32))
                mrc_pred.voxel_size = self.resolution_in_angstroms

                mrc_pred = mrcfile.new(
                    save_folder + '/' + f'{validset_type}_{plot_suffix}_lig_epoch{str(epoch)}_exp{str(pos_idx.item())}_rmsd{str(rmsd_out)[:6]}_feats.map',
                    overwrite=True)
                mrc_pred.set_data(plot_lig.transpose(0, 2).numpy().astype(np.float32))
                mrc_pred.voxel_size = self.resolution_in_angstroms

    def plot_correlation_energy(self, energy_grid, pos_idx, epoch, rmsd_out):
        plotting_energy_volume = self.dockingFFT.swap_quadrants(energy_grid.squeeze().detach().cpu())
        plotting_energy_softmin_volume = -torch.softmax(-plotting_energy_volume.flatten(), dim=0).view(self.padded_dim,
                                                                                                       self.padded_dim,
                                                                                                       self.padded_dim).detach().cpu()

        correlation_energy = self.P.plot_complex_volume(
            list_of_volumes=[
                plotting_energy_volume,
                plotting_energy_softmin_volume,

            ],
            opacities=[0.1],
            isomins=[None],
            surface_counts=[8],
            colorscale=['Picnic'],
            labels=['energy volume',
                    'energy softmin'],
            show=False

        )

        save_folder = 'Figs/CorrelationFFTvolumes/' + self.experiment
        os.makedirs(save_folder, exist_ok=True)
        save_html_path = save_folder + '/' + \
                         'CorrelationEnergy_Argmin_epoch' + str(epoch) + '_ex' + str(pos_idx.item()) + "_BSrmsd" + str(
            rmsd_out)[
                                                                                                                   :4] + \
                         self.experiment.split('exp')[0] + ".html"


        correlation_energy.write_html(save_html_path)

        
        save_map_path = save_html_path
        mrc_pred = mrcfile.new(save_map_path + '_energy_volume.map', overwrite=True)
        mrc_pred.set_data(
            plotting_energy_volume.transpose(0, 2).unsqueeze(0).detach().cpu().numpy().astype(
                # plotting_energy_volume.unsqueeze(0).detach().cpu().numpy().astype(
                np.float32))
        mrc_pred.voxel_size = self.resolution_in_angstroms

        mrc_gt = mrcfile.new(save_map_path + '_softmin_volume.map', overwrite=True)
        mrc_gt.set_data(
            plotting_energy_softmin_volume.transpose(0, 2).unsqueeze(0).detach().cpu().numpy().astype(
                # plotting_energy_softmin_volume.unsqueeze(0).detach().cpu().numpy().astype(
                np.float32))
        mrc_gt.voxel_size = self.resolution_in_angstroms

    def plot_feature_volumes(self,
                             rec_feat,
                             lig_feat,
                             epoch, pos_idx, rmsd_out,
                             show_plot=False):

        # percent_max_plotted = 0.5
        # min_plot_pred = int(torch.max(feature_stack[feature_index, :, :, :]) * percent_max_plotted)
        # pred_feat_isomin, target_isomin, rec_bulk_stack_isomin = min_plot_pred, 0.01, 0.02
        # lig_bulk_stack_isomin = target_isomin

        feature_volume_plot = self.P.plot_complex_volume(
            list_of_volumes=
            [
                torch.sum(rec_feat.squeeze(), dim=0).detach().cpu(),
                torch.sum(lig_feat.squeeze(), dim=0).detach().cpu(),
            ],
            isomins=[
                None,
                None,

            ],
            opacities=[
                0.25,
                0.25,

            ],
            surface_counts=[
                8,
                8,
            ],
            colorscale=[
                'Picnic',
                'Picnic',
            ],
            labels=[
                'Receptor feature summed',
                'Ligand feature summed',
            ],
            show=show_plot
        )
        save_folder = 'Figs/Feature_volumes/' + self.experiment
        os.makedirs(save_folder, exist_ok=True)
        # print(self.cofactor_type, str(self.cofactor_type))
        save_html_path = save_folder + '/' + \
                         'epoch' + str(epoch) \
                         + '_RLfeats' \
                         + '_ex' + str(pos_idx.item()) \
                         + '_RMSD_' + str(rmsd_out)[:3] + '.html'

        feature_volume_plot.write_html(save_html_path)

    def plot_input_volumes(self,
                           receptor,
                           ligand,
                           epoch, pos_idx, rmsd_out,
                           show_plot=False):

        # percent_max_plotted = 0.5
        # min_plot_pred = int(torch.max(feature_stack[feature_index, :, :, :]) * percent_max_plotted)
        # pred_feat_isomin, target_isomin, rec_bulk_stack_isomin = min_plot_pred, 0.01, 0.02
        # lig_bulk_stack_isomin = target_isomin

        receptor_input_volume = torch.sum(receptor.squeeze(), dim=0).detach().cpu()
        ligand_input_volume = torch.sum(ligand.squeeze(), dim=0).detach().cpu()
        input_volume_plot = self.P.plot_complex_volume(
            list_of_volumes=
            [
                receptor_input_volume,
                ligand_input_volume
            ],
            isomins=[
                None,
                None,

            ],
            opacities=[
                0.25,
                0.25,

            ],
            surface_counts=[
                8,
                8,
            ],
            colorscale=[
                'Picnic',
                'Picnic',
            ],
            labels=[
                'Receptor input volume summed',
                'Ligand input volume summed',
            ],
            show=show_plot
        )
        save_folder = 'Figs/Input_volumes/' + self.experiment
        os.makedirs(save_folder, exist_ok=True)
        # print(self.cofactor_type, str(self.cofactor_type))
        save_html_path = save_folder + '/' + \
                         'epoch' + str(epoch) \
                         + '_RLvolumes' \
                         + '_ex' + str(pos_idx.item()) \
                         + '_RMSD_' + str(rmsd_out)[:3] + '.html'

        input_volume_plot.write_html(save_html_path)

        save_map_path = save_html_path
        mrc_pred = mrcfile.new(save_map_path + 'transpose02_plotting_receptor_volume.map', overwrite=True)
        mrc_pred.set_data(
            # plotting_energy_volume.transpose(0, 2).unsqueeze(0).detach().cpu().numpy().astype(
            receptor_input_volume.transpose(0, 2).unsqueeze(0).detach().cpu().numpy().astype(
                np.float32))
        mrc_pred.voxel_size = self.resolution_in_angstroms

        mrc_gt = mrcfile.new(save_map_path + 'transpose02_plotting_ligand_volume.map', overwrite=True)
        mrc_gt.set_data(
            # plotting_energy_softmin_volume.transpose(0, 2).unsqueeze(0).detach().cpu().numpy().astype(
            ligand_input_volume.transpose(0, 2).unsqueeze(0).detach().cpu().numpy().astype(
                np.float32))
        mrc_gt.voxel_size = self.resolution_in_angstroms

    @staticmethod
    def save_checkpoint(state, filename, model):
        """
        Save current state of the model to a checkpoint dictionary.

        :param state: checkpoint state dictionary
        :param filename: name of saved file
        :param model: model to save, either docking or interaction models
        """
        model.eval()
        torch.save(state, filename)

    # @staticmethod
    def load_checkpoint(self, checkpoint_fpath, model, optimizer):
        """
        Load saved checkpoint state dictionary.


        :param checkpoint_fpath: path to saved model
        :param model:  model to load, either docking or interaction models
        :param optimizer: model optimizer
        :return: `model`, `optimizer`, `checkpoint['epoch']`
        """
        model.eval()
        checkpoint = torch.load(checkpoint_fpath, map_location=self.device)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        return model, optimizer, checkpoint['epoch']

    def resume_training_or_not(self, resume_training, 
                               resume_epoch,
                               train_stream=None,
                               valid_stream=None,
                               test_stream=None
                               ):
        """
        Resume training the model at specified epoch or not.

        :param resume_training: set to `True` to resume training, `False` to start fresh training.
        :param resume_epoch: epoch number to resume from
        :return: starting epoch number, 1 if `resume_training is True`, `resume_epoch+1` otherwise.
        """

        rmsd_header_prefix = 'IF'
        rmsd_prefix = 'log_RMSDs'

        if resume_training:
            print('Loading docking model at', str(resume_epoch))
            ckp_path = self.model_savepath + 'docking_' + self.experiment + str(resume_epoch) + '.th'
            self.sampling_model, self.optimizer, start_epoch = self.load_checkpoint(ckp_path, self.sampling_model,
                                                                                    self.optimizer)
            print('Loading interaction model at', str(resume_epoch))
            ckp_path = self.model_savepath + self.experiment + str(resume_epoch) + '.th'
            self.interaction_model, self.interaction_optimizer, start_epoch = self.load_checkpoint(
                                                    ckp_path, self.interaction_model, self.interaction_optimizer)
            ## print model and params being loaded
            print('\ndocking model:\n', self.sampling_model)
            self.U.check_model_gradients(self.sampling_model)
            print('\ninteraction model:\n', self.interaction_model)
            self.U.check_model_gradients(self.interaction_model)

            print('\nSTARTING MODEL AT EPOCH', resume_epoch, '\n')
            print('\nInput: train_stream, valid_stream, test_stream')
            print(train_stream, valid_stream, test_stream)

            start_epoch = resume_epoch

            if train_stream:
                data_stream = 'TRAINset'
                rmsd_filename = self.logfile_savepath + rmsd_prefix + data_stream + '_epoch' + str(
                    start_epoch) + self.experiment + '.txt'
                rmsd_header = rmsd_header_prefix + data_stream + ' RMSD\n'
                with open(rmsd_filename, 'w') as fout:
                    fout.write(rmsd_header)
            if valid_stream:
                data_stream = 'VALIDset'
                rmsd_filename = self.logfile_savepath + rmsd_prefix + data_stream + '_epoch' + str(
                    start_epoch) + self.experiment + '.txt'
                rmsd_header = rmsd_header_prefix + data_stream + ' RMSD\n'
                with open(rmsd_filename, 'w') as fout:
                    fout.write(rmsd_header)
            if test_stream:
                data_stream = 'TESTset'
                rmsd_filename = self.logfile_savepath + rmsd_prefix + data_stream + '_epoch' + str(
                    start_epoch) + self.experiment + '.txt'
                rmsd_header = rmsd_header_prefix + data_stream + ' RMSD\n'
                with open(rmsd_filename, 'w') as fout:
                    fout.write(rmsd_header)

        else:
            ### Loss log files
            train_log_name = self.logfile_savepath + 'log_loss_TRAINset_' + self.experiment + '.txt'
            valid_log_name = self.logfile_savepath + 'log_loss_VALIDset_' + self.experiment + '.txt'
            test_log_name = self.logfile_savepath + 'log_loss_TESTset_' + self.experiment + '.txt'

            if not os.path.exists(train_log_name):
                with open(train_log_name, "w") as fout:
                    fout.write(f'{rmsd_header_prefix} Training Loss:\n')
                    fout.write(self.log_header)
            if not os.path.exists(valid_log_name):
                with open(valid_log_name, "w") as fout:
                    fout.write(f'{rmsd_header_prefix} Validation Loss:\n')
                    fout.write(self.log_header)
            if not os.path.exists(test_log_name):
                with open(test_log_name, "w") as fout:
                    fout.write(f'{rmsd_header_prefix} Testing Loss:\n')
                    fout.write(self.log_header)

            start_epoch = 0

        return start_epoch

    def freeze_weights(self):
        """
        Freeze model weights depending on the experiment training case.
        These range from A) frozen pretrained docking model,
        B) unfrozen pretrained docking model,
        C) unfrozen pretrained docking model scoring coefficients, but frozen conv net,
        D) train the docking model from scratch.
        """
        if not self.param_to_freeze:
            print('\nAll docking model params unfrozen\n')
            return
        for name, param in self.sampling_model.named_parameters():
            if self.param_to_freeze == 'all':
                print('Freeze ALL Weights', name)
                param.requires_grad = False
            elif self.param_to_freeze in name:
                print('Freeze Weights', name)
                param.requires_grad = False
            else:
                print('Unfreeze docking model weights', name)
                param.requires_grad = True

    def set_docking_model_state(self):
        """
        Initialize the docking model training case.
        A) frozen pretrained docking model,
        B) unfrozen pretrained docking model,
        C) frozen conv net, unfrozen pretrained docking model scoring coefficients
        D) train the docking model from scratch.
        """
        # CaseA: train with docking model frozen
        if self.training_case == 'A':
            print('Training expA (docking model frozen)')
            self.param_to_freeze = 'all'
            self.sampling_model.load_state_dict(torch.load(self.path_pretrain)['state_dict'])
        # CaseB: train with docking model unfrozen
        if self.training_case == 'B':
            print('Training expB (docking model unfrozen)')
            self.param_to_freeze = None
            print("testing pre_trainModle load::", type(torch.load(self.path_pretrain)['state_dict']))
            print("type", type(torch.load(self.path_pretrain)['state_dict']))
            self.sampling_model.load_state_dict(torch.load(self.path_pretrain)['state_dict'])
        # CaseC: train with docking model CNN frozen
        if self.training_case == 'C':
            print('Training expC ("a" scoring coeffs unfrozen)')
            self.param_to_freeze = 'conv'  #
            self.sampling_model.load_state_dict(torch.load(self.path_pretrain)['state_dict'])
        # Case scratch: train everything from scratch
        if self.training_case == 'scratch':
            print('Training from scratch')
            self.param_to_freeze = None
            # self.experiment = self.training_case + '_' + self.experiment

    def run_trainer(self, train_epochs, train_stream=None, valid_stream=None, test_stream=None, resume_training=False,
                    resume_epoch=0):
        """
        Helper function to run trainer.

        :param train_epochs:  number of epoch to train
        :param train_stream: training set data stream
        :param valid_stream: valid set data stream
        :param test_stream: test set data stream
        :param resume_training: resume training from a loaded model state or train fresh model
        :param resume_epoch: epoch to load model and resume training
        """
        self.train_model(train_epochs, train_stream, valid_stream, test_stream,
                         resume_training=resume_training, resume_epoch=resume_epoch)
