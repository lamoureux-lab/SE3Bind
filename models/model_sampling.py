import torch
import torch.nn as nn
from models.model_docking import Docking


class SamplingModel(nn.Module):
    def __init__(self, dockingFFT,
                 device,
                 dtype,
                 experiment=None,
                 training=False,
                 docked_complex=False,
                 zero_feature=False
                 ):
        """
        Initialize sampling for the two molecular recognition tasks, IP and FI.

        :param dockingFFT: dockingFFT initialized to match dimensions of current sampling scheme
        :param device: torch device (cpu or cuda)
        :param dtype: torch data type
        :param experiment: current experiment name
        :param training: if True, model is in training mode
        :param docked_complex: if True, uses docked complex volume in forward pass
        :param zero_feature: if True, uses zero features for ablation studies
        """
        super(SamplingModel, self).__init__()

        self.docked_complex = docked_complex
        self.zero_feature = zero_feature
        self.device = device
        self.dtype = dtype
        self.training = training
        self.docker = SamplingDocker(dockingFFT, docked_complex=self.docked_complex, zero_feature=self.zero_feature, device=self.device, dtype=self.dtype)

        self.plot_idx = 0

        self.experiment = experiment

    def forward(self, receptor, ligand, docked_complex_volume=None, angle=None):
        """
        Run models with the option of multiple sampling procedures.

        :param receptor: receptor shape grid image
        :param ligand: ligand shape grid image
        :param angle: rotation, default is `None`
        :param training: train if `True`, else evaluate
        :return: depends on which model and task is being trained/evaluated
        """

        if self.docked_complex:
            energy_grid, pred_rotation_4x4, pred_txyz, rec_feat, lig_feat, docked_complex_feat,a_coef_list = self.docker(receptor=receptor, ligand=ligand,
                                                                                        docked_complex_volume=docked_complex_volume,
                                                                                        angle=angle
                                                                                        )

            return energy_grid, pred_rotation_4x4, pred_txyz, rec_feat, lig_feat, docked_complex_feat,a_coef_list
        else:
            energy_grid, pred_rotation_4x4, pred_txyz, rec_feat, lig_feat, docked_complex_feat,a_coef_list = self.docker(receptor=receptor, ligand=ligand, angle=angle)

            return energy_grid, pred_rotation_4x4, pred_txyz, rec_feat, lig_feat, docked_complex_feat, a_coef_list


class SamplingDocker(nn.Module):
    def __init__(self,
                 dockingFFT,
                 device,
                 dtype,
                 debug=False,
                 docked_complex=False,
                 zero_feature=False
                ):
        """
        Initialize docking FFT and feature generation using the SE(3)-CNN.

        :param dockingFFT: dockingFFT initialized to match dimensions of current sampling scheme
        :param device: torch device (cpu or cuda)
        :param dtype: torch data type
        :param debug:  set to True show debug verbose model and plots
        """
        super(SamplingDocker, self).__init__()

        self.docked_complex = docked_complex
        self.zero_feature = zero_feature
        self.device = device
        self.dtype = dtype
        self.dockingFFT = dockingFFT
        self.dockingConv = Docking(dockingFFT=dockingFFT, docked_complex=self.docked_complex, zero_feature=self.zero_feature, device=self.device, dtype=self.dtype, debug=debug)

    def forward(self, receptor, ligand, docked_complex_volume=None, angle=None):
        """
        Uses TorchDockingFFT() to compute feature correlations for a rotationally sampled stack of examples.

        :param receptor: receptor shape grid image
        :param ligand: ligand shape grid image
        :param angle: pass rotation(s) for angle sampled correlation
        :return: `lowest_energy`, `pred_rot`, `pred_txy`, `energy_grid`
        """
        if self.docked_complex:
            energy_grid, pred_rotation_4x4, pred_txyz, rec_feat, lig_feat, docked_complex_feat,a_coef_list = self.dockingConv(receptor=receptor, ligand=ligand,
            docked_complex_volume=docked_complex_volume,
            angle=angle)
            return energy_grid, pred_rotation_4x4, pred_txyz, rec_feat, lig_feat, docked_complex_feat,a_coef_list
        else:
            energy_grid, pred_rotation_4x4, pred_txyz, rec_feat, lig_feat, docked_complex_feat,a_coef_list = self.dockingConv(receptor=receptor, ligand=ligand, angle=angle)

            return energy_grid, pred_rotation_4x4, pred_txyz, rec_feat, lig_feat, docked_complex_feat,a_coef_list


if __name__ == "__main__":
    pass