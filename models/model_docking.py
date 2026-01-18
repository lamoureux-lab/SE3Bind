import e3nn
import torch
from torch import nn
from src.VoxelConvolution import Convolution
from e3nn import o3
from src.UtilityFunctions import UtilityFunctions
import re


class Docking(nn.Module):

    def __init__(self, dockingFFT, device, dtype, plot_freq=10, debug=False, docked_complex=False, zero_feature=False):
        """
        """
        super(Docking, self).__init__()
        self.docked_complex = docked_complex
        self.zero_feature = zero_feature
        self.device = device
        self.dtype = dtype
        self.debug = debug
        self.plot_freq = plot_freq
        self.a_coeffs = nn.Parameter(torch.ones(2, requires_grad=True).to(device=self.device, dtype=self.dtype))

        self.UtilityFunctions = UtilityFunctions(device=self.device, dtype=self.dtype)

        self.dockingFFT = dockingFFT
        self.normalization = "integral"  # matches standard table of real spherical harmonics

        self.kernel_size = 9
        self.num_radial_basis = 7
        self.lmax = 3
        self.stride = (1, 1, 1)

        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax=self.lmax)

        ####RefModel zero feature 3s2v DCcent
        self.scalar_input = "5x0y"
        self.irreps_per_layer = "8x0y+8x1y+4x2y"  # y parity ## 2xF exp5
        self.irreps_final_layer = "3x0y+2x1y" ## 3s 2v

        self.non_linearity = torch.tanh
        self.norm_activation = e3nn.nn.NormActivation(self.irreps_per_layer, self.non_linearity)
        self.num_feats_per_shape = self.calculate_total_features(self.irreps_final_layer)

        if self.zero_feature and self.num_feats_per_shape[0] > 1:
            self.num_feats_per_shape[0] -= 1

        self.conv1 = Convolution(device=self.device,
                                 dtype=self.dtype,
                                irreps_in=self.scalar_input,
                                irreps_out=self.irreps_per_layer,
                                irreps_sh=self.irreps_sh,
                                diameter=self.kernel_size, 
                                num_radial_basis=self.num_radial_basis,
                                normalization=self.normalization)
        self.conv2 = Convolution(device=self.device,
                                 dtype=self.dtype,
                                 irreps_in=self.irreps_per_layer,
                                 irreps_out=self.irreps_per_layer,
                                 irreps_sh=self.irreps_sh,
                                 diameter=self.kernel_size, num_radial_basis=self.num_radial_basis,
                                 normalization=self.normalization)
        self.conv3 = Convolution(device=self.device,
                                 dtype=self.dtype,
                                 irreps_in=self.irreps_per_layer,
                                 irreps_out=self.irreps_per_layer,
                                #  irreps_out=self.irreps_final_layer,
                                 irreps_sh=self.irreps_sh,
                                 diameter=self.kernel_size, num_radial_basis=self.num_radial_basis,
                                 normalization=self.normalization)
        self.conv4 = Convolution(device=self.device,
                                 dtype=self.dtype,
                                 irreps_in=self.irreps_per_layer,
                                 irreps_out=self.irreps_final_layer,
                                #  irreps_out=self.irreps_per_layer,
                                 irreps_sh=self.irreps_sh,
                                 diameter=self.kernel_size, num_radial_basis=self.num_radial_basis,
                                 normalization=self.normalization)
        # self.conv5 = Convolution(device=self.device,
        #                          dtype=self.dtype,
        #                          irreps_in=self.irreps_per_layer,
        #                          irreps_out=self.irreps_final_layer,
        #                          irreps_sh=self.irreps_sh,
        #                          diameter=self.kernel_size, num_radial_basis=self.num_radial_basis,
        #                          normalization=self.normalization)

    @staticmethod
    def calculate_total_features(input_string):
        """

        Args:
            input_string:

        Returns:

        """

        # Define the multipliers for each order of features
        n = 10
        feature_multipliers = {order: (order * 2) + 1 for order in range(n+1)}

        # Regular expression to find all matches of the form "NxM[oye] for 'odd', 'y', and 'even', parities"
        pattern = r'(\d+)x(\d+)[oye]'
        matches = re.findall(pattern, input_string)

        if not all(re.fullmatch(pattern, part) for part in input_string.split('+')):
            raise ValueError("Input string is not formatted correctly.")

        num_feats_per_ylm_order = []

        for match in matches:
            count = int(match[0])
            order = int(match[1])

            num_feats_per_ylm_order.append(count * feature_multipliers[order])

        return num_feats_per_ylm_order

    def channel_wise_activation(self, volume, activation_function, softmax=False):
        '''
        Vectorized indexing through channel dim (c).
        Args:
            volume: torch.tensor [b,c,x,y,z]

        Returns: torch.tensor [b,c,x,y,z]

        '''

        volume = volume.squeeze()
        channel_num, xyz_dim = volume.shape[0], volume.shape[-1]
        sub_arr = volume.permute(1, 2, 3, 0).contiguous().view(-1, channel_num)
        sub_arr_activated = activation_function(sub_arr)
        if softmax:
            sub_arr_activated = self.softmax(sub_arr_activated)
        result = sub_arr_activated.view(xyz_dim, xyz_dim, xyz_dim, channel_num).permute(3, 0, 1, 2)

        return result.unsqueeze(0)

    def forward(self, receptor, ligand, docked_complex_volume=None, angle=None):
        '''
        :param receptor:
        :param ligand:
        :param angle:
        :return:
        '''
        rec_feat = self.conv1(receptor)
        rec_feat = self.channel_wise_activation(rec_feat, self.norm_activation)
        rec_feat = self.conv2(rec_feat)
        rec_feat = self.channel_wise_activation(rec_feat, self.norm_activation)
        rec_feat = self.conv3(rec_feat)
        rec_feat = self.channel_wise_activation(rec_feat, self.norm_activation)
        rec_feat = self.conv4(rec_feat)
        # rec_feat = self.channel_wise_activation(rec_feat, self.norm_activation)
        # rec_feat = self.conv5(rec_feat)

        # rec_feat = rec_feat.squeeze()

        lig_feat = self.conv1(ligand)
        lig_feat = self.channel_wise_activation(lig_feat, self.norm_activation)
        lig_feat = self.conv2(lig_feat)
        lig_feat = self.channel_wise_activation(lig_feat, self.norm_activation)
        lig_feat = self.conv3(lig_feat)
        lig_feat = self.channel_wise_activation(lig_feat, self.norm_activation)
        lig_feat = self.conv4(lig_feat)
        # lig_feat = self.channel_wise_activation(lig_feat, self.norm_activation)
        # lig_feat = self.conv5(lig_feat)

        # lig_feat = lig_feat.squeeze()

        if self.docked_complex:
            docked_complex_feat = self.conv1(docked_complex_volume)
            docked_complex_feat = self.channel_wise_activation(docked_complex_feat, self.norm_activation)
            docked_complex_feat = self.conv2(docked_complex_feat)
            docked_complex_feat = self.channel_wise_activation(docked_complex_feat, self.norm_activation)
            docked_complex_feat = self.conv3(docked_complex_feat)
            docked_complex_feat = self.channel_wise_activation(docked_complex_feat, self.norm_activation)
            docked_complex_feat = self.conv4(docked_complex_feat)
            # docked_complex_feat = self.channel_wise_activation(docked_complex_feat, self.norm_activation)
            # docked_complex_feat = self.conv5(docked_complex_feat)

            # docked_complex_feat = docked_complex_feat.squeeze()
        else:
            docked_complex_feat = None

        fft_score = self.dockingFFT.dock_rotations(
            num_feats_per_shape=self.num_feats_per_shape,
            receptor_feats=rec_feat,
            ligand_feats=lig_feat,
            docked_complex_feats=docked_complex_feat,
            angle=angle,
            a_coeffs=self.a_coeffs

        )

        return fft_score


if __name__ == '__main__':
    print('works')
    print(Docking(dockingFFT=None, device='cuda', dtype=torch.float32).conv1.parameters)
