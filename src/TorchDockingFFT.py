import torch
import torch.nn.functional as F
from ProcessCoords import ProcessCoords
from UtilityFunctions import UtilityFunctions


class TorchDockingFFT:
    """
    Utility class to perform FFT-based docking.
    """

    def __init__(self,
                 device,
                 dtype,
                 box_dim,
                 padded_dim,
                 so3_angles,
                 docked_complex=False,
                 zero_feature=False,
                 swap_plot_quadrants=False, normalization='ortho',
                 debug=False, plotting=False):
        """
        Initialize docking FFT based on desired usage.

        :param padded_dim: dimension of final padded box to follow Nyquist's theorem.
        # :param num_angles: number of angles to sample
        :param angle: single angle to rotate a shape and evaluate FFT
        :param swap_plot_quadrants: swap FFT output quadrants to make plots origin centered
        :param normalization: specify normalization for the `torch.fft2()` and `torch.irfft2()` operations, default is set to `ortho`
        :param debug: shows what rotations look like depending on `num_angles`
        """
        # self.training = training
        self.docked_complex = docked_complex
        self.zero_feature = zero_feature
        self.device = device
        self.dtype = dtype
        self.debug = debug
        self.swap_plot_quadrants = swap_plot_quadrants  ## used only to make plotting look nice
        self.box_dim = box_dim
        self.padded_dim = padded_dim

        self.norm = normalization
        self.onehot_3Dgrid = torch.zeros([self.padded_dim, self.padded_dim, self.padded_dim]).to(
            self.device, dtype=self.dtype)

        self.UtilityFunctions = UtilityFunctions(device=self.device, dtype=self.dtype)

        self.plotting = plotting

        # R = torch.tensor(8.314 / 4184.0)
        # T = 298.01
        # self.BETA = R * T
        # self.BF_volume = -22.5570  ##-torch.log(torch.tensor(1854 * 150 ** 3).cuda())
        # self.one_div_sqrt_two = 1/torch.sqrt(torch.tensor(2.0, dtype=self.dtype))

        self.so3_angles = so3_angles

    def encode_transform(self, gt_txyz, target_volume=False):
        '''
        Encode the ground-truth transformation as a (flattened) 3D one-hot array.

        :param gt_txy: ground truth translation `[[x], [y], [z]`.
        :return: flattened one hot encoded array.
        '''
        index_txyz = gt_txyz.type(torch.long) % self.padded_dim
        # print(index_txy)
        self.onehot_3Dgrid[index_txyz[0], index_txyz[1], index_txyz[2]] = -1
        target_flatindex = torch.argmin(self.onehot_3Dgrid.flatten())
        if target_volume:
            import scipy.ndimage as ndimage
            return torch.tensor(
                ndimage.gaussian_filter(self.onehot_3Dgrid.squeeze().detach().cpu()
                                        , sigma=1.0, truncate=3.0), requires_grad=False).to(device=self.device, dtype=self.dtype)
        ## reset 3D one-hot array after computing flattened index
        self.onehot_3Dgrid[index_txyz[0], index_txyz[1], index_txyz[2]] = 0

        return target_flatindex

    def extract_transform(self, energy_grid):
        # Find the flat index of the minimum energy value
        pred_index = torch.argmin(energy_grid)

        # Compute the x, y, and z indices of the translation vector in the grid
        x_index = torch.div(pred_index, self.padded_dim, rounding_mode='floor') % self.padded_dim
        y_index = torch.fmod(pred_index, self.padded_dim)
        z_index = torch.div(pred_index, self.padded_dim ** 2, rounding_mode='floor')

        # Compute the translation vector
        tzxy = torch.stack((z_index, x_index, y_index), dim=0)

        # Just to make translation values look nice caused by grid wrapping + or - signs
        tzxy[tzxy > self.box_dim] -= self.padded_dim

        return tzxy

    def dock_rotations(self,
                       num_feats_per_shape,
                       receptor_feats, ligand_feats, angle,
                       a_coeffs,
                       docked_complex_feats=None
                       ):
        """
        Compute FFT scores of shape features in the space of all rotations and translation ligand features.
        Rotationally sample the the ligand feature using specified number of angles, and repeat the receptor features to match in size.
        Then compute docking energy using :func:`~dock_translations`.

        :param receptor_feats: receptor bulk and boundary feature single features
        :param ligand_feats: ligand bulk and boundary feature single features
        :param angle: angle is the case where we only want to sample 1 correlation at a specific angle, default is `None`,
            otherwise the num_angles just does `np.linspace()` from 0 to 360.
        :param weight_bulk: bulk scoring coefficient
        :param weight_crossterm: crossterm scoring coefficient
        :param weight_bound: boundary scoring coefficient
        :return: scored docking feature correlation
        """
        # print('angle', angle, angle.shape)
        # print('gt_rot_check', gt_rot_check)

        if angle is None:
            perm_angles = []
            # for i in self.so3_angles:
            #     perm_angles.append(self.U.permute_rotation(i))
            # angle_step_size = 16 # fits on 3080
            # angle_step_size = 4 ## torch64: 27.66 GiB. GPU 0 has a total capacity of 10.00 GiB of which 0 bytes is free.
            angle_step_size = 4
            for i in range(0, len(self.so3_angles), angle_step_size):
                perm_angles.append(self.UtilityFunctions.permute_rotation(self.so3_angles[i]))
            perm_angles_stack = torch.stack(perm_angles, dim=0)
            num_angles_at_inference = len(perm_angles_stack)
            rec_feat_repeated = receptor_feats.unsqueeze(0).repeat(num_angles_at_inference, 1, 1, 1, 1)
            lig_feat_repeated = ligand_feats.unsqueeze(0).repeat(num_angles_at_inference, 1, 1, 1, 1)

            lig_feat_repeated_rot = self.UtilityFunctions.rotate_volume(lig_feat_repeated, perm_angles_stack)

            rec_feat_repeated_pad, lig_feat_repeated_rot_pad = self.pad_feats(rec_feat_repeated,
                                                                              lig_feat_repeated_rot)
            print(rec_feat_repeated.shape)
            energy_grid,a_coef_list = self.dock_translations(
                                                    rec_feat_repeated_pad, lig_feat_repeated_rot_pad,
                                                    num_feats_per_shape,
                                                    a_coeffs,
                                                  
                                                    )
            print(energy_grid.shape)

            del rec_feat_repeated_pad
            del lig_feat_repeated_rot
            del lig_feat_repeated_rot_pad
            current_min_energy = float('Inf')
            pred_rot_index = None
            pred_txyz = None
            for i in range(energy_grid.shape[0]):
                pred_txyz = self.extract_transform(energy_grid[i, :, :, :])
                min_energy = energy_grid[i, pred_txyz[0], pred_txyz[1], pred_txyz[2]]
                if min_energy < current_min_energy:
                    current_min_energy = min_energy
                    pred_rot_index = i
                    pred_txyz = pred_txyz

            energy_grid = energy_grid[pred_rot_index, :, :, :]
            pred_rotation_4x4 = self.UtilityFunctions.build_4x4_transform_mat(self.so3_angles[pred_rot_index])

        else:
            if self.zero_feature is True:
                receptor_feats_FFT = receptor_feats[:, 1:, :, :, :]
                ligand_feats_FFT = ligand_feats[:, 1:, :, :, :]
           
            else:
                receptor_feats_FFT = receptor_feats
                ligand_feats_FFT = ligand_feats

            ## input already rotated. No further rotation needed.

            receptor_feats_FFT_pad, ligand_feats_FFT_pad = self.pad_feats(receptor_feats_FFT,
                                                                          ligand_feats_FFT)
            energy_grid, a_coef_list = self.dock_translations(receptor_feats_FFT_pad, ligand_feats_FFT_pad,
                                                 num_feats_per_shape,
                                                 a_coeffs,
                                                 # weight_0, weight_1
                                                 )
            pred_txyz = self.extract_transform(energy_grid)
            pred_rotation_4x4 = None

        return energy_grid, pred_rotation_4x4, pred_txyz, receptor_feats, ligand_feats, docked_complex_feats,a_coef_list
    
    ###for 4s4v a_ceof FFT implementation
    def dock_translations(self, receptor_sampled_stack, ligand_sampled_stack,
                          num_feats_per_ylm_order,
                          a_coeffs
                          ):
        """
        Compute FFT score on receptor and rotationally sampled ligand feature stacks of bulk, crossterms, and boundary features.
        Maximum score -> minimum energy.

        :param receptor_sampled_stack: `self.num_angles` repeated stack of receptor bulk and boundary features
        :param ligand_sampled_stack: `self.num_angles` *rotated* and repeated stack of receptor bulk and boundary features
        :param a_coeffs: ``a'' coefficients N/2 x N/2, one coefficient per feature pair.
        :return: FFT score using scoring function
        """
        receptor_split = torch.split(receptor_sampled_stack, split_size_or_sections=num_feats_per_ylm_order, dim=1)
        ligand_split = torch.split(ligand_sampled_stack, split_size_or_sections=num_feats_per_ylm_order, dim=1)
        last_3_dims = (-3, -2, -1)

        energy_grid = 0
        a_coef_list = []
        for i in range(len(num_feats_per_ylm_order)):
            split_index = None if num_feats_per_ylm_order[i] == 1 else num_feats_per_ylm_order[i] // 2

            a_coeff = torch.abs(a_coeffs[i])

            rec_feats = receptor_split[i]
            lig_feats = ligand_split[i]

            rec_feat_1 = rec_feats[:, :split_index, :, :, :]
            rec_feat_2 = rec_feats[:, split_index:, :, :, :]

            lig_feat_1 = lig_feats[:, :split_index, :, :, :]
            lig_feat_2 = lig_feats[:, split_index:, :, :, :]
 
            if num_feats_per_ylm_order[0] == 4 and i ==0: #4 scalar only
                cplx_rec = torch.fft.rfftn(rec_feat_1, dim=last_3_dims, norm=self.norm) #*shape [1,2,100,100,100]
                cplx_lig = torch.fft.rfftn(lig_feat_1, dim=last_3_dims, norm=self.norm)

                correlation1_intermediate = torch.fft.irfftn(cplx_rec * torch.conj(cplx_lig), dim=last_3_dims, norm=self.norm)

                # Element-wise computation
                result_0 = torch.abs(a_coeffs[0]) * correlation1_intermediate[:, 0:1,:, :, :]  # Corresponds to channel 1
                result_1 = -torch.abs(a_coeffs[0]) * correlation1_intermediate[:, 1:2,:, :, :]  # Corresponds to channel 2
    
                correlation1 = torch.cat((result_0, result_1), dim=1) # dim=1 concat along channel dim

                ### feat2
                cplx_rec = torch.fft.rfftn(rec_feat_2, dim=last_3_dims, norm=self.norm) #*shape [1,2,100,100,100]
                cplx_lig = torch.fft.rfftn(lig_feat_2, dim=last_3_dims, norm=self.norm)

                correlation2_intermediate = torch.fft.irfftn(cplx_rec * torch.conj(cplx_lig), dim=last_3_dims, norm=self.norm)

                # Element-wise computation
                result_0 = torch.abs(a_coeffs[0]) * correlation2_intermediate[:, 0:1,:, :, :]  # Corresponds to channel 1 shape 
                result_1 = -torch.abs(a_coeffs[0]) * correlation2_intermediate[:, 1:2,:, :, :]  # Corresponds to channel 2
            
                # Combine results
                correlation2 = torch.cat((result_0, result_1), dim=1) # dim=1 concat along channel dim shape[1,2,100,100,100]

            if num_feats_per_ylm_order[1] == 12 and i ==1: ## 4 vector feats only
                
                cplx_rec = torch.fft.rfftn(rec_feat_1, dim=last_3_dims, norm=self.norm)
                cplx_lig = torch.fft.rfftn(lig_feat_1, dim=last_3_dims, norm=self.norm)

                correlation1_intermediate = torch.fft.irfftn(cplx_rec * torch.conj(cplx_lig), dim=last_3_dims, norm=self.norm)
                # Element-wise computation
                result_0 = torch.abs(a_coeffs[1]) * correlation1_intermediate[:, :3,:, :, :]  # Corresponds to channel 1 shape [1,100,100,100]
                result_1 = -torch.abs(a_coeffs[1]) * correlation1_intermediate[:, 3:,:, :, :]  # Corresponds to channel 2

                correlation1 = torch.cat((result_0, result_1), dim=1) # dim=1 concat along channel dim shape[1,2,100,100,100]

                ## feature 2 vectors:
                cplx_rec = torch.fft.rfftn(rec_feat_2, dim=last_3_dims, norm=self.norm)
                cplx_lig = torch.fft.rfftn(lig_feat_2, dim=last_3_dims, norm=self.norm)
                correlation2_intermediate = torch.fft.irfftn(cplx_rec * torch.conj(cplx_lig), dim=last_3_dims, norm=self.norm)

                result_0 = torch.abs(a_coeffs[1]) * correlation2_intermediate[:, :3,:, :, :]  # Corresponds to channel 1 shape [1,100,100,100]
                result_1 = -torch.abs(a_coeffs[1]) * correlation2_intermediate[:, 3:,:, :, :]  # Corresponds to channel 2
                
                correlation2 = torch.cat((result_0, result_1), dim=1) # dim=1 concat along channel dim shape[1,2,100,100,100]

            else:
                ## <4s4v cases
                cplx_rec = torch.fft.rfftn(rec_feat_1, dim=last_3_dims, norm=self.norm)
                cplx_lig = torch.fft.rfftn(lig_feat_1, dim=last_3_dims, norm=self.norm)

                correlation1 = torch.abs(a_coeff) * torch.fft.irfftn(cplx_rec * torch.conj(cplx_lig), dim=last_3_dims, norm=self.norm)

                cplx_rec = torch.fft.rfftn(rec_feat_2, dim=last_3_dims, norm=self.norm)
                cplx_lig = torch.fft.rfftn(lig_feat_2, dim=last_3_dims, norm=self.norm)

                correlation2 = -torch.abs(a_coeff) * torch.fft.irfftn(cplx_rec * torch.conj(cplx_lig), dim=last_3_dims, norm=self.norm)

            energy_grid += torch.sum(correlation1, dim=1)
            energy_grid += torch.sum(correlation2, dim=1)

            a_coef_list.append([a_coeffs[0],a_coeffs[1]])

        return energy_grid.squeeze(), a_coef_list


    def pad_feats(self, rec_feat_repeated, lig_feat_repeated, padded_dim_explicit=None):
        if padded_dim_explicit:
            padded_dim = padded_dim_explicit
        else:
            padded_dim = self.padded_dim
        initbox_size = rec_feat_repeated.shape[-1]
        if initbox_size < padded_dim:
            pad_size = (padded_dim - initbox_size) // 2
            if initbox_size % 2 == 0:
                padding = [pad_size, pad_size, pad_size, pad_size, pad_size, pad_size + (initbox_size % 2)]
            else:
                padding = [pad_size, pad_size + 1, pad_size, pad_size + 1, pad_size, pad_size + (initbox_size % 2)]
            rec_feat_repeated = F.pad(rec_feat_repeated, pad=padding, mode='constant', value=0)
            lig_feat_repeated = F.pad(lig_feat_repeated, pad=padding, mode='constant', value=0)

        return rec_feat_repeated, lig_feat_repeated

    def swap_quadrants(self, input_volume):
        """
        FFT returns features centered with the origin at the center of the image, not at the corners.
        :param input_volume: FFT output array
        """
        # num_features = input_volume.size(0)
        L = input_volume.size(-1)
        L2 = int(L / 2)
        output_volume = torch.zeros(L, L, L, device=input_volume.device, dtype=self.dtype)

        output_volume[:L2, :L2, :L2] = input_volume[L2:L, L2:L, L2:L]
        output_volume[L2:L, L2:L, L2:L] = input_volume[:L2, :L2, :L2]

        output_volume[:L2, L2:L, :L2] = input_volume[L2:L, :L2, L2:L]
        output_volume[L2:L, :L2, L2:L] = input_volume[:L2, L2:L, :L2]

        output_volume[:L2, :L2, L2:L] = input_volume[L2:L, L2:L, :L2]
        output_volume[L2:L, L2:L, :L2] = input_volume[:L2, :L2, L2:L]

        output_volume[:L2, L2:L, L2:L] = input_volume[L2:L, :L2, :L2]
        output_volume[L2:L, :L2, :L2] = input_volume[:L2, L2:L, L2:L]

        return output_volume

    def check_feat_rotation(self, ligand_feats, lig_feat_repeated):
        ## check rotation working
        P = ProcessCoords(dim=self.padded_dim // 2)
        surface_count = 5
        # print(ligand_feats.shape)
        # print(lig_feat_rot_sampled.shape)
        plot_ligand_feats = (ligand_feats[0, :, :, :]).detach().cpu()
        plot_lig_feat_rot_sampled = (lig_feat_repeated.squeeze()[0, :, :, :]).detach().cpu()
        P.plot_volume(volume=plot_ligand_feats, surface_count=surface_count, isomin=None)
        P.plot_volume(volume=plot_lig_feat_rot_sampled, surface_count=surface_count, isomin=None)

    def plot_feats(self, receptor_feats, ligand_feats):
        ## plot passed features
        with torch.no_grad():
            P = ProcessCoords(dim=self.padded_dim // 2)
            # surface_count = 10
            plot_receptor_feats_bulk = (receptor_feats[0, :, :, :]).detach().cpu()
            plot_ligand_feats_bulk = (ligand_feats[0, :, :, :]).detach().cpu()
            plot_receptor_feats_bound = (receptor_feats[1, :, :, :]).detach().cpu()
            plot_ligand_feats_bound = (ligand_feats[1, :, :, :]).detach().cpu()
            P.plot_complex_volume(list_of_volumes=
                                  [plot_receptor_feats_bulk,
                                   # plot_ligand_feats_bulk,
                                   # plot_receptor_feats_bound,
                                   # plot_ligand_feats_bound
                                   ],
                                  surface_counts=[10],
                                  colorscale=['RdBu'],
                                  midpoints=[0]
                                  )

    def plot_correlation(self, plot_energy, weight_bulk=None, weight_crossterm=None, weight_bound=None, show=False):
        # plot energies of translation correlation working
        P = ProcessCoords(dim=self.padded_dim)

        max_plotting_threshold = 0.0
        offset = 1.0
        min_plotting_threshold = plot_energy.min().item() + offset
        plot_energy = plot_energy.trunc()
        # plot_energy = plot_energy[25:75,25:75,25:75]
        plot_energy_filtered = plot_energy.clone()
        plot_energy_filtered[plot_energy_filtered > min_plotting_threshold] = max_plotting_threshold

        # plot_score_clone[plot_score_clone < min_plotting_threshold] = -1000

        if show:
            P.plot_complex_volume(list_of_volumes=[plot_energy, plot_energy_filtered],
                                  colorscale=['Picnic'],
                                  isomins=[None],
                                  opacities=[0.25],
                                  surface_counts=[3],
                                  show=False
                                  )

        print('\nPlotting correlation:')
        print("\nlowest energy", plot_energy.min())
        print('\nweight_bulk, weight_crossterm, weight_bound', weight_bulk, weight_crossterm, weight_bound)

        return plot_energy, plot_energy_filtered, plot_energy.min().item()


class SwapQuadrants(torch.autograd.Function):

    def forward(ctx, input_volume, **kwargs):
        """
        FFT returns features centered with the origin at the center of the image, not at the corners.
        :param **kwargs:
        :param input_volume: FFT output array
        """
        L = input_volume.size(-1)
        L2 = int(L / 2)
        output_volume = torch.zeros(L, L, L, device=input_volume.device, dtype=input_volume.dtype)

        output_volume[:L2, :L2, :L2] = input_volume[L2:L, L2:L, L2:L]
        output_volume[L2:L, L2:L, L2:L] = input_volume[:L2, :L2, :L2]

        output_volume[:L2, L2:L, :L2] = input_volume[L2:L, :L2, L2:L]
        output_volume[L2:L, :L2, L2:L] = input_volume[:L2, L2:L, :L2]

        output_volume[:L2, :L2, L2:L] = input_volume[L2:L, L2:L, :L2]
        output_volume[L2:L, L2:L, :L2] = input_volume[:L2, :L2, L2:L]

        output_volume[:L2, L2:L, L2:L] = input_volume[L2:L, :L2, :L2]
        output_volume[L2:L, :L2, :L2] = input_volume[:L2, L2:L, L2:L]

        return output_volume

    def backward(ctx, input_volume):
        """
        FFT returns features centered with the origin at the center of the image, not at the corners.
        :param input_volume: FFT output array
        """
        L = input_volume.size(-1)
        L2 = int(L / 2)
        output_volume = torch.zeros(L, L, L, device=input_volume.device, dtype=input_volume.dtype)

        output_volume[:L2, :L2, :L2] = input_volume[L2:L, L2:L, L2:L]
        output_volume[L2:L, L2:L, L2:L] = input_volume[:L2, :L2, :L2]

        output_volume[:L2, L2:L, :L2] = input_volume[L2:L, :L2, L2:L]
        output_volume[L2:L, :L2, L2:L] = input_volume[:L2, L2:L, :L2]

        output_volume[:L2, :L2, L2:L] = input_volume[L2:L, L2:L, :L2]
        output_volume[L2:L, L2:L, :L2] = input_volume[:L2, :L2, L2:L]

        output_volume[:L2, L2:L, L2:L] = input_volume[L2:L, :L2, :L2]
        output_volume[L2:L, :L2, :L2] = input_volume[:L2, L2:L, L2:L]

        return output_volume


if __name__ == '__main__':
    pass
