import torch
from torch import nn

class InteractionModel(nn.Module):
    def __init__(self, device, dtype):
        """
        Initialize parameters for Interaction module and free energy integral calculation.
        The learned parameter for free energy decision threshold, :math:`F_0`, is initialized here.
        For the free energy integral, the volume used in the denominator is also initialized here.
        """
        super(InteractionModel, self).__init__()

        self.log_N_0 = nn.Parameter(torch.zeros(1, requires_grad=True))

        R = (8.314 / 4184)
        T = 298
        self.BETA = R * T
        self.dims = (0, 1, 2)
        self.device = device
        self.dtype = dtype


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


    def forward(self, energy_grid, docked_complex_feat_0, receptor_feat_0, ligand_feat_0):
        """
        """

    
        delta_E_0 = torch.sum(docked_complex_feat_0) - torch.sum(receptor_feat_0) - torch.sum(ligand_feat_0)

        # # # sum of abs(feature_0)
        # delta_E_0 = torch.sum(torch.abs(docked_complex_feat_0)) - torch.sum(torch.abs(receptor_feat_0)) - torch.sum(torch.abs(ligand_feat_0))

        deltaF = energy_grid[0, 0, 0] + delta_E_0 + self.log_N_0 #torch.log(self.N_0) ## maybe no log(N_0)?

        return deltaF, self.log_N_0, delta_E_0 #docked_complex_feat_0, receptor_feat_0, ligand_feat_0

        # #############################################################
        # with torch.no_grad():
        #     delta_E_0 = 0

        # # # # ## for ablation study without delta_E_0
        # deltaF = energy_grid[0, 0, 0] + self.log_N_0 #torch.log(self.N_0) ## maybe no log(N_0)?

        # return deltaF, self.log_N_0, delta_E_0

if __name__ == '__main__':
    print('works')
    print(InteractionModel())
    print(list(InteractionModel().parameters()))
