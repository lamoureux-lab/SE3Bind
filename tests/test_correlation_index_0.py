import torch

# dims = (20,20,20)
# one_hot_index = (10,10,10)
last_3_dims = (0,1,2)
array1 = torch.zeros(20,20,20)
array2 = torch.zeros(20,20,20)

print('array1.shape', array1.shape)

array1[10,10,10] = 1
array2[10,10,10] = 1

import torch.nn.functional as F
def pad_feats(rec_feat_repeated, lig_feat_repeated):
    padded_dim = rec_feat_repeated.shape[-1] * 2
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


array1_padded, array2_padded = pad_feats(array1, array2)
cplx_rec = torch.fft.rfftn(array1_padded, dim=last_3_dims)
cplx_lig = torch.fft.rfftn(array2_padded, dim=last_3_dims)
energy_grid = torch.fft.irfftn(cplx_rec * torch.conj(cplx_lig), dim=last_3_dims)

print('energy_grid.shape', energy_grid.shape)
print(energy_grid[0,0,0], energy_grid[1,1,1], energy_grid[-1,-1,-1],)
# argmax_ind = torch.argmax(energy_grid)
# print(argmax_ind)
