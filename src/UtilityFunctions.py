import pickle as pkl
import torch
import torch.nn.functional as F
from pathlib import Path


class UtilityFunctions:
    def __init__(self, device, dtype, experiment=None):
        self.device = device
        self.dtype = dtype
        self.experiment = experiment
        self.bottom_row_1x4 = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).to(self.device, dtype=self.dtype)
        self.right_col_3x1 = torch.tensor([[0.0], [0.0], [0.0]]).to(self.device, dtype=self.dtype)

    @staticmethod
    def read_config(file_path):
        """
        Read config file with experiment parameters.
        Args:
            file_path: ./config.txt

        Returns: parsed config variables

        """
        config_dict = {}
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    value = value.strip()
                    if str(value).lower() == 'none':
                        value = None
                    if str(value).lower() == 'true':
                        value = True
                    if str(value).lower() == 'false':
                        value = False
                    config_dict[key.strip()] = value
        return config_dict

    @staticmethod
    def create_directory(path):
        """
        Create a directory and any necessary parent directories.

        :param path: The path of the directory to create.
        :type path: str
        """
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
            print(f"Directory '{path}' created successfully")
        except Exception as error:
            print(f"Error creating directory '{path}': {error}")

    @staticmethod
    def write_pkl(data, filename):
        '''
        :param data: to write to .pkl  file
        :param filename: specify `filename.pkl`
        '''
        print('\nwriting ' + filename + ' to .pkl\n')
        with open(filename, 'wb') as fout:
            pkl.dump(data, fout)
        fout.close()

    @staticmethod
    def read_pkl(filename):
        '''
        :param filename: `filename.pkl` to load
        :return: data
        '''
        print('\nreading ' + filename + '\n')
        with open(filename, 'rb') as fin:
            data = pkl.load(fin)
        fin.close()
        return data

    @staticmethod
    def check_model_gradients(model):
        '''
        Check current model parameters and gradients in-place.
        Specifically if weights are frozen or updating
        '''
        for n, p in model.named_parameters():
            if p.requires_grad:
                print('name', n, 'param', p, 'gradient', p.grad)

    @staticmethod
    def weights_init(model):
        '''
        Initialize weights for SE(3)-equivariant convolutional network.
        Generally unused for SE(3) network, as e2nn library has its own Kaiming He weight initialization.
        '''
        if isinstance(model, torch.nn.Conv3d):
            print('updating convnet weights to kaiming uniform initialization')
            torch.nn.init.kaiming_uniform_(model.weight)

    def build_4x4_transform_mat(self, rot_3x3=torch.eye(3), translation=None):
        '''
        Build 4x4 transformation matrix using 3x3 rotation and 3x1 translation column, padded at the bottom by a 1x4 [[0,0,0,1]]
        :param rot_3x3: 1x3x3 rotation matrix
        :param translation: 1x3x1 translation
        :return: 1x4x4 transformation matrix
        '''
        rot_3x3 = rot_3x3.to(device=self.device, dtype=self.dtype)
        if translation is not None:
            translation = translation.to(device=self.device, dtype=self.dtype)
            out3x4 = torch.cat((rot_3x3, translation), dim=1)
        else:
            out3x4 = torch.cat((rot_3x3, self.right_col_3x1), dim=1)
        transform_matrix = torch.cat((out3x4, self.bottom_row_1x4), dim=0)
        return transform_matrix

    def permute_rotation(self, gt_transform):
        """
        Applies a permutation rotation to change the bases of the given ground truth transformation matrix
         (derived from SOI angles, see `src/Rotations`) to match the
         pytorch tensor convention and converts it to a 4x4 matrix.

        Args:
            gt_transform (torch.Tensor): A tensor representing the ground truth transformation matrix.
                This tensor is expected to be at least 3-dimensional.

        Returns:
            torch.Tensor: A 4x4 transformation matrix after applying the permutation and the build_4x4_transform_mat process.
        """
        gt_transform_test = gt_transform.squeeze().unsqueeze(0)
        perm = torch.tensor([2, 1, 0], dtype=torch.long)
        gt_transform_perm = gt_transform_test[:, :, perm]
        gt_transform_2xperm = gt_transform_perm[:, perm, :] 
        gt_transform_2xperm_transpose = gt_transform_2xperm.transpose(1, 2).squeeze()
        gt_transform_2xperm_transpose_4x4 = self.build_4x4_transform_mat(gt_transform_2xperm_transpose,
                                                                         translation=None)
        return gt_transform_2xperm_transpose_4x4

    def rotate_volume(self, protein_volume, R):
        """
        Rotate a grid image using 3D rotation matrix, centering before rotation by subtracting translations.

        :param protein_volume: input grid image
        :param R: 1x4x4 transformation matrix to a 1x3x4 used in `affine_grid()` with
         the translation column replaced with [0,0,0,1]
        :return: rotated grid image
        """
        N = R.shape[0]
        # print('R.shape',R.shape)
        R_Nx3x3 = R[:, :3, :3]
        # print("R_1x3x3.shape", R_1x3x3.shape)
        # print("R_1x3x3", R_1x3x3)
        R_Nx3x4 = torch.cat((R_Nx3x3, torch.zeros(N, 3, 1).to(device=self.device, dtype=self.dtype)), dim=2)
        # print("R_1x3x4.shape", R_1x3x4.shape)
        # print("R_1x3x4", R_1x3x4)
        curr_grid = F.affine_grid(R_Nx3x4, size=protein_volume.shape, align_corners=True)
        return F.grid_sample(protein_volume, curr_grid, align_corners=True)
