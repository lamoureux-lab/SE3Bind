import os
import torch
from math import *
import random
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from src.UtilityFunctions import UtilityFunctions, Path


class Rotations(object):
    '''
    Build an SO(3) angular grid for sampling equivariant 3D rotations, using precomputed rotation matrices (SOI dataset) from this work.
    Yershova, Anna, Steven M. LaValle, and Julie C. Mitchell.
     "Generating uniform incremental grids on SO (3) using the Hopf fibration."
      Algorithmic Foundation of Robotics VIII: Selected Contributions of the Eight International Workshop
      on the Algorithmic Foundations of Robotics. Springer Berlin Heidelberg, 2010.
      https://mitchell-lab.biochem.wisc.edu/SOI/index.php
    '''
    def __init__(self, device, dtype, angle_inc=20):
        '''
        Read in a file containing rotation matrices at a specified angular grid resolution.
        See gridpoints_count.txt for total angle counts per oim*.eul file.

        Args:
            device: set device torch tensors
            dtype: set dtype torch tensors
            angle_inc: SOI angle increment matching SOI/rotation_samples/oim*.eul
        '''

        self.device = device
        self.dtype = dtype

        self.path_to_SOI = '/data/SOI/rotation_samples/'
        self.angle_inc = angle_inc
        self.rotation_mats = self.loadSOI(self.angle_inc).to(device=self.device)
        print('self.rotation_mats.dtype', self.rotation_mats.dtype)
        self.num_angles = self.rotation_mats.shape[0]
        print("Angle increment:", self.angle_inc)
        print("Number of rotations:", self.num_angles)

        self.identity_matrix = torch.eye(3)

    def write_matrix(self, i, phi, theta, psi):
        cpsi = cos(psi)
        spsi = sin(psi)
        ctheta = cos(theta)
        stheta = sin(theta)
        cphi = cos(phi)
        sphi = sin(phi)
        self.rotation_mats[i, 0, 0] = cpsi * cphi - spsi * ctheta * sphi
        self.rotation_mats[i, 0, 1] = -cpsi * sphi - spsi * ctheta * cphi
        self.rotation_mats[i, 0, 2] = spsi * stheta

        self.rotation_mats[i, 1, 0] = spsi * cphi + cpsi * ctheta * sphi
        self.rotation_mats[i, 1, 1] = -spsi * sphi + cpsi * ctheta * cphi
        self.rotation_mats[i, 1, 2] = -cpsi * stheta

        self.rotation_mats[i, 2, 0] = stheta * sphi
        self.rotation_mats[i, 2, 1] = stheta * cphi
        self.rotation_mats[i, 2, 2] = ctheta

    def get_matrix(self, phi, theta, psi):
        R = torch.zeros(3, 3, dtype=self.dtype)
        cpsi = cos(psi)
        spsi = sin(psi)
        ctheta = cos(theta)
        stheta = sin(theta)
        cphi = cos(phi)
        sphi = sin(phi)
        R[0, 0] = cpsi * cphi - spsi * ctheta * sphi
        R[0, 1] = -cpsi * sphi - spsi * ctheta * cphi
        R[0, 2] = spsi * stheta

        R[1, 0] = spsi * cphi + cpsi * ctheta * sphi
        R[1, 1] = -spsi * sphi + cpsi * ctheta * cphi
        R[1, 2] = -cpsi * stheta

        R[2, 0] = stheta * sphi
        R[2, 1] = stheta * cphi
        R[2, 2] = ctheta
        return R

    def loadSOI(self, angle_inc):
        """
        Loading SOI samples from SO(3) group:
        https://mitchell-lab.biochem.wisc.edu/SOI/index.php
        The oim*.eul files contain precomputed 3D Euler angular vectors [phi, theta, psi] at different angular resolutions.
        """
        current_dir = Path(__file__).parent
        soi_dir = current_dir / '..' / 'data' / 'SOI' / 'rotation_samples'
        filename = soi_dir / f'oim{angle_inc}.eul'

        if not filename.exists():
            raise Exception("Can't find rotation angles:", str(filename))
        if angle_inc == 20:
            N = 1854
        elif angle_inc == 15:
            N = 4392
        elif angle_inc == 12:
            N = 8580
        elif angle_inc == 10:
            N = 14868
        elif angle_inc == 8:
            N = 29025
        elif angle_inc == 6:
            N = 68760
        elif angle_inc == 4:
            N = 232020
        else:
            raise Exception("Can't find rotation angles:", angle_inc)

        self.rotation_mats = torch.zeros(N, 3, 3, dtype=self.dtype)
        self.angles = torch.zeros(N, 3, dtype=self.dtype)
        index = 0
        print('Opening file', filename, '...')
        with open(filename, 'r') as fin:
            for line in fin:
                sline = line.split()
                phi = float(sline[0])
                theta = float(sline[1])
                psi = float(sline[2])
                # self.write_matrix(index, phi, theta, psi)
                self.rotation_mats[index, :, :] = self.get_matrix(phi, theta, psi)
                self.angles[index, 0] = phi
                self.angles[index, 1] = theta
                self.angles[index, 2] = psi
                index += 1

        return self.rotation_mats

    def random_rotation(self):
        '''

        Returns: A random rotation from set of rotation matrices

        '''
        idx = random.randint(0, self.num_angles-1)
        return self.rotation_mats[idx, :, :]

    def build_angular_grid(self, adj_threshold, show_thirds_of_trace=False):
        '''
        Build angular grid using adjacency value [-0.33, 1.0), using the sum of the diagonal (trace) of the R_i \matmul inverse(R_j), divided by 3.
        If the product of R_i and R_j is the identity matrix, the rotations are identical and the trace sums to 3.

        Args:
            adj_threshold: adjacency value [-0.33, 1.0) defines similarity.
            show_thirds_of_trace: Build a list of all similarity scores encountered for downstream metrics/plotting

        Returns:
            so3_adjacency_dict: dictionary of rotation matrix indices that point to rotation matrices satisfying adj_threshold
            third_of_trace_list: list of all thirds_of_trace values computed

        '''
        adj_threshold = torch.tensor(adj_threshold)
        so3_adjacency_dict = defaultdict(dict)
        third_of_trace_list = []
        for i in tqdm(range(self.num_angles)):
            vertex = i
            R_i = self.rotation_mats[i, :, :]
            so3_adjacency_dict[vertex].update({i: i})
            for j in range(self.num_angles):
                if i != j:
                    R_j = self.rotation_mats[j, :, :]
                    third_of_trace = self.third_of_trace_similarity(R_i, R_j)
                    if show_thirds_of_trace:
                        third_of_trace_list.append(third_of_trace)
                    if third_of_trace > adj_threshold:
                        so3_adjacency_dict[vertex].update({j: j})

        return so3_adjacency_dict, third_of_trace_list

    @staticmethod
    def third_of_trace_similarity(R_i, R_j):
        '''
        Calulate similarity of two rotation matrices.
        Args:
            R_i: 3x3 rotation matrix
            R_j: 3x3 rotation matrix

        Returns:
            third_of_trace: similarity score [-0.33, 1.0)
        '''
        inverse_transform = torch.matmul(R_i, torch.pinverse(R_j))
        third_of_trace = torch.div(torch.trace(inverse_transform), 3)

        return third_of_trace

    @staticmethod
    def get_N_neighborhoods(initial_neighborhood_indices, so3_adjacency_dict, num_neighborhoods=1):
        '''
        Build multiple neighborhoods of rotations for sampling beyond the first sets of closest rotations.
        Args:
            initial_neighborhood_indices: initial neighborhoods built on angular grid satisfying `adj_threshold`
            so3_adjacency_dict: prebuilt angular grid of rotation matrix indices from `build_angular_grid()`
            num_neighborhoods: Number of neighborhoods specified, including the first neighborhood set `initial_neighborhood_indices`

        Returns:
            multi_neighborhood: multiple neighborhoods specified in `num_neighborhoods`
        '''

        if num_neighborhoods == 1:
            return initial_neighborhood_indices
        else:
            multi_neighborhood = {}
            neighborhood_next = initial_neighborhood_indices
            # multi_neighborhood = initial_neighborhood_indices
            for i in range(num_neighborhoods-1):
                # print('multi_neighborhood', multi_neighborhood)
                # neighborhood_next = multi_neighborhood

                for key in list(neighborhood_next.keys()):
                    neighborhood_next = so3_adjacency_dict[key]
                    for key in list(neighborhood_next.keys()):
                        multi_neighborhood[key] = key
                neighborhood_next = multi_neighborhood

            # print('final', multi_neighborhood)
            # print(len(multi_neighborhood))

            return multi_neighborhood


def check_sample_loop(so3_adjacency_dict, rot_mats_list, num_samples=100, print_outs=False):
    '''
    Validate the angular grid traversal for rotation sampling using a random walk.
    Args:
        so3_adjacency_dict: prebuilt angular grid of rotation matrix indices from `build_angular_grid()`
        rot_mats_list: list of 3x3 rotation matrices to pull from
        num_samples: number of rotations for pull
        print_outs: print traversal

    Returns:
        rot_indices_sampled: rotation indices samples from `so3_adjacency_dict`
        sampled_rot_mats: rotation matrices corresponding to sampled
        indices from the random walk through `so3_adjacency_dict`

    '''
    cur_index = random.choices(list(so3_adjacency_dict.keys()), k=1)[0]
    next_index = random.choices(list(so3_adjacency_dict[cur_index])[1:], k=1)[0]
    # rot_current = rot_mats_list[cur_index]
    # rot_next = rot_mats_list[next_index]
    # print('initializing random rotation')
    # print('start', cur_index)
    # print('rot current', rot_current)
    # print('next', next_index)
    # print('rot next', rot_next)

    print('running sampling loop')
    rot_indices_sampled = []
    sampled_rot_mats = []
    for i in range(num_samples):
        if i > 0:
            cur_index = next_index
        next_index = random.choices(list(so3_adjacency_dict[cur_index])[1:], k=1)[0]

        rot_current = rot_mats_list[next_index]
        rot_indices_sampled.append(next_index)
        sampled_rot_mats.append(rot_current)
    # #
    if print_outs:
        print('rot_indices_sampled', rot_indices_sampled)
        print('rotations sampled', sampled_rot_mats)

    return rot_indices_sampled, sampled_rot_mats


def avg_neighbor_count(so3_adjacency_dict, print_outs=False):
    '''
    Check the neighborhood sizes of a prebuilt `so3_adjacency_dict`
    Args:
        so3_adjacency_dict: prebuilt angular grid of rotation matrix indices from `build_angular_grid()`
        print_outs: print metrics

    Returns:
        neighbor_count: average number of neighbors in the provided `so3_adjacency_dict`
    '''
    neighbor_count = []
    print_count = 0
    print_limit = 2000
    for key, value in so3_adjacency_dict.items():
        if print_count < print_limit:
            print(key, ' : ', value)
        print_count += 1

        neighbor_count.append(len(value)-1)

    if print_outs:
        print(neighbor_count)
        print('average num neighbors')
        print(sum(neighbor_count) / len(neighbor_count))

    return neighbor_count


def plot_plt_original_angular_grid(rot_mats):
    '''
    Plot the angular grid using matplotlib
    Args:
        rot_mats: rotation matrices
    Returns:
        plots the "vector" given the rotation matrix
    '''

    identity_mat = torch.eye(3)
    number_of_points = 3
    for i in range(number_of_points):
        coords = torch.matmul(identity_mat[:][i], rot_mats)
        coords = torch.transpose(coords.squeeze(), dim0=0, dim1=1).detach().cpu()

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(projection='3d')
        ax.scatter(coords[0][:], coords[1][:], coords[2][:])

        plt.show()


def plot_plotly_original_angular_grid(rot_mats, sampled_rot_mats, save_fig=True, save_path=None):
    '''
    Plot the angular grid and the a sampled random walk of rotations, using plotly.
    Args:
        rot_mats: rotation matrices
        sampled_rot_mats: rotation matrices sampled

    Returns:
        None
    '''

    identity_mat = torch.eye(3)
    sampled_rot_mats = torch.concat((sampled_rot_mats), dim=0)
    columns = ['x', 'y', 'z']
    df_sampled_rot_mats = pd.DataFrame(sampled_rot_mats.numpy(), columns=columns)

    # df_all_rot_mats = pd.DataFrame()
    df_all_rotsX = pd.DataFrame()
    df_all_rotsY = pd.DataFrame()
    df_all_rotsZ = pd.DataFrame()

    for i in range(3):
        coords = torch.matmul(identity_mat[:][i], rot_mats)
        if i == 0:
            df_all_rotsX = pd.DataFrame(coords, columns=columns)
        if i == 1:
            df_all_rotsY = pd.DataFrame(coords, columns=columns)
        if i == 2:
            df_all_rotsZ = pd.DataFrame(coords, columns=columns)
        # print(coords)
        # all_coords.append(coords)
        # cur_df = pd.DataFrame(coords, columns=['x', 'y', 'z'])
        # df_all_rot_mats = pd.DataFrame.append(df_all_rot_mats, cur_df)
        # print(df_all_rot_mats)

    combined_plot = [
        go.Scatter3d(
            x=df_sampled_rot_mats['x'], y=df_sampled_rot_mats['y'], z=df_sampled_rot_mats['z'],
            marker=dict(
                size=6,
                color=df_sampled_rot_mats['z'],
                # colorscale='Viridis',
            ),
            line=dict(
                color='darkblue',
                width=2
            )
        ),
        go.Scatter3d(
            x=df_all_rotsX['x'],
            y=df_all_rotsX['y'],
            z=df_all_rotsX['z'],
            marker=go.scatter3d.Marker(size=3),
            opacity=0.33,
            mode='markers'
        ),
        go.Scatter3d(
            x=df_all_rotsY['x'],
            y=df_all_rotsY['y'],
            z=df_all_rotsY['z'],
            marker=go.scatter3d.Marker(size=3),
            opacity=0.33,
            mode='markers'
        ),
        go.Scatter3d(
            x=df_all_rotsZ['x'],
            y=df_all_rotsZ['y'],
            z=df_all_rotsZ['z'],
            marker=go.scatter3d.Marker(size=3),
            opacity=0.33,
            mode='markers'
        ),
    ]
    fig = go.Figure(data=combined_plot)
    fig.show()

    if save_fig:
        fig.write_html(save_path)

def plotly_hist_third_of_trace(third_of_trace_list, binwidth=0.01):
    df_third_of_trace = pd.DataFrame(third_of_trace_list, columns=['traces'])
    fig = px.histogram(x=df_third_of_trace['traces'])
    fig.update_layout(
        xaxis=dict(
            tickmode='linear',
            dtick=binwidth
        ),
        yaxis_range=[0, 10000]
    )

    fig.show()


if __name__ == "__main__":

    print_outs = False
    show_thirds_of_trace = False
    third_of_trace_list = None
    show_plots = True

    device = 'cuda'
    dtype = torch.float32

    if show_plots:
        import plotly.graph_objects as go
        import plotly.express as px

    data_path = '../data/saved_angular_grids/'
    angle_inc = 20
    # angle_inc = 15 # current accepted neighbor count 14.26 avg
    adj_threshold = 0.95  # current accepted neighbor count 4.65 avg

    Rs = Rotations(angle_inc=angle_inc, device=device, dtype=dtype)
    print('\nrandom rotation test\n', Rs.random_rotation())
    U = UtilityFunctions(device=device, dtype=dtype)

    ### check for existing anglular grid .pkl file
    param_string = data_path+'so3_angular_grid_threshold' + str(adj_threshold) + 'angle_inc' + str(angle_inc)
    pkl_dict = param_string +'.pkl'
    trace_pkl = param_string + '_traces.pkl'
    if not os.path.exists(pkl_dict):
        print('adjacency dict', pkl_dict, 'does not exist, regenerating...')
        so3_adjacency_dict, third_of_trace_list = Rs.build_angular_grid(adj_threshold=adj_threshold, show_thirds_of_trace=show_thirds_of_trace)
        print('writing', pkl_dict)
        U.write_pkl(so3_adjacency_dict, pkl_dict)
        # with open(pkl_dict, 'wb') as handle:
        #     pkl.dump(so3_adjacency_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)
        # if show_thirds_of_trace:
        #     print('writing', trace_pkl)
        #     with open(trace_pkl, 'wb') as handle:
        #         pkl.dump(third_of_trace_list, handle, protocol=pkl.HIGHEST_PROTOCOL)
    else:
        so3_adjacency_dict = U.read_pkl(pkl_dict)
        # print('loading', pkl_dict)
        # with open(pkl_dict, 'rb') as handle:
        #     so3_adjacency_dict = pkl.load(handle)
        # if show_thirds_of_trace:
        #     print('loading', trace_pkl)
        #     with open(trace_pkl, 'rb') as handle:
        #         third_of_trace_list = pkl.load(handle)

    ## get average neighbor count of grid.
    neighbor_count = avg_neighbor_count(so3_adjacency_dict, print_outs=print_outs)
    print(sum(neighbor_count)/len(neighbor_count))
    plt.close()
    plt.hist(neighbor_count, bins=21)
    title = pkl_dict.split('/')[-1][:-3]
    print(title)
    plt.title(title)
    plt.xlabel('neighbor_count')
    plt.xlim([0,20])
    plt.xticks(np.arange(0,20,1))
    plt.savefig('Figs/'+title)
    # plt.show()


    ## load all rotation matrices
    rot_mats = Rs.loadSOI(angle_inc=angle_inc)
    rot_mats_list = list(rot_mats)

    ## random walk on nested adjacency dict
    print('doing random-walk sampling')
    num_samples = 10
    rot_indices_sampled, sampled_rot_mats = check_sample_loop(so3_adjacency_dict, rot_mats_list, num_samples=num_samples,
                                                               print_outs=print_outs)
    print(f'rot_indices_sampled\n {rot_indices_sampled}')

    if show_plots:
        if show_thirds_of_trace:
            print('plotting all thirds of traces')
            plotly_hist_third_of_trace(third_of_trace_list)

        print('plotting angular grid and', num_samples, 'random-walk sampled rotations')
        plot_plotly_original_angular_grid(rot_mats, sampled_rot_mats, save_fig=True,
                                          save_path=param_string+'random_walk_trace.html')

    # print('plotting matplotlib angular grid')
    # plot_plt_original_angular_grid(rot_mats)

    # random_index = np.random.randint(0, len(so3_adjacency_dict))
    num_neighborhoods = 3
    multi_neighborhood = Rotations.get_N_neighborhoods(so3_adjacency_dict[0], so3_adjacency_dict, num_neighborhoods=num_neighborhoods)
    print(f'Multiple neighborhoods combined for neighborhood search:'
          f'\n num_neighborhoods: {num_neighborhoods}'
          f'\n total neighbors: {len(multi_neighborhood)}'
          f'\n multi_neighborhood: indices{multi_neighborhood}')
