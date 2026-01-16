import copy

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from Bio.PDB import *
from scipy import ndimage

from src.UtilityFunctions import UtilityFunctions

class ProcessCoords:

    def __init__(self, device, dtype, dim, resolution_in_angstroms=1.0, sigma=1.0, normalize_single_atom_kernel=False):
        self.device = device
        self.dtype = dtype
        self.dim = dim
        self.resolution_in_angstroms = resolution_in_angstroms
        self.sigma = sigma
        self.normalize_single_atom_kernel = normalize_single_atom_kernel

        self.U = UtilityFunctions(self.device, dtype=self.dtype)

    def get_coordinates(self, filename):
        '''
        Grab C, N, O, S coordinates and return as list of lists.
        :param filename: pdb file to open
        :return: dictionary of dataframes of selected atom type coordinates
        '''
        atom_types = ['C', 'N', 'O', 'S']
        C_coords = []
        N_coords = []
        O_coords = []
        S_coords = []

        p = PDBParser(PERMISSIVE=1, QUIET=True)
        # for filename in filenames:
        structure = p.get_structure('name', filename)
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.id[0] == ' ': #only std amino acids
                        for atom in residue:
                            if atom.name[0] == 'C':
                                C_coords.append(list(atom.get_coord()))
                            elif atom.name[0] == 'N':
                                N_coords.append(list(atom.get_coord()))
                            elif atom.name[0] == 'O':
                                O_coords.append(list(atom.get_coord()))
                            elif atom.name[0] == 'S':
                                S_coords.append(list(atom.get_coord()))
        del structure

        CNOS_coords = [C_coords, N_coords, O_coords, S_coords]
        df_CNOS_coords = pd.DataFrame(CNOS_coords).transpose()
        df_CNOS_coords.columns = atom_types
        dict_df_coords = {}
        for i in range(len(atom_types)):
            atom_name = atom_types[i]
            df = pd.DataFrame(df_CNOS_coords[atom_name].dropna().tolist(), columns=['x', 'y', 'z'])
            # print('df before', df)
            df = df/self.resolution_in_angstroms
            # print('df after', df)
            df.name = atom_types[i]
            dict_df_coords[df.name] = df

        return dict_df_coords

    def transform_coords(self, dict_df_CNOS_coords, transformation_mat):
        input_coords_dict = copy.deepcopy(dict_df_CNOS_coords)
        if transformation_mat is not None:
            for key in input_coords_dict:
                for index, row in input_coords_dict[key].iterrows():
                    x, y, z, dummy = row['x'], row['y'], row['z'], 1.0

                    vector = torch.tensor([[x, y, z, dummy]]).to(device=self.device, dtype=self.dtype)
                    vector_rot = torch.matmul(transformation_mat, vector.t()).t()
                    outx, outy, outz = vector_rot[0][0].item(), vector_rot[0][1].item(), vector_rot[0][2].item()

                    row['x'], row['y'], row['z'] = outx, outy, outz

        return input_coords_dict

    @staticmethod
    def center_of_mass(X, Y, Z):
        x_com = np.mean(X)
        y_com = np.mean(Y)
        z_com = np.mean(Z)

        return x_com, y_com, z_com

    def get_centering_translation_matrix(self, dict_df_CNOS_coords, desired_center=(0.0, 0.0, 0.0),
                                         get_center_of_mass=False):
        if len(desired_center) == 1:
            center = [desired_center, desired_center, desired_center]
        else:
            center = desired_center

        coords_df = dict_df_CNOS_coords['C']
        x_com, y_com, z_com = self.center_of_mass(coords_df['x'], coords_df['y'], coords_df['z'])

        x_shift, y_shift, z_shift = (center[0] - x_com), \
                                    (center[1] - y_com), \
                                    (center[2] - z_com)
        translation = torch.tensor([[x_shift], [y_shift], [z_shift]])

        transform_matrix = self.U.build_4x4_transform_mat(translation=translation)

        if get_center_of_mass:
            return translation
        else:
            return transform_matrix

    def coords_to_volume_rounded(self, dict_df_CNOS_coords):
        box_dim = [self.dim] * 3
        volume = np.zeros(box_dim)
        for key in dict_df_CNOS_coords:
            for index, row in dict_df_CNOS_coords[key].iterrows():
                # print(row['x'], row['y'], row['z'])
                x, y, z = int(np.round(row['x']) % self.dim), \
                          int(np.round(row['y']) % self.dim), \
                          int(np.round(row['z']) % self.dim)
                volume[x, y, z] = 1.0

        volume = ndimage.gaussian_filter(volume, sigma=self.sigma, truncate=3)

        return volume

    def coords_to_volume(self, dict_df_CNOS_coords):
        box_dim = [self.dim] * 3
        volume = torch.zeros(box_dim)

        # scale the spread of gaussian based on grid resolution
        # num_std_devs = 2/self.resolution_in_angstroms

        # create a grid of indices
        x_vals = torch.arange(self.dim)
        y_vals = torch.arange(self.dim)
        z_vals = torch.arange(self.dim)
        xx, yy, zz = torch.meshgrid(x_vals, y_vals, z_vals, indexing='ij')

        sigma_squared = self.sigma ** 2
        for key in dict_df_CNOS_coords:
            for index, row in dict_df_CNOS_coords[key].iterrows():
                # retrieve the atomic coordinates from current dataframe
                x, y, z = row['x'], row['y'], row['z']
                # evaluate the Gaussian function at each point on the grid
                distance_squared = (xx - x) ** 2 + (yy - y) ** 2 + (zz - z) ** 2
                kernel = torch.exp(-distance_squared / sigma_squared)

                # Normalize the kernel
                if self.normalize_single_atom_kernel:
                    kernel_sum = torch.sum(kernel)
                    if kernel_sum > 0:
                        torch.divide(kernel, kernel_sum, out=kernel)

                # add the kernel to the 3D array
                volume += kernel

        return volume

    def plot_CNOS_coords_scatter(self, dict_coords,
                                 colors=None,
                                 opacities=None,
                                 plot_complex=False):

        if colors is None:
            colors = {'C': 'black', 'N': 'blue', 'O': 'red', 'S': 'yellow'}

        if opacities is None:
            opacities = {'C': 0.5, 'N': 0.5, 'O': 0.5, 'S': 0.5}

        figure_list = []
        for key in dict_coords:
            figure_list.append(
                go.Scatter3d(
                    x=dict_coords[key]['x'],
                    y=dict_coords[key]['y'],
                    z=dict_coords[key]['z'],
                    marker=go.scatter3d.Marker(size=10, color=colors[key]),
                    opacity=opacities[key],
                    mode='markers',
                    name=key,
                )
            )
        fig = go.Figure(data=figure_list)
        if not plot_complex:
            fig.show()
        else:
            return figure_list

    def plot_complex_scatter(self, list_of_dicts,
                             colors=None,
                             opacities=None,
                             show=True):
        list_of_figure_lists = []
        for d in range(len(list_of_dicts)):
            list_of_figure_lists.append(self.plot_CNOS_coords_scatter(list_of_dicts[d], colors=colors[d%len(colors)],
                                                                      opacities=opacities[d%len(opacities)],
                                                                      plot_complex=True))

        joined_list = []
        for i in range(len(list_of_figure_lists)):
            joined_list += list_of_figure_lists[i]

        invisible_scale = go.Scatter3d(
            name="",
            visible=True,
            showlegend=False,
            opacity=0,
            hoverinfo='none',
            x=[0, self.dim],
            y=[0, self.dim],
            z=[0, self.dim]
        )
        joined_list.append(invisible_scale)

        fig = go.Figure(data=joined_list,
                        # layout=layout,
                        )

        if show:
            fig.show()

        return fig

    def plot_volume(self, volume,
                    colorscale='Picnic_r',
                    label=None,
                    surface_count=3,
                    isomin=None,
                    isomax=None,
                    midpoint=0.0,
                    opacity=0.25,
                    plot_complex=False):

        box_dim = volume.shape
        X, Y, Z = np.mgrid[:box_dim[0], :box_dim[0], :box_dim[0]]
        figure = [
        go.Volume(
            x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
            value=volume.flatten(),
            colorscale=colorscale,
            isomin=isomin,
            isomax=isomax,
            opacity=opacity,
            surface_count=surface_count,
            name=label,
            # legendgrouptitle=label,
            showlegend=True,
            colorbar={'orientation':'h'},
            cmin=isomin,
            cmid=midpoint,
            cmax=isomax,
            # hoverinfo='skip',
            hoverinfo=None,

        )
        ]

        if not plot_complex:
            fig = go.Figure(data=figure)
            fig.show()
        else:
            return figure[0]

    def plot_complex_volume(self, list_of_volumes,
                            labels=[None],
                            isomins=[0.01],
                            colorscale=['RdBu_r', 'RdBu'],
                            opacities=[0.25],
                            surface_counts=[2],
                            midpoints=[0.0],
                            show=True,
                            save_fig=False,
                            save_name='complex_volume',):

        combined_plot = []
        # names = ['antibody', 'antigen']
        # colorscale = ['blackbody', 'Spectral', 'RdBu_r']
        for v in range(len(list_of_volumes)):
            combined_plot.append(self.plot_volume(list_of_volumes[v],
                                                  label=labels[v%len(labels)],
                                                  surface_count=surface_counts[v%len(surface_counts)],
                                                  colorscale=colorscale[v%len(colorscale)], isomin=isomins[v%len(isomins)],
                                                  opacity=opacities[v%len(opacities)], midpoint=midpoints[v%len(midpoints)],
                                                  plot_complex=True
                                                  )
                                 )
        fig = go.Figure(data=combined_plot)
        # update_layout method used to modify change and size
        fig.update_layout(legend=dict(title_font_family="Ariel",
                                      font=dict(size=36)
                                      ))
        if save_fig:
            save_html_path = f'{save_name}.html'
            fig.write_html(save_html_path)

        if show:
            fig.show()

        return fig

    def combine_coords(self, dict_original_coords_antibody, dict_original_coords_antigen):
        combined_dict = {}
        for key in dict_original_coords_antibody:
            combined_key = pd.concat([dict_original_coords_antibody[key],dict_original_coords_antigen[key]], keys=["x", "y", "z"])
            combined_dict[key] = combined_key

        return combined_dict

    def translate_coords_to_center(self, dict_coords, desired_center, debug=False):
        dict_coords_in = copy.deepcopy(dict_coords)
        centering_t = self.get_centering_translation_matrix(dict_coords_in, desired_center=desired_center)
        dict_centered_coords = self.transform_coords(dict_coords_in, centering_t)
        if debug:
            self.plot_complex_scatter([dict_centered_coords])

        return dict_centered_coords, centering_t
