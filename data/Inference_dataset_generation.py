import os
import sys
from itertools import chain
from os.path import exists

# Add parent directory to path to import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from Bio.PDB import PDBParser, PDBIO, Select
from tqdm import tqdm

from src.ProcessCoords import ProcessCoords
from src.UtilityFunctions import *


def build_bounding_box(dict_df_cofactor_coords):
    """
    Create a bounding box around molecules.

    :param dict_df_cofactor_coords: Dictionary of dataframe of the cofactor only
    :return: 1)  list: Center coordinate of the cofactor bounding box. 2) list: max/min of bounding box, [x_min, x_max, y_min, y_max, z_min, z_max]
    """

    combined_dataframe = pd.DataFrame(columns=['x', 'y', 'z'])

    for k in dict_df_cofactor_coords:
        combined_dataframe = pd.concat([combined_dataframe, dict_df_cofactor_coords[k]], axis=0, ignore_index=True)

    x_max, y_max, z_max = combined_dataframe.max()
    x_min, y_min, z_min = combined_dataframe.min()

    ### Cofactor bb box edges
    vertices = [
        (x_min, y_min, z_min),
        (x_min, y_min, z_max),
        (x_min, y_max, z_min),
        (x_min, y_max, z_max),
        (x_max, y_min, z_min),
        (x_max, y_min, z_max),
        (x_max, y_max, z_min),
        (x_max, y_max, z_max)
    ]

    center_x = sum([v[0] for v in vertices]) / len(vertices)
    center_y = sum([v[1] for v in vertices]) / len(vertices)
    center_z = sum([v[2] for v in vertices]) / len(vertices)
    center_coords = [center_x, center_y, center_z]
    bb_min_max = [x_min, x_max, y_min, y_max, z_min, z_max]

    return center_coords, bb_min_max


def find_overlapping_box(bb1_min_max, bb2_min_max):
    """
    Find the overlapping bounding box and its center given two bounding boxes.

    :param bb1_min_max: list of min/max coordinates for the first bounding box [x_min, x_max, y_min, y_max, z_min, z_max]
    :param bb2_min_max: list of min/max coordinates for the second bounding box [x_min, x_max, y_min, y_max, z_min, z_max]
    :return: 1) list: Center coordinates of the overlapping bounding box. 2) list: min/max of the overlapping bounding box
    """

    x_min_overlap = max(bb1_min_max[0], bb2_min_max[0])
    x_max_overlap = min(bb1_min_max[1], bb2_min_max[1])
    y_min_overlap = max(bb1_min_max[2], bb2_min_max[2])
    y_max_overlap = min(bb1_min_max[3], bb2_min_max[3])
    z_min_overlap = max(bb1_min_max[4], bb2_min_max[4])
    z_max_overlap = min(bb1_min_max[5], bb2_min_max[5])

    # Check if there is an actual overlap
    if x_min_overlap <= x_max_overlap and y_min_overlap <= y_max_overlap and z_min_overlap <= z_max_overlap:
        overlap_min_max = [x_min_overlap, x_max_overlap, y_min_overlap, y_max_overlap, z_min_overlap, z_max_overlap]
        center_x = (x_min_overlap + x_max_overlap) / 2
        center_y = (y_min_overlap + y_max_overlap) / 2
        center_z = (z_min_overlap + z_max_overlap) / 2
        center_coords = [center_x, center_y, center_z]
        return center_coords, overlap_min_max
    else:
        return None, None


def plot_bounding_boxes(bb1_vertices, bb2_vertices, overlap_vertices, center_coords_1, center_coords_2,
                        overlap_center_coords):
    # Create a scatter3d trace for the vertices of the first bounding box
    bb1_trace = go.Scatter3d(
        x=[v[0] for v in bb1_vertices],
        y=[v[1] for v in bb1_vertices],
        z=[v[2] for v in bb1_vertices],
        mode='markers',
        marker=dict(
            size=5,
            color='blue',
        ),
        name='Bounding Box 1'
    )

    # Create a scatter3d trace for the vertices of the second bounding box
    bb2_trace = go.Scatter3d(
        x=[v[0] for v in bb2_vertices],
        y=[v[1] for v in bb2_vertices],
        z=[v[2] for v in bb2_vertices],
        mode='markers',
        marker=dict(
            size=5,
            color='green',
        ),
        name='Bounding Box 2'
    )

    # Create a scatter3d trace for the vertices of the overlapping bounding box
    overlap_trace = go.Scatter3d(
        x=[v[0] for v in overlap_vertices],
        y=[v[1] for v in overlap_vertices],
        z=[v[2] for v in overlap_vertices],
        mode='markers',
        marker=dict(
            size=5,
            color='red',
        ),
        name='Overlapping Box'
    )

    # Create a scatter3d trace for the center points
    center_trace = go.Scatter3d(
        x=[center_coords_1[0], center_coords_2[0], overlap_center_coords[0] if overlap_center_coords else None],
        y=[center_coords_1[1], center_coords_2[1], overlap_center_coords[1] if overlap_center_coords else None],
        z=[center_coords_1[2], center_coords_2[2], overlap_center_coords[2] if overlap_center_coords else None],
        mode='markers',
        marker=dict(
            size=10,
            color=['blue', 'green', 'red'],
        ),
        name='Centers'
    )

    # Create the layout for the plot
    layout = go.Layout(
        title='Bounding Boxes and Overlapping Box',
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z')
        )
    )

    fig = go.Figure(data=[bb1_trace, bb2_trace, overlap_trace, center_trace], layout=layout)

    return fig


def check_bounding_box_overlaps(bb_min_max_1, bb_min_max_2, overlap_bb_min_max, center_coords_1, center_coords_2,  overlap_center_coords):
    # Get vertices for the original and overlapping bounding boxes
    bb1_vertices = [
        (bb_min_max_1[0], bb_min_max_1[2], bb_min_max_1[4]),
        (bb_min_max_1[0], bb_min_max_1[2], bb_min_max_1[5]),
        (bb_min_max_1[0], bb_min_max_1[3], bb_min_max_1[4]),
        (bb_min_max_1[0], bb_min_max_1[3], bb_min_max_1[5]),
        (bb_min_max_1[1], bb_min_max_1[2], bb_min_max_1[4]),
        (bb_min_max_1[1], bb_min_max_1[2], bb_min_max_1[5]),
        (bb_min_max_1[1], bb_min_max_1[3], bb_min_max_1[4]),
        (bb_min_max_1[1], bb_min_max_1[3], bb_min_max_1[5])
    ]

    bb2_vertices = [
        (bb_min_max_2[0], bb_min_max_2[2], bb_min_max_2[4]),
        (bb_min_max_2[0], bb_min_max_2[2], bb_min_max_2[5]),
        (bb_min_max_2[0], bb_min_max_2[3], bb_min_max_2[4]),
        (bb_min_max_2[0], bb_min_max_2[3], bb_min_max_2[5]),
        (bb_min_max_2[1], bb_min_max_2[2], bb_min_max_2[4]),
        (bb_min_max_2[1], bb_min_max_2[2], bb_min_max_2[5]),
        (bb_min_max_2[1], bb_min_max_2[3], bb_min_max_2[4]),
        (bb_min_max_2[1], bb_min_max_2[3], bb_min_max_2[5])
    ]

    if overlap_bb_min_max:
        overlap_vertices = [
            (overlap_bb_min_max[0], overlap_bb_min_max[2], overlap_bb_min_max[4]),
            (overlap_bb_min_max[0], overlap_bb_min_max[2], overlap_bb_min_max[5]),
            (overlap_bb_min_max[0], overlap_bb_min_max[3], overlap_bb_min_max[4]),
            (overlap_bb_min_max[0], overlap_bb_min_max[3], overlap_bb_min_max[5]),
            (overlap_bb_min_max[1], overlap_bb_min_max[2], overlap_bb_min_max[4]),
            (overlap_bb_min_max[1], overlap_bb_min_max[2], overlap_bb_min_max[5]),
            (overlap_bb_min_max[1], overlap_bb_min_max[3], overlap_bb_min_max[4]),
            (overlap_bb_min_max[1], overlap_bb_min_max[3], overlap_bb_min_max[5])
        ]
    else:
        overlap_vertices = []

    # Plot the bounding boxes and overlapping box
    bounding_boxes_fig = plot_bounding_boxes(bb1_vertices, bb2_vertices, overlap_vertices, center_coords_1, center_coords_2,
                        overlap_center_coords)

    return bounding_boxes_fig


def split_pdb_files(pdb_path: str, chain_ids: list):
    """Given a PDB with complex, extract chains for Receptor and Ligand
    :param pdb_path: str, path to the PDB file or directory
    :param chain_ids: list, chain IDs for receptor (R) and ligand (L), e.g. ['R', 'L']
        if multiple chains for R or L, use nested lists, e.g. [['R1', 'R2'], ['L1', 'L2']]
    :return: dict, keys are 'receptor' and 'ligand', values are DataFrames with coordinates
    """
    
    parser = PDBParser(PERMISSIVE=1, QUIET=True)

    pdb_files = []
    print('pdb_path', pdb_path)
    # Only process original PDB files, exclude already-split files
    file_names = [f for f in os.listdir(pdb_path) 
                  if os.path.isfile(os.path.join(pdb_path, f)) 
                  and f.endswith('.pdb')
                  and not f.endswith('_antibody.pdb') 
                  and not f.endswith('_antigen.pdb')]
    # print('filename', file_names)
    for file in file_names:

        full_path = os.path.join(pdb_path, file)
        structure = parser.get_structure("file", full_path)

        # Prepare output PDB files
        antibody_chains = chain_ids[0]
        antigen_chains = chain_ids[1]

        print('antibody_chains flat', antibody_chains)
        print('antigen_chains flat', antigen_chains)

        # Create output file names
        antibody_outfile = os.path.join(pdb_path, f"{os.path.splitext(file)[0]}_antibody.pdb")
        antigen_outfile = os.path.join(pdb_path, f"{os.path.splitext(file)[0]}_antigen.pdb")

        # Skip writing if output files already exist
        if os.path.exists(antibody_outfile):
            print(f"Antibody output file {antibody_outfile} already exists, skipping.")
            continue
        if os.path.exists(antigen_outfile):
            print(f"Antigen output file {antigen_outfile} already exists, skipping.")
            continue

        class ChainSelect(Select):
            def __init__(self, chains):
                self.chains = set([c.upper() for c in chains])
            def accept_chain(self, chain):
                return chain.id.upper() in self.chains

        io = PDBIO()
        # Save antibody chains
        io.set_structure(structure)
        # Save each antibody chain separately
        # Combine all antibody chains and save to a single file
        combined_chains = []
        # Flatten antibody_chains if nested and split comma-separated chains
        combined_chains = []
        for chain_group in antibody_chains:
            if isinstance(chain_group, str):
                combined_chains.extend([c.strip() for c in chain_group.split(',')])
            else:
                for chain in chain_group:
                    combined_chains.extend([c.strip() for c in chain.split(',')])
        print('combined antibody chains', combined_chains)
        io.set_structure(structure)
        io.save(antibody_outfile, select=ChainSelect(combined_chains))
        print(f"Saved combined antibody chains to {antibody_outfile}")

        # Save antigen chains
        combined_antigen_chains = []
        for chain in antigen_chains:
            combined_antigen_chains.extend(chain.split(','))
        print('combined antigen chains', combined_antigen_chains)
        io.set_structure(structure)
        io.save(antigen_outfile, select=ChainSelect(combined_antigen_chains))
        print(f"Saved combined antigen chains to {antigen_outfile}")
    

def generate_dataset(split_pdb_files_path, data_savepath, save_name_trainset):
    """
    Generate dataset from split PDB files (antibody and antigen).
    
    :param split_pdb_files_path: Path to directory containing split antibody and antigen PDB files
    :param data_savepath: Path to save the generated dataset pickle file
    :param save_name_trainset: Name of the output pickle file
    """
    dataset = []
    deltaG_values = []
    clusterid = ''
    structure_ids = ''

    file_names = [f for f in os.listdir(split_pdb_files_path) if os.path.isfile(os.path.join(split_pdb_files_path, f))]

    print("file_names", file_names)
    # for file_index in tqdm(range(dataset_len)):
    # Group antibody and antigen files by base name
    antibody_files = [f for f in file_names if f.endswith('_antibody.pdb')]
    antigen_files = [f for f in file_names if f.endswith('_antigen.pdb')]

    # Match antibody and antigen files by base name
    for antibody in antibody_files:
        base_name = antibody.replace('_antibody.pdb', '')
        # print('basename', base_name)
        antigen = f"{base_name}_antigen.pdb"
        antigen_path = os.path.join(split_pdb_files_path, antigen)
        antibody_path = os.path.join(split_pdb_files_path, antibody)

        if not os.path.exists(antigen_path):
            print(f"Antigen file {antigen} not found for antibody {antibody}, skipping this pair...")
            continue
            
        print('antibody', antibody)
        print('antigen', antigen)
        dict_original_coords_AB = P.get_coordinates(antibody_path)
        dict_original_coords_AG = P.get_coordinates(antigen_path)
        # print(dict_original_coords_AB)
        # print(dict_original_coords_AG)

        structure_ids = antibody.replace('_antibody.pdb', '')

        center_coords_AB, bb_min_max_1 = build_bounding_box(dict_original_coords_AB)
        center_coords_AG, bb_min_max_2 = build_bounding_box(dict_original_coords_AG)

        overlap_center_coords, overlap_bb_min_max = find_overlapping_box(bb_min_max_1, bb_min_max_2)

        # Skip examples with no overlapping bounding box
        if not overlap_center_coords:
                print(
                      f'has no overlapping bounding box, excluding this example\n')
                continue
        translation = torch.tensor(tuple(map(lambda x, y: x - y, desired_center, overlap_center_coords))).view(3,1)
            
        transform_matrix = U.build_4x4_transform_mat(translation=translation)
        print("applied transform_matrix", transform_matrix)

        AB_coords_transformed = P.transform_coords(dict_df_CNOS_coords=dict_original_coords_AB, transformation_mat=transform_matrix)
        AG_coords_transformed = P.transform_coords(dict_df_CNOS_coords=dict_original_coords_AG, transformation_mat=transform_matrix)

        AB_volume_transformed = P.coords_to_volume(AB_coords_transformed)
        AG_volume_transformed = P.coords_to_volume(AG_coords_transformed)

        # Build identity 4x4 transformation matrix, even though this method has no ground truth translation or rotation.
        # identity_4x4_transformation_matrix = U.build_4x4_transform_mat()
        identity_4x4_transformation_matrix = torch.eye(4, 4)

        # ### uncomment to visualize volumes: 
        # P.plot_complex_volume(list_of_volumes=[AB_volume_transformed, AG_volume_transformed, 
        #                                        docked_complex_volume],
        #                               # colorscale=['Blackbody', 'Spectral'],
        #                               opacities=[0.2, 0.2],
        #                               surface_counts=[3, 3],
        #                               midpoints=[0.0, 0.0, 0.0, 0.0],
        #                               # isomins=[0.01, 0.01],
        #                               save_fig=False,
        #                               show=True,
        #                               )

        if docked_complex:
                print("Generating docked complex volume")
                docked_complex_coords = P.combine_coords(AB_coords_transformed, AG_coords_transformed)
                docked_complex_volume = P.coords_to_volume(docked_complex_coords)
                docked_complex_volumes = [docked_complex_volume]
                for atom in 'CNOS':
                    # print(atom)
                    docked_complex_atom_type_dict = {atom: docked_complex_coords[atom]}
                    docked_complex_single_atom_volume_antibody = P.coords_to_volume(
                        docked_complex_atom_type_dict)
                    docked_complex_volumes.append(docked_complex_single_atom_volume_antibody)
                docked_complex_volumes = np.stack(docked_complex_volumes)
    
        if atom_type_as_channels:
            # Add standardized total volume as original channel
            antibody_atom_volumes = [AB_volume_transformed]
            antigen_atom_volumes = [AG_volume_transformed]
            for atom in 'CNOS':
                antibody_single_atom_type_dict = {atom: AB_coords_transformed[atom]}
                standardized_single_atom_volume_antibody = P.coords_to_volume(antibody_single_atom_type_dict)
                antibody_atom_volumes.append(standardized_single_atom_volume_antibody)

                antigen_single_atom_type_dict = {atom: AG_coords_transformed[atom]}
                standardized_single_atom_volume_antigen = P.coords_to_volume(antigen_single_atom_type_dict)
                antigen_atom_volumes.append(standardized_single_atom_volume_antigen)

            AB_volume_transformed = np.stack(antibody_atom_volumes)
            AG_volume_transformed = np.stack(antigen_atom_volumes)

        else:
            AB_volume_transformed = np.expand_dims(AB_volume_transformed, axis=0)
            AG_volume_transformed = np.expand_dims(AG_volume_transformed, axis=0)
    
        # Build lists with coordinates for each structure
        list_of_AB_coord_dicts = []
        list_of_AG_coord_dicts = []

        for key in "CNOS":
            list_of_AB_coord_dicts.append(AB_coords_transformed[key].to_numpy())
            list_of_AG_coord_dicts.append(AG_coords_transformed[key].to_numpy())

        # Create ONE training example per antibody-antigen pair
        print('currently structure_ids', structure_ids)
        if docked_complex:
            training_example = [AB_volume_transformed,
                                AG_volume_transformed,
                                docked_complex_volumes,
                                identity_4x4_transformation_matrix,
                                identity_4x4_transformation_matrix,
                                list_of_AB_coord_dicts,
                                list_of_AG_coord_dicts,
                                deltaG_values,
                                clusterid,
                                structure_ids]
        else:
            training_example = [AB_volume_transformed,
                                AG_volume_transformed,
                                identity_4x4_transformation_matrix,
                                identity_4x4_transformation_matrix,
                                list_of_AB_coord_dicts,
                                list_of_AG_coord_dicts,
                                deltaG_values,
                                clusterid,
                                structure_ids]
        
        print("deltaG_values,cluster_id, structure_ids", deltaG_values, clusterid, structure_ids)
        print("Number of Training examples::",len(training_example))
        dataset.append(training_example)
    
    if save_pickle:
        print(f'Saving {len(dataset)} examples to {data_savepath + save_name_trainset}')
        U.write_pkl(data=dataset, filename=data_savepath + save_name_trainset)
        
        # Also save dataset summary to text file
        txt_filename = data_savepath + save_name_trainset.replace('.pkl', '_summary.txt')
        with open(txt_filename, 'w') as f:
            f.write(f"Dataset Summary\n")
            f.write(f"=" * 80 + "\n")
            f.write(f"Total examples: {len(dataset)}\n")
            f.write(f"=" * 80 + "\n\n")
            
            for idx, example in enumerate(dataset):
                f.write(f"\nExample {idx + 1}:\n")
                f.write(f"-" * 40 + "\n")
                if docked_complex:
                    f.write(f"  Structure ID: {example[9]}\n")
                    f.write(f"  Antibody volume shape: {example[0]}\n")
                    f.write(f"  Antigen volume shape: {example[1]}\n")
                    f.write(f"  Docked complex volume shape: {example[2]}\n")
                    f.write(f"  Number of antibody coordinates: {[len(coords) for coords in example[5]]}\n")
                    f.write(f"  Number of antigen coordinates: {[len(coords) for coords in example[6]]}\n")
                    f.write(f"  Transformation matrix:\n{example[3]}\n")
                else:
                    f.write(f"  Structure ID: {example[8]}\n")
                    f.write(f"  Antibody volume shape: {example[0]}\n")
                    f.write(f"  Antigen volume shape: {example[1]}\n")
                    f.write(f"  Number of antibody coordinates: {[len(coords) for coords in example[4]]}\n")
                    f.write(f"  Number of antigen coordinates: {[len(coords) for coords in example[5]]}\n")
                    f.write(f"  Transformation matrix:\n{example[2]}\n")
        
        print(f'Dataset summary saved to {txt_filename}')
           
    print(f'Dataset generation complete: {len(dataset)} examples saved')
    return dataset



if __name__ == '__main__':
    """
    Usage:
        python Inference_dataset_generation.py <csv_file> <pdb_directory> <output_path> <output_name>
    
    Arguments:
        csv_file: Path to CSV file with columns: filename, antibody_chains, antigen_chains
        pdb_directory: Directory containing the PDB files listed in the CSV
        output_path: Directory where the output pickle file will be saved
        output_name: Name of the output pickle file (e.g., 'inference_dataset.pkl')
    
    Example:
        python Inference_dataset_generation.py mappings_example.csv ./pdb_files/ ./datasets/ inference_dataset.pkl
    """
    
    # Parse command line arguments
    if len(sys.argv) != 5:
        print("Usage: python Inference_dataset_generation.py <csv_file> <pdb_directory> <output_path> <output_name>")
        print("\nExample: python Inference_dataset_generation.py mappings_example.csv ./pdb_files/ ./datasets/ inference_dataset.pkl")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    pdb_directory = sys.argv[2]
    output_path = sys.argv[3]
    output_name = sys.argv[4]
    
    print("here")
    # Validate inputs
    print(f"\nValidating inputs...")
    print(f"  Current working directory: {os.getcwd()}")
    print(f"  CSV file argument: {csv_file}")
    print(f"  CSV file absolute path: {os.path.abspath(csv_file)}")
    print(f"  CSV file exists: {os.path.exists(csv_file)}")
    
    if not os.path.exists(csv_file):
        print(f"\nError: CSV file not found: {csv_file}")
        print(f"Absolute path checked: {os.path.abspath(csv_file)}")
        print(f"\nFiles in current directory:")
        for f in os.listdir('.'):
            print(f"  {f}")
        sys.exit(1)
    
    if not os.path.exists(pdb_directory):
        print(f"\nError: PDB directory not found: {pdb_directory}")
        print(f"Absolute path checked: {os.path.abspath(pdb_directory)}")
        sys.exit(1)
    
    if not os.path.exists(output_path):
        print(f"Creating output directory: {output_path}")
        os.makedirs(output_path, exist_ok=True)


    # Fixed parameters for dataset generation
    docked_complex = True
    atom_type_as_channels = True
    normalize_single_atom_kernel = False
    save_pickle = True
    box_dim = 75
    resolution_in_angstroms = 2.0
    sigma = 1.0
    device = 'cpu'
    dtype = torch.float32
    desired_center = (box_dim / 2,) * 3

    print("\n" + "="*80)
    print("Dataset Generation Parameters:")
    print("="*80)
    print(f"Resolution: {resolution_in_angstroms} Å")
    print(f"Box dimension: {box_dim}")
    print(f"Sigma: {sigma}")
    print(f"Device: {device}")
    print(f"Data type: {dtype}")
    print(f"Docked complex: {docked_complex}")
    print(f"Atom type as channels: {atom_type_as_channels}")
    print("="*80 + "\n")
    
    # Initialize utilities
    U = UtilityFunctions(device=device, dtype=dtype)
    P = ProcessCoords(dim=box_dim, resolution_in_angstroms=resolution_in_angstroms,
                      normalize_single_atom_kernel=normalize_single_atom_kernel, sigma=sigma, 
                      device=device, dtype=dtype)

    # Read CSV file with PDB mappings
    print(f"Reading CSV file: {csv_file}")
    df_mappings = pd.read_csv(csv_file)
    # Strip whitespace from column names
    df_mappings.columns = df_mappings.columns.str.strip()
    print(f"Found {len(df_mappings)} PDB files to process")
    print(f"CSV columns: {list(df_mappings.columns)}\n")
    print(df_mappings.head())
    
    # Process each PDB file in the CSV
    for idx, row in df_mappings.iterrows():
        pdb_filename = row['filename']
        # Split by semicolon (;) since chains are separated by semicolons in the CSV
        antibody_chains = [c.strip() for c in str(row['antibody_chains']).split(';')]
        antigen_chains = [c.strip() for c in str(row['antigen_chains']).split(';')]
        
        pdb_file_path = os.path.join(pdb_directory, pdb_filename)
        
        if not os.path.exists(pdb_file_path):
            print(f"Warning: PDB file not found: {pdb_file_path}, skipping...")
            continue
        
        print(f"\nProcessing {pdb_filename}")
        print(f"  Antibody chains: {antibody_chains}")
        print(f"  Antigen chains: {antigen_chains}")
        
        # Step 1: Split PDB file into antibody and antigen
        split_pdb_files(pdb_directory, chain_ids=[antibody_chains, antigen_chains])
    
    # Step 2: Generate dataset from all split PDB files
    print(f"\nGenerating dataset from split PDB files in: {pdb_directory}")
    dataset = generate_dataset(pdb_directory, output_path, output_name)
    
    print(f"\n{'='*80}")
    print(f"Dataset generation complete!")
    print(f"Output saved to: {os.path.join(output_path, output_name)}")
    print(f"Total examples: {len(dataset)}")
    print(f"{'='*80}\n")