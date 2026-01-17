from pymol import cmd
import numpy as np
import re
import sys
import os


class TransformationMatrixHandler:
    def __init__(self, tf_mat_file):
        self.tf_mat_file = tf_mat_file
        self.tf_mat_dict = self.parse_inputvol_TFM()

    def parse_inputvol_TFM(self):
        combined_dict = {}

        with open(self.tf_mat_file, 'r') as file:
            content = file.read()

        # Regex to match each entry
        pattern = r"\[\[(.*?)\], tensor\((.*?)\)\]"
        matches = re.finditer(pattern, content.strip(), re.DOTALL)

        for match in matches:
            # Extract keys and matrix for each match
            keys_str = match.group(1).strip()
            matrix_str = match.group(2).strip()

            # Convert keys to a list
            keys = [k.strip().strip("'") for k in keys_str.split(",")]

            # Convert the matrix to a numpy array
            matrix = np.array(eval(matrix_str))

            # Add to the dictionary
            combined_dict[tuple(keys)] = matrix

        return combined_dict

    def reformat_tf4pymol(self):
        reformat_dict = {}
        for k, v in self.tf_mat_dict.items():
            if v.shape != (4, 4):
                raise ValueError("Input matrix must be of shape (4, 4).")

            # Multiply the last column except the last row by 2
            v[:-1, -1] *= 2

            # Flatten the matrix into a list
            flattened_matrix = v.flatten().tolist()
            reformat_dict[k] = flattened_matrix

        return reformat_dict

    def write_pymol_format(self, tf_scaled_flat_matrix, save_path):
        """
        Formats the scaled-transformed matrix for PyMOL transformation and applies it.

        Args:
            tf_scaled_flat_matrix (list): Flattened transformation matrix (length 16).
            complex_name (str): Name of the PyMOL object/selection to transform.

        Returns:
            None: The function directly applies the transformation to the PyMOL object.
        """
        complex_name = ''
        tf_matrix_dict = {}
        for k, v in tf_scaled_flat_matrix.items():
            k_split = k[0].split("_")
            prefix = '*'
            comp_name = f"{prefix}_{k_split[-1][:-4]}"
            complex_name = comp_name
        
            tf_scaled_flat_matrix = v

            if len(tf_scaled_flat_matrix) != 16:
                raise ValueError("Input matrix must be a flattened list of length 16.")

            tf_matrix_dict[complex_name] = tf_scaled_flat_matrix
            # save_pymol_str = f"cmd.transform_selection({complex_name}, {tf_scaled_flat_matrix}, homogenous=1)"
            # print('save_pymol_str\n', save_pymol_str)

        return tf_matrix_dict
    
def set_pymol_env(working_dir):
    """
    Sets the PyMOL environment settings.
    """
    cmd.pymol.finish_launching()

    os.chdir(working_dir)
    cmd.bg_color("white")
    cmd.set("normalize_ccp4_maps", 0)
    cmd.set('internal_gui_width',400)
    cmd.set("ray_opaque_background", 0)

def load_volumes(volume_dir, epoch, example_number, prefix='test', rmsd_value='0.3333'):
    """
    Loads all volume files from the specified directory into PyMOL.
    
    Parameters:
        volume_dir (str): Path to the directory containing volume files.
        epoch (int): Epoch number
        example_number (int): Example number
        prefix (str): Prefix for volume files (e.g., 'test', 'Homology', 'crystal')
        rmsd_value (str): RMSD value for volume filenames
    """

    # List all files in the directory
    volume_files = []
    volume_files = [f for f in os.listdir(volume_dir) if (f.startswith(prefix) and (f.endswith('.map') or f.endswith('.ccp4')))]
    
    if not volume_files:
        print(f"No volume files found in directory: {volume_dir}")
        return

    for volume_file in volume_files:
        if f'_epoch{epoch}_exp{example_number}_' in volume_file:
        # if 'scalar_0_complex_lig_scalars_epoch2770_exp5_rmsd0.0_feats' in volume_file:
            volume_path = os.path.join(volume_dir, volume_file)
            volume_name = os.path.splitext(volume_file)[0]

            # print(f"Loading volume: {volume_name}")
            cmd.load(volume_path, volume_name)

            # Define the volume object with the specified visualization settings
            volume_object_name = f"{volume_name}_volume"
            
            cmd.volume(volume_object_name, volume_name)

    all_objects = cmd.get_names("objects")
    existing_copies = [obj for obj in all_objects if obj.startswith('copy_')]


    for obj in all_objects:
        obj_split = obj.split("_")
        
        # if obj.startswith('diff') or obj.startswith('sum') and obj.endswith('_volume'):
        if obj_split[1]=='diff' or obj_split[1] =='sum' and obj.endswith('_volume'):
            vol_posval_colors(obj)

        
        if obj_split[1]==('scalar_0') and obj.endswith('_volume') and not obj_split[1]==('copy_scalar_0_complex') and not obj_split[1]==('scalar_0_complex'):
            vol_docked_feature0_ramp(obj)

        if len(obj.split("_")) > 3 and obj.split("_")[1] == 'scalar' and obj.split("_")[2] == '0' and obj.split("_")[3] == 'complex' and obj.endswith('_volume'):
            print('found docked scalar 0 ')
            vol_docked_feature0_ramp(obj)

    print(all_objects)

    # Create dynamic filenames using f-strings
    diff_lig = f"{prefix}_diff_lig_epoch{epoch}_exp{example_number}_rmsd{rmsd_value}_feats"
    diff_rec = f"{prefix}_diff_rec_epoch{epoch}_exp{example_number}_rmsd{rmsd_value}_feats"
    sum_rec = f"{prefix}_sum_rec_epoch{epoch}_exp{example_number}_rmsd{rmsd_value}_feats"
    sum_lig = f"{prefix}_sum_lig_epoch{epoch}_exp{example_number}_rmsd{rmsd_value}_feats"
    f0_complex = f"{prefix}_scalar_0_complex_epoch{epoch}_exp{example_number}_rmsd{rmsd_value}_feats"
    f0_lig = f"{prefix}_scalar_0_lig_epoch{epoch}_exp{example_number}_rmsd{rmsd_value}_feats"
    f0_rec = f"{prefix}_scalar_0_rec_epoch{epoch}_exp{example_number}_rmsd{rmsd_value}_feats"

    cmd.slice_new('diff_lig_slice', diff_lig)
    cmd.slice_new('diff_rec_slice', diff_rec)
    cmd.slice_new('sum_rec_slice', sum_rec)
    cmd.slice_new('sum_lig_slice', sum_lig)

    cmd.slice_new('f0_complex_slice', f0_complex)
    cmd.slice_new('f0_lig_slice', f0_lig)
    cmd.slice_new('f0_rec_slice', f0_rec)
      
    ## setting color ramps for volume slices:
    cmd.ramp_new("sum_rec_ramp", sum_rec, [-1, -0.3, 0.3, 1], [[1,0.5,0], [1,1,1], [1,1,1], [0,0.5,0.5]])
    cmd.ramp_new("sum_lig_ramp", sum_lig, [-1, -0.3, 0.3, 1], [[1,0.5,0], [1,1,1], [1,1,1], [0,0.5,0.5]])
    cmd.ramp_new("diff_rec_ramp", diff_rec, [-1, -0.3, 0.3, 1], [[1,0.5,0], [1,1,1], [1,1,1], [0,0.5,0.5]])
    cmd.ramp_new("diff_lig_ramp", diff_lig, [-1, -0.3, 0.3, 1], [[1,0.5,0], [1,1,1], [1,1,1], [0,0.5,0.5]])

    ## red blue color scheme
    cmd.ramp_new("sum_rec_ramp_redblue", sum_rec, [-1, -0.3, 0.3, 1], [[1, 0, 0], [1, 1, 1], [1, 1, 1], [0, 0, 1]])


    # feature 0 same color as above slices
    cmd.ramp_new("feature0comp_ramp", f0_complex, [-0.001, -0.0003, 0.0003, 0.0015], [[1,0.5,0], [1,1,1], [1,1,1], [0,0.5,0.5]])
    
    ## red(-), blue(+) heat map color for feature 0 slices
    cmd.ramp_new("feature0comp_ramp_heatmap", f0_complex, [-0.001, -0.0003, 0.0003, 0.0015], [[1,0,0], [1,1,1], [1,1,1], [0,0,1]])

    ## save images 
    cmd.png("f0_rec_slice_red_blue", width=2400, height=1600, dpi=250, ray=1, )


def load_pdbs(pdb_dir, pdb_id, tf_matrix):

    """
    Loads all PDB files with specific file names containing '_3ngb' from the specified directory into PyMOL.
    
    Parameters:
        pdb_dir (str): Path to the directory containing PDB files.
    """
    # List all files in the directory
    pdb_id = pdb_id
    pdb_files = [f for f in os.listdir(pdb_dir) if f.endswith('.pdb') and pdb_id in f]

    if not pdb_files:
        print(f"No PDB files with {pdb_id} found in directory: {pdb_dir}")
        return
    
    # print('tf_matrix in load pdb', tf_matrix)
    # print(type(tf_matrix))


    for pdb_file in pdb_files:
        pdb_path = os.path.join(pdb_dir, pdb_file)
        pdb_name = os.path.splitext(pdb_file)[0]

        print(f"Loading PDB: {pdb_name}")
        cmd.load(pdb_path, pdb_name)

        cmd.spectrum("b", "cbmr")
        cmd.show("cartoon", pdb_name)
        cmd.cartoon("putty", pdb_name)
        
    pdb_id = f'*{pdb_id}*'
    # print("input, pdb_id", pdb_id)
    for k, v in tf_matrix.items():
        # print(k, pdb_id)
        if k == pdb_id:
            # print('match found k=pdb', pdb_id)
            cmd.transform_selection(pdb_id, v, homogenous=1)
            # print(f"Transformation matrix for {pdb_name}: {tf_matrix[pdb_id]}")
    
    cmd.do("run_apbs")
    cmd.reset()

def vol_negval_colors(volume_object):
    """
    Applies a custom color ramp to the specified volume object in PyMOL.
    
    Parameters:
        volume_object (str): Name of the volume object to which the color ramp will be applied.
    """
    print(f"Inside vol_negval_colors, applying to: {volume_object}")

    # cmd.volume_ramp_new('ramp625', [
    #     -1.24, 1.00, 0.80, 0.10, 0.08, 
    #     -0.95, 1.00, 0.80, 0.10, 0.03, 
    #     -0.92, 1.00, 0.80, 0.10, 0.01, 
    # ])

    cmd.volume_ramp_new('ramp625', [\
     -4.75, 0.22, 1.00, 0.08, 0.75, \
     -3.73, 0.04, 0.53, 0.07, 0.00, \
     -2.47, 0.04, 0.53, 0.07, 0.01, \
     -1.27, 0.04, 0.53, 0.07, 0.00, \
     -0.96, 0.02, 0.35, 0.04, 0.07, \
    ])

    # Check if the volume object exists
    # all_objects = cmd.get_names("objects")
    # print("all_objects", all_objects)
    # if volume_object not in all_objects:
    #     print(f"Error: Volume object '{volume_object}' does not exist.")
    #     return

    # Apply the color ramp to the volume object
    try:
        cmd.volume_color(volume_object, 'ramp625')
        print(f"Applied color ramp 'ramp625' to volume object '{volume_object}'.")
    except Exception as e:
        print(f"Error applying color ramp: {e}")

def vol_posval_colors(volume_object,):
    """
    Applies a custom color ramp to the specified volume object in PyMOL.
    
    Parameters:
        volume_object (str): Name of the volume object to which the color ramp will be applied.
    """
 

    cmd.volume_ramp_new('ramp006', [\
      0.63, 0.83, 0.55, 0.00, 0.08, \
      1.08, 0.83, 0.55, 0.00, 0.02, \
      3.06, 0.83, 0.55, 0.00, 0.00, \
      4.28, 0.80, 0.98, 0.02, 0.05, \
      4.98, 1.00, 1.00, 0.00, 0.88, \
    ])
    

    # Apply the color ramp to the volume object
    try:
        cmd.volume_color(volume_object, 'ramp006')
        print(f"Applied color ramp 'ramp006' to volume object '{volume_object}'.")
    except Exception as e:
        print(f"Error applying color ramp: {e}")

def volpos_feature0_ramp(volume_object):
    """
    Applies a custom color ramp to the Feature 0 volume object in PyMOL.
    
    Parameters:
        volume_object (str): Name of the volume object to which the color ramp will be applied.
    """
    
    cmd.volume_ramp_new('ramp378', [
        0.01, 0.33, 0.67, 0.00, 0.00, 
        0.01, 0.33, 0.67, 0.00, 0.06, 
        0.02, 0.33, 0.67, 0.00, 0.16, 
    ])

    # Apply the color ramp to the volume object
    try:
        cmd.volume_color(volume_object, 'ramp378')
        print(f"Applied color ramp 'ramp378' to volume object '{volume_object}'.")
    except Exception as e:
        print(f"Error applying color ramp: {e}")

def volneg_feature0_ramp(volume_object):
    """
    Applies a custom color ramp to the Feature 0 volume object in PyMOL.
    
    Parameters:
        volume_object (str): Name of the volume object to which the color ramp will be applied.
    """
    
    # cmd.volume_ramp_new('ramp729', [
    #     -0.00, 1.00, 0.67, 0.00, 0.11, 
    #     -0.00, 1.00, 0.67, 0.50, 0.00, 
    # ])
    cmd.volume_ramp_new('ramp450', [
        -0.00, 1.00, 0.67, 0.00, 0.23, 
        -0.00, 1.00, 0.67, 0.00, 0.07, 
        -0.00, 1.00, 0.67, 0.00, 0.01, 
    ])


    # Apply the color ramp to the volume object
    try:
        cmd.volume_color(volume_object, 'ramp450')
        print(f"Applied color ramp 'ramp450' to volume object '{volume_object}'.")
    except Exception as e:
        print(f"Error applying color ramp: {e}")

def vol_docked_feature0_ramp(volume_object):
    """
    Applies a custom color ramp to the Feature 0 volume object in PyMOL.
    
    Parameters:
        volume_object (str): Name of the volume object to which the color ramp will be applied.
    """


    print("coloring using ramp for vol_docked_feature0_ramp")
    cmd.volume_ramp_new('ramp642', [\
      0.008, 1.00, 0.67, 0.50, 0.32, \
      0.01, 1.00, 0.68, 0.17, 0.00, \
      0.04, 1.00, 0.58, 0.27, 0.00, \
      0.06, 1.00, 0.00, 1.00, 0.01, \
      0.13, 0.63, 0.20, 1.00, 0.00, \
      0.15, 0.91, 0.46, 1.00, 0.30, \
    ])

    try:
        cmd.volume_color(volume_object, 'ramp642')
        print(f"Applied color ramp 'ramp642' to volume object '{volume_object}'.")
        # print("checking color ",cmd.get_color_tuple('ramp642'))
    except Exception as e:
        print(f"Error applying color ramp: {e}")



experiment_name = 'Experiment_NAME'
base_dir = os.path.join(os.path.expanduser('~'),  'SE3Bind')
volume_path = os.path.join(base_dir, 'src', 'Figs', 'Feature_volumes', experiment_name)
working_dir = os.path.join(base_dir, 'src')

input_vol_transmat_path = os.path.join('..', 'data', 'datasets', 'transformation_mat.txt')

pdb_dir = os.path.join(os.path.expanduser('~'), 'Documents', 'pdbs')

tf_handler = TransformationMatrixHandler(input_vol_transmat_path)

reformated_matrix_dict = tf_handler.reformat_tf4pymol()

tf_matrix = tf_handler.write_pymol_format(reformated_matrix_dict, save_path=input_vol_transmat_path)


load_epoch = 1500
example_number = 10
pdb_id = '_5mev_'
data_prefix = 'test'  # Change to 'Homology' or 'crystal' as needed
rmsd = '0.3333'

#set up pymol environments
set_pymol_env(working_dir)

load_volumes(volume_path, load_epoch, example_number, prefix=data_prefix, rmsd_value=rmsd)

load_pdbs(pdb_dir, pdb_id, tf_matrix)



"""Step after running script in pymol:

run script, orient protein complex, then center all, and zoom uniformly. 
center * 
zoom center, 50

"""
# save pngs from saved scenes:
## png protein_complex_bfactor.png, dpi=1000, height=1200, width=1450
