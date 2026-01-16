import pickle as pkl
from UtilityFunctions import UtilityFunctions

test_file ="/Users/Anushriya/Documents/Lamoureux_lab/AntibodyDocking/data/test_2example_inference.pkl"


with open(test_file, 'rb') as fin:
    data = pkl.load(fin)
    fin.close()



for i in range(len(data)):
    receptor, ligand, \
    docked_complex_volume,  \
    gt_rot_4x4, gt_txyz_4x4, \
    list_of_AB_coord_dicts, list_of_AG_coord_dicts, \
    deltaGlist, cluster_id, structure_ids = data[i]

    

    print(f"structure_ids : {structure_ids}")
    print(f"receptor: {receptor}")
    print(f"ligand: {ligand}")
    print(f"docked_complex_volume: {docked_complex_volume}")
