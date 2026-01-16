import _pickle as pkl
import numpy as np
import torch
from torch.utils.data import Dataset, RandomSampler


class InteractionPoseDataset(Dataset):
    def __init__(self, path, docked_complex=False, clustering=False, max_size=None):
        '''
        Preprocess .pkl dataset into torch datastream for interaction pose (IP) prediction

        Args:
            path: path to docking dataset .pkl file.
            docked_complex: Pass in the docked complex structure (both receptor and ligand in a volume) as input to the model
            clustering: cluster the dataset defined in InteractionPoseDataset.select_clusters()
            max_size: number of docking examples to be loaded into data stream
        '''
        self.docked_complex = docked_complex
        self.clustering = clustering
        self.path = path

        if self.clustering:
            with open(self.path, 'rb') as fin:
                self.data_in = pkl.load(fin)

            self.select_clusters()
        else:
            self.path = path
            with open(self.path, 'rb') as fin:
                self.data = pkl.load(fin)

        if not max_size:
            max_size = len(self.data)
        self.data = self.data[:max_size]
        self.dataset_size = len(list(self.data))

        print("Dataset file: ", self.path)
        print("Dataset size: ", self.dataset_size)

    def __getitem__(self, index):
        """
        :return: unpacks and repacks values at index of interaction data
        """
        if self.clustering and index == 0:
            print('shuffling dataset for cluster selection')
            self.select_clusters()

        if self.docked_complex:
            receptor, ligand,\
            docked_complex_volume, rotation4x4, translation4x4, list_of_AB_coord_dicts, list_of_AG_coord_dicts, deltaG_list, clusterID, structure_ids = self.data[index]
        else:
            receptor, ligand, \
            rotation4x4, translation4x4, list_of_AB_coord_dicts, list_of_AG_coord_dicts, deltaG_list, clusterID, structure_ids = self.data[index]

        # print('len(list_of_AB_coord_dicts)',len(list_of_AB_coord_dicts))
        # print('len(list_of_AG_coord_dicts)',len(list_of_AG_coord_dicts))

        if len(list_of_AB_coord_dicts[-1]) == 0:
            list_of_AB_coord_dicts = list_of_AB_coord_dicts[:3]
        if len(list_of_AG_coord_dicts[-1]) == 0:
            list_of_AG_coord_dicts = list_of_AG_coord_dicts[:3]

        # [print(i) for i in self.data[index][:]]

        if not clusterID:
            clusterID = ''

        if self.docked_complex:
            return receptor, ligand,\
                   docked_complex_volume, rotation4x4, translation4x4, list_of_AB_coord_dicts, list_of_AG_coord_dicts, deltaG_list, clusterID, structure_ids
        else:
            return receptor, ligand,\
                   rotation4x4, translation4x4, list_of_AB_coord_dicts, list_of_AG_coord_dicts, deltaG_list, clusterID, structure_ids

    def __len__(self):
        """
        :return: length of the dataset
        """
        return self.dataset_size

    def select_clusters(self):
        '''
        Cluster out examples from the dataset based on predetermined cluster ids.

        Returns:
            Kept examples that are not found in cluster_id_list
        '''
        np.random.shuffle(self.data_in)
        cluster_id_list = []
        self.data = []
        for example in self.data_in:
            cluster_id = example[-1]
            # print(cluster_id)
            # if cluster_id == 'UniRef50_P01857':
            #     print(example[3])
            if cluster_id not in cluster_id_list:
                cluster_id_list.append(cluster_id)
                self.data.append(example)
        # print('represented unique clusters', cluster_id_list)


def get_docking_stream(data_path, shuffle=True, docked_complex=False, clustering=False, max_size=None, num_workers=0):
    '''
        Get docking data as a torch data stream that is randomly shuffled per epoch.
        This function is typically called in Trainer*.py

    Args:
        data_path: path to dataset .pkl file.
        shuffle: shuffle using RandomSampler() or not
        docked_complex: Pass in the docked complex structure (both receptor and ligand in a volume) as input to the model
        clustering: cluster the dataset defined in InteractionPoseDataset.select_clusters()
        max_size: number of docking examples to be loaded into data stream
        num_workers: number of cpu threads

    Returns:
        docking data stream in format as defined in InteractionPoseDataset.__getitem__()
    '''
    dataset = InteractionPoseDataset(path=data_path, max_size=max_size, clustering=clustering, docked_complex=docked_complex)
    if shuffle:
        sampler = RandomSampler(dataset)
    else:
        sampler = None
    train_loader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=1, num_workers=num_workers)
    return train_loader
