"""
This file contains the definition of different heterogeneous datasets used for training
"""
import torch
import numpy as np

from .base_dataset import BaseDataset

class MixedDataset(torch.utils.data.Dataset):

    def __init__(self, ds_list, mode, dataset_mix_pdf, **kwargs):
        self.dataset_list = ds_list
        print('Training Dataset list: ', self.dataset_list)
        self.num_datasets = len(self.dataset_list)

        self.datasets = []
        for ds in self.dataset_list:
            if ds in ['rich', 'prox']:
                self.datasets.append(BaseDataset(ds, mode, model_type='smplx', **kwargs))
            elif ds in ['hot', 'hot_nosupport', 'dca']:
                self.datasets.append(BaseDataset(ds, mode, model_type='smpl', **kwargs))
            else:
                raise ValueError('Dataset not supported')

        total_length = sum([len(ds) for ds in self.datasets])
        length_itw = sum([len(ds) for ds in self.datasets])
        self.length = max([len(ds) for ds in self.datasets])

        # convert list of strings to list of floats
        self.partition = [float(i) for i in dataset_mix_pdf] # should sum to 1.0 unless you want to weight by dataset size
        assert sum(self.partition) == 1.0, "Dataset Mix PDF must sum to 1.0 unless you want to weight by dataset size"
        assert len(self.partition) == self.num_datasets, "Number of partitions must be equal to number of datasets"
        self.partition = np.array(self.partition).cumsum()

    def __getitem__(self, index):
        p = np.random.rand()
        for i in range(self.num_datasets):
            if p <= self.partition[i]:
                return self.datasets[i][index % len(self.datasets[i])]

    def __len__(self):
        return self.length
