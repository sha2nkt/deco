# This script can be used to convert the contact labels from SMPL to SMPL-X format and vice-versa.

import os
import argparse
import pickle as pkl
import torch
import numpy as np
from common import constants

def convert_contacts(contact_labels, mapping):
    """
    Converts the contact labels from SMPL to SMPL-X format and vice-versa.

    Args:
        contact_labels: contact labels in SMPL or SMPL-X format
        mapping: mapping from SMPL to SMPL-X vertices or vice-versa

    Returns:
        contact_labels_converted: converted contact labels
    """
    bs = contact_labels.shape[0]
    mapping = mapping[None].expand(bs, -1, -1)
    contact_labels_converted = torch.bmm(mapping, contact_labels[..., None])
    contact_labels_converted = contact_labels_converted.squeeze()
    return contact_labels_converted

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--contact_npz', type=str, required=True, help='path to contact npz file',
                        default='../datasets/ReleaseDatasets/damon/hot_dca_train.npz')
    parser.add_argument('--input_type', type=str, required=True, help='input type: smpl or smplx',
                        default='smpl')
    args = parser.parse_args()
    
    if args.input_type == 'smpl':
        # load mapping from smpl to smplx vertices 
        mapping_pkl = os.path.join(constants.CONTACT_MAPPING_PATH, "smpl_to_smplx.pkl")
    elif args.input_type == 'smplx':
        # load mapping from smplx to smpl vertices
        mapping_pkl = os.path.join(constants.CONTACT_MAPPING_PATH, "smplx_to_smpl.pkl")
    else:
        raise ValueError('input_type must be smpl or smplx')
    
    with open(mapping_pkl, 'rb') as f:
        mapping = pkl.load(f)
        mapping = mapping["matrix"]

    # load contact labels
    contact_data = np.load(args.contact_npz, allow_pickle=True)
    contact_data = dict(contact_data)
    contact_labels = contact_data['contact_label']
    if not isinstance(contact_labels, torch.Tensor):
        contact_labels = torch.from_numpy(contact_labels).float()
    if not isinstance(mapping, torch.Tensor):
        mapping = torch.from_numpy(mapping).float()
    contact_labels_converted = convert_contacts(contact_labels, mapping)
    contact_data['contact_label_smplx'] = contact_labels_converted.numpy()
    # save the converted contact labels
    np.savez(args.contact_npz, **contact_data)



