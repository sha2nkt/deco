'''
Fix paths  to cropped images, partmasks and segmasks
'''

import argparse
import os
import numpy as np
from tqdm import tqdm

def convert_rich_npz(orig_npz, out_dir):
    # go through all keys in the npz
    # if the key is imgname, partmask or segmask, replace the path with the new path
    # save the new npz

    # structs we use
    imgnames_ = []
    poses_, shapes_, transls_ = [], [], []
    cams_k_ = []
    contact_label_ = []
    scene_seg_, part_seg_ = [], []

    # load the npz
    npz = np.load(orig_npz)
    for i in tqdm(range(len(npz['imgname']))):

        if not os.path.exists(npz['imgname'][i]):
            print(npz['imgname'][i])
            continue

        new_scene_seg = os.path.exists(npz['scene_seg'][i].replace('seg_masks_new', 'segmentation_masks'))

        if not new_scene_seg:
            print(new_scene_seg)
            continue

        if not os.path.exists(npz['part_seg'][i]):
            print(npz['part_seg'][i])
            continue

        imgnames_.append(npz['imgname'][i])
        poses_.append(npz['pose'][i])
        transls_.append(npz['transl'][i])
        shapes_.append(npz['shape'][i])
        cams_k_.append(npz['cam_k'][i])
        contact_label_.append(npz['contact_label'][i])
        scene_seg_.append(npz['scene_seg'][i].replace('seg_masks_new', 'segmentation_masks'))
        part_seg_.append(npz['part_seg'][i])

    # save the new npz
    out_dir = out_dir+'_cropped'
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, os.path.basename(args.orig_npz))
    np.savez(out_file,
             imgname=imgnames_,
             pose=poses_,
             transl=transls_,
             shape=shapes_,
             cam_k=cams_k_,
             contact_label=contact_label_,
             scene_seg=scene_seg_,
             part_seg=part_seg_)

    print('Saved to: ', out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--orig_npz_dir', type=str, default='/is/cluster/fast/achatterjee/rich/scene_npzs/train')
    parser.add_argument('--cluster_idx', type=int)
    args = parser.parse_args()
    # get all npz files in the directory
    npz_files = [os.path.join(args.orig_npz_dir, f) for f in os.listdir(args.orig_npz_dir) if f.endswith('.npz')]
    # get the npz file for this cluster
    orig_npz = npz_files[args.cluster_idx]
    convert_rich_npz(orig_npz, out_dir=args.orig_npz_dir)