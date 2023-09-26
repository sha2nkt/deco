'''
Combine cropped npz given a folder of npz
'''

import argparse
import os
import numpy as np

def combine_npz(npz_dir, out_npz):
    # get all npz files in the folder

    npz_files = [os.path.join(npz_dir, out_dir, f) for f in os.listdir(npz_dir, out_dir) if f.endswith('.npz')]
    print('Found {} npz files'.format(len(npz_files)))

    # combine all the values in all keys in all npz files
    c_imgname = []
    c_pose = []
    c_transl = []
    c_shape = []
    c_cam_k = []
    c_contact_label = []
    c_scene_seg = []
    c_part_seg = []

    for npz_file in npz_files:
        npz= np.load(npz_file)
        c_imgname.extend(npz['imgname'])
        c_pose.extend(npz['pose'])
        c_transl.extend(npz['transl'])
        c_shape.extend(npz['shape'])
        c_cam_k.extend(npz['cam_k'])
        c_contact_label.extend(npz['contact_label'])
        c_scene_seg.extend(npz['scene_seg'])
        c_part_seg.extend(npz['part_seg'])

    # convert to numpy arrays
    c_imgname = np.concatenate(c_imgname, axis=0)
    c_pose = np.concatenate(c_pose, axis=0)
    c_transl = np.concatenate(c_transl, axis=0)
    c_shape = np.concatenate(c_shape, axis=0)
    c_cam_k = np.concatenate(c_cam_k, axis=0)
    c_contact_label = np.concatenate(c_contact_label, axis=0)
    c_scene_seg = np.concatenate(c_scene_seg, axis=0)
    c_part_seg = np.concatenate(c_part_seg, axis=0)

    # save the new npz
    np.savez(out_npz,
                imgname=c_imgname,
                pose=c_pose,
                transl=c_transl,
                shape=c_shape,
                cam_k=c_cam_k,
                contact_label=c_contact_label,
                scene_seg=c_scene_seg,
                part_seg=c_part_seg)
    print('Saved combined npz to {}'.format(out_npz))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--orig_npz_dir', type=str, default='/is/cluster/fast/achatterjee/rich/scene_npzs/train')
    parser.add_argument('--out_npz', type=str, required=True,default='/is/cluster/fast/achatterjee/rich/scene_npzs/train_combined.npz')
    args = parser.parse_args()
    combine_npz(npz_dir = args.orig_npz_dir, out_npz = args.out_npz)