'''
Combine the npzs
'''

import numpy as np

d1 = np.load('/is/cluster/work/stripathi/pycharm_remote/dca_contact/data/dataset_extras/hot_noprox_supporting_False_trainval_0.npz',allow_pickle=True)
d2 = np.load('/is/cluster/work/stripathi/pycharm_remote/dca_contact/data/dataset_extras/hot_noprox_supporting_False_trainval_1.npz',allow_pickle=True)
d3 = np.load('/is/cluster/work/stripathi/pycharm_remote/dca_contact/data/dataset_extras/hot_noprox_supporting_False_trainval_2.npz',allow_pickle=True)
d4 = np.load('/is/cluster/work/stripathi/pycharm_remote/dca_contact/data/dataset_extras/hot_noprox_supporting_False_trainval_3.npz',allow_pickle=True)


print(list(d1.keys()))
print(type(d1['imgname']))
print(d1['imgname'].shape)

c_imgname = np.concatenate((d1['imgname'], d2['imgname'], d3['imgname'], d4['imgname']), axis=0)
c_pose = np.concatenate((d1['pose'], d2['pose'], d3['pose'], d4['pose']), axis=0)
c_transl = np.concatenate((d1['transl'], d2['transl'], d3['transl'], d4['transl']), axis=0)
c_shape = np.concatenate((d1['shape'], d2['shape'], d3['shape'], d4['shape']), axis=0)
c_cam_k = np.concatenate((d1['cam_k'], d2['cam_k'], d3['cam_k'], d4['cam_k']), axis=0)
c_polygon_2d_contact = np.concatenate((d1['polygon_2d_contact'], d2['polygon_2d_contact'],
                                       d3['polygon_2d_contact'], d4['polygon_2d_contact']), axis=0)
c_contact_label = np.concatenate((d1['contact_label'], d2['contact_label'], d3['contact_label'],
                                  d4['contact_label']), axis=0)
c_scene_seg = np.concatenate((d1['scene_seg'], d2['scene_seg'], d3['scene_seg'], d4['scene_seg']), axis=0)
c_part_seg = np.concatenate((d1['part_seg'], d2['part_seg'], d3['part_seg'], d4['part_seg']), axis=0)

outfile = '/is/cluster/work/stripathi/pycharm_remote/dca_contact/data/dataset_extras/hot_noprox_supporting_False_trainval_combined.npz'
np.savez(outfile,
         imgname=c_imgname,
         pose=c_pose,
         transl=c_transl,
         shape=c_shape,
         cam_k=c_cam_k,
         polygon_2d_contact=c_polygon_2d_contact,
         contact_label=c_contact_label,
         scene_seg=c_scene_seg,
         part_seg=c_part_seg
        )
print(f'Saved to {outfile}')