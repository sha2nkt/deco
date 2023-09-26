import numpy as np
import os
import json
import trimesh
import seaborn as sns


# Load the combined dca train, val and test npzs
dir = '/is/cluster/work/stripathi/pycharm_remote/dca_contact/data/dataset_extras'
trainval_npz = np.load(os.path.join(dir, 'hot_dca_trainval.npz'), allow_pickle=True)
test_npz = np.load(os.path.join(dir, 'hot_dca_test.npz'), allow_pickle=True)

# combine the two npz
combined_npz = {}
for key in trainval_npz.keys():
    combined_npz[key] = np.concatenate([trainval_npz[key], test_npz[key]], axis=0)

segmentation_path = 'data/smpl_vert_segmentation.json'
with open(segmentation_path, 'rb') as f:
    part_segmentation = json.load(f)

combine_keys = {'leftFoot': ['leftToeBase'],
                'rightFoot': ['rightToeBase'],
                'leftHand': ['leftHandIndex1'],
                'rightHand': ['rightHandIndex1'],
                'spine': ['spine1', 'spine2'],
                'head': ['neck'],}

for key in combine_keys:
    for subkey in combine_keys[key]:
        part_segmentation[key] += part_segmentation[subkey]
        del part_segmentation[subkey]

# reverse the part segmentation
part_segmentation_rev = {}
for part in part_segmentation:
    for vert in part_segmentation[part]:
        part_segmentation_rev[vert] = part

# count the number of contact instances per vertex
per_vert_contact_count = np.zeros(6890)
for cls in combined_npz['contact_label']:
    per_vert_contact_count += cls

# calculate the maximum contact count per part
part_contact_max = {}
for part in part_segmentation:
    part_contact_max[part] = np.max(per_vert_contact_count[part_segmentation[part]])

# calculate the contact probability globally
contact_prob = np.zeros(6890)
for vid in range(6890):
    contact_prob[vid] = (per_vert_contact_count[vid] / max(per_vert_contact_count)) ** 0.3

# save the contact probability mesh
outdir = "/is/cluster/work/stripathi/pycharm_remote/dca_contact/hot_analysis"

# load template smpl mesh
mesh = trimesh.load_mesh('data/smpl/smpl_neutral_tpose.ply')
vertex_colors = trimesh.visual.interpolate(contact_prob, 'jet')
# set the vertex colors of the mesh
mesh.visual.vertex_colors = vertex_colors
# save the mesh
out_path = os.path.join(outdir, "contact_probability_mesh.obj")
mesh.export(out_path)

# # calculate the contact probability per part
# contact_prob = np.zeros(6890)
# for vid in range(6890):
#     if 'Hand' in part_segmentation_rev[vid]:
#     contact_prob[vid] = (per_vert_contact_count[vid] / part_contact_max[part_segmentation_rev[vid]]) ** 0.4 if 'Hand' not in part_segmentation_rev[vid] else (per_vert_contact_count[vid] / part_contact_max[part_segmentation_rev[vid]]) ** 0.8
#
# # save the contact probability mesh
# outdir = "/is/cluster/work/stripathi/pycharm_remote/dca_contact/hot_analysis"
#
# # load template smpl mesh
# mesh = trimesh.load_mesh('data/smpl/smpl_neutral_tpose.ply')
# vertex_colors = trimesh.visual.interpolate(contact_prob, 'jet')
# # set the vertex colors of the mesh
# mesh.visual.vertex_colors = vertex_colors
# # save the mesh
# out_path = os.path.join(outdir, "contact_probability_mesh_part.obj")
# mesh.export(out_path)



