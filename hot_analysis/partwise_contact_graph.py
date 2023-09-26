import numpy as np
import os
import json
import os.path as osp


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

# reverse the part segmentation
part_segmentation_rev = {}
for part in part_segmentation:
    for vert in part_segmentation[part]:
        part_segmentation_rev[vert] = part

part_names_combined = []
for cls in combined_npz['contact_label']:
    cls_idx = np.where(cls == 1)[0]
    part_names = [part_segmentation_rev[vert] for vert in cls_idx]
    # find count for every part in part_names
    part_count = {}
    for part in part_names:
        if part not in part_count:
            part_count[part] = 0
        part_count[part] += 1
    # find the parts with count > 10
    part_names = [part for part in part_count if part_count[part] > 10]
    part_names_combined.append(part_names)

# make a histogram of the part names
part_names_combined = [item for sublist in part_names_combined for item in sublist]
part_names_combined = np.array(part_names_combined)
unique, counts = np.unique(part_names_combined, return_counts=True)
part_names_combined = dict(zip(unique, counts))

print('Total number of parts: ', len(part_names_combined))
print(part_names_combined.keys())
# set keys to combine;
combine_keys = {'leftFoot': ['leftToeBase'],
                'rightFoot': ['rightToeBase'],
                'leftHand': ['leftHandIndex1'],
                'rightHand': ['rightHandIndex1'],
                'spine': ['spine1', 'spine2'],
                'head': ['neck'],}

for key in combine_keys:
    for subkey in combine_keys[key]:
        part_names_combined[key] += part_names_combined[subkey]
        del part_names_combined[subkey]

print('Total number of parts: ', len(part_names_combined))

# sort the dictionary
part_names_combined = {k: v for k, v in sorted(part_names_combined.items(), key=lambda item: item[1], reverse=True)}

# convert to pandas
import pandas as pd
df = pd.DataFrame.from_dict(part_names_combined, orient='index', columns=['count'])
df.to_csv(osp.join(dir, 'partwise_contact_graph.csv'))



# make a beautiful bar plot using seabord and change the default color palette
import seaborn as sns
import matplotlib.pyplot as plt

# set the color palette

# make the plot
fig, ax = plt.subplots(figsize=(20, 10))
sns.barplot(x=list(part_names_combined.keys()), y=list(part_names_combined.values()), ax=ax,
            palette=sns.color_palette("rocket", len(part_names_combined)))
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.xticks(rotation=45, ha='right', fontsize=30)
plt.yticks(fontsize=30)
# avoid clipping of xtick labels
plt.tight_layout()



# save the plot
outdir = "/is/cluster/work/stripathi/pycharm_remote/dca_contact/hot_analysis"
out_path = os.path.join(outdir, "partwise_contact_graph.png")
plt.savefig(out_path, transparent=True, dpi=300)



