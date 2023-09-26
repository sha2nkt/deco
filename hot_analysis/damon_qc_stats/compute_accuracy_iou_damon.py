# load amt csv, go through each line in vertices, combine the vertices for each object label and then compute the iou with GT from RICH and PROX
import pandas as pd
import numpy as np

# load csv
csv_path = './quality_assurance_accuracy.csv'
df = pd.read_csv(csv_path)

# load gt npz
gt_path = './qa_accuracy_gt_contact_combined.npz'
gt = np.load(gt_path)

def compute_iou(pred_verts, gt_verts):
    if len(pred_verts) != 0:
        intersect = list(set(pred_verts) & set(gt_verts))
        iou = len(intersect) / (len(pred_verts) + len(gt_verts) - len(intersect))
    else:
        iou = 0
    return iou

all_ious = []
# for loop each row in df
for index, row in df.iterrows():
    combined_annotation_ids = []
    imgname = []
    # get vertices
    annotation_dict = eval(row['vertices'])
    worker_id = row['WorkerId']
    # single for loop in the dictionary
    for im, anno in annotation_dict.items():
        imgname.append(im)
        for ann in anno:
            # single for loop in the dict
            for k, v in ann.items():
                combined_annotation_ids.extend(v)
    # remove repeated values
    combined_annotation_ids = list(set(combined_annotation_ids))

    assert len(imgname) == 1
    imgname = imgname[0]

    # get gt for the imgname
    gt_ids = gt[imgname]
    if 'prox' in imgname:
        continue

    # compute iou
    iou = compute_iou(combined_annotation_ids, gt_ids)
    print('worker id: ', worker_id, 'imgname: ', imgname, 'iou: ', iou)
    all_ious.append(iou)

# compute mean iou
mean_iou = np.mean(all_ious)
print('mean iou: ', mean_iou)





