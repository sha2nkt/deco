import os
import cv2
import numpy as np
from tqdm import tqdm
import sys
import imagesize
import argparse
import torch
import pandas as pd
import json

import monai.metrics as metrics

PROX_TRAIN_SPLIT = "/ps/scratch/ps_shared/ychen2/4shashank/split/prox_train.odgt"
PROX_VAL_SPLIT = "/ps/scratch/ps_shared/ychen2/4shashank/split/prox_validation.odgt"
PROX_TEST_SPLIT = "/ps/scratch/ps_shared/ychen2/4shashank/split/prox_test.odgt"

def metric(mask, pred, back=True):
  iou = metrics.compute_meaniou(pred, mask, back, False)
  iou = iou.mean()
  return iou


def combine_hot_prox_split(split):
    if split == 'train':
        with open(PROX_TRAIN_SPLIT, "r") as f:
            records = [
                json.loads(line.strip("\n")) for line in f.readlines()
            ]
    elif split == 'val':
        with open(PROX_VAL_SPLIT, "r") as f:
            records = [
                json.loads(line.strip("\n")) for line in f.readlines()
            ]
    elif split == 'test':
        with open(PROX_TEST_SPLIT, "r") as f:
            records = [
                json.loads(line.strip("\n")) for line in f.readlines()
            ]
    return records

def hot_extract(img_dataset_path, smpl_params_path, dca_csv_path, out_dir, split=None, vis_path=None, visualize=False, downsample_factor=4):

    n_vertices = 6890

    # structs we use
    imgnames_ = []
    poses_, shapes_, transls_ = [], [], []
    cams_k_ = []
    polygon_2d_contact_ = []
    contact_3d_labels_ = []
    scene_seg_, part_seg_ = [], []

    img_dir = os.path.join(img_dataset_path, 'images', 'training')
    smpl_params = np.load(smpl_params_path)
    # smpl_params = np.load(smpl_params_path, allow_pickle=True)
    # smpl_params = smpl_params['arr_0'].item()
    annotations_dir = img_dir.replace('images', 'annotations')
    records = combine_hot_prox_split(split)

    # load dca csv
    dca_csv = pd.read_csv(dca_csv_path)

    iou_thresh = 0

    num_with_3d_contact = 0

    focal_length_accumulator = []
    for i, record in enumerate(tqdm(records, dynamic_ncols=True)):
        imgpath = record['fpath_img']
        imgname = os.path.basename(imgpath)
        # save image in temp_images
        if visualize:
            img = cv2.imread(os.path.join(img_dir, imgname))
            cv2.imwrite(os.path.join(vis_path, os.path.basename(imgname)), img)

        # load image to get the size
        img_w, img_h = record["width"], record["height"]

        # get mask anns
        polygon_2d_contact_path = os.path.join(annotations_dir, os.path.splitext(imgname)[0] + '.png')


        # Get 3D contact annotations from DCA mturk csv
        dca_row = dca_csv.loc[dca_csv['imgnames'] == imgname] # if no imgnames column, run scripts/datascripts/add_imgname_column_to_deco_csv.py
        if len(dca_row) == 0:
            contact_3d_labels = []
        else:
            num_with_3d_contact += 1
            supporting_object = dca_row['supporting_object'].values[0]
            vertices = eval(dca_row['vertices'].values[0])
            contact_3d_list = vertices[os.path.join('hot/training/', imgname)]
            # Aggregate values in all keys
            contact_3d_idx = []
            for item in contact_3d_list:
                # one iteration loop as it is a list of one dict key value
                for k, v in item.items():
                    contact_3d_idx.extend(v)
            # removed repeated values
            contact_3d_idx = list(set(contact_3d_idx))
            contact_3d_labels = np.zeros(n_vertices) # smpl has 6980 vertices
            contact_3d_labels[contact_3d_idx] = 1.

        # find indices that match the imname
        inds = np.where(smpl_params['imgname'] == os.path.join(img_dir, imgname))[0]
        select_inds = []
        ious = []
        for ind in inds:
            # part mask
            part_path = smpl_params['part_seg'][ind]
            # load the part_mask
            part_mask = cv2.imread(part_path)
            # binarize the part mask
            part_mask = np.where(part_mask > 0, 1, 0)
            # save part mask
            if visualize:
                cv2.imwrite(os.path.join(vis_path, os.path.basename(part_path)), part_mask*255)

            # load gt polygon mask
            polygon_2d_contact = cv2.imread(polygon_2d_contact_path)
            # binarize the gt polygon mask
            polygon_2d_contact = np.where(polygon_2d_contact > 0, 1, 0)

            # save gt polygon mask in temp_images
            if visualize:
                cv2.imwrite(os.path.join(vis_path, os.path.basename(polygon_2d_contact_path)), polygon_2d_contact*255)

            polygon_2d_contact = torch.from_numpy(polygon_2d_contact)[None,:].permute(0,3,1,2)
            part_mask = torch.from_numpy(part_mask)[None,:].permute(0,3,1,2)
            # compute iou with part mask and gt polygon mask
            iou = metric(polygon_2d_contact, part_mask)
            if iou > iou_thresh:
                ious.append(iou)
                select_inds.append(ind)

        # get select_ind with maximum iou
        if len(select_inds) > 0:
            max_iou_ind = select_inds[np.argmax(ious)]
        else:
            continue

        for ind in select_inds:
            # part mask
            part_path = smpl_params['part_seg'][ind]

            # scene mask
            scene_path = smpl_params['scene_seg'][ind]

            # get smpl params
            pose = smpl_params['pose'][ind]
            shape = smpl_params['shape'][ind]
            transl = smpl_params['global_t'][ind]
            focal_length = smpl_params['focal_l'][ind]
            camC = np.array([[img_w//2, img_h//2]])

            # read GT 2D keypoints
            K = np.eye(3, dtype=np.float64)
            K[0, 0] = focal_length
            K[1, 1] = focal_length
            K[:2, 2:] = camC.T

            # store data
            imgnames_.append(os.path.join(img_dir, imgname))
            polygon_2d_contact_.append(polygon_2d_contact_path)
            # we use the heuristic that the 3D contact labeled is for the person with maximum iou with HOT contacts
            if ind == max_iou_ind:
                contact_3d_labels_.append(contact_3d_labels)
            else:
                contact_3d_labels_.append([])
            scene_seg_.append(scene_path)
            part_seg_.append(part_path)
            poses_.append(pose.squeeze())
            transls_.append(transl.squeeze())
            shapes_.append(shape.squeeze())
            cams_k_.append(K.tolist())
        focal_length_accumulator.append(focal_length)

    print('Average focal length: ', np.mean(focal_length_accumulator))
    print('Median focal length: ', np.median(focal_length_accumulator))
    print('Std Dev focal length: ', np.std(focal_length_accumulator))

    # store the data struct
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f'hot_prox_{split}.npz')
    np.savez(out_file, imgname=imgnames_,
                       pose=poses_,
                       transl=transls_,
                       shape=shapes_,
                       cam_k=cams_k_,
                       polygon_2d_contact=polygon_2d_contact_,
                       contact_label=contact_3d_labels_,
                        scene_seg=scene_seg_,
                        part_seg=part_seg_
             )
    print(f'Total number of rows: {len(imgnames_)}')
    print('Saved to ', out_file)
    print(f'Number of images with 3D contact labels: {num_with_3d_contact}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dataset_path', type=str, default='/ps/project/datasets/HOT/Contact_Data/')
    parser.add_argument('--smpl_params_path', type=str, default='/ps/scratch/ps_shared/stripathi/deco/4agniv/hot/hot.npz')
    parser.add_argument('--dca_csv_path', type=str, default='/ps/scratch/ps_shared/stripathi/deco/4agniv/hot/dca.csv')
    parser.add_argument('--out_dir', type=str, default='/is/cluster/work/stripathi/pycharm_remote/dca_contact/data/dataset_extras')
    parser.add_argument('--vis_path', type=str, default='/is/cluster/work/stripathi/pycharm_remote/dca_contact/temp_images')
    parser.add_argument('--visualize', action='store_true', default=False)
    parser.add_argument('--split', type=str, default='train')
    args = parser.parse_args()

    hot_extract(img_dataset_path=args.img_dataset_path,
                smpl_params_path=args.smpl_params_path,
                dca_csv_path=args.dca_csv_path,
                out_dir=args.out_dir,
                vis_path=args.vis_path,
                visualize=args.visualize,
                split=args.split)

