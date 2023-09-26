# python test_chore_phosa.py --dataset_root /is/cluster/fast/achatterjee/CHORE_DECO/phosa_test --npz_file /is/cluster/fast/achatterjee/Datasets/hot_phosa/hot_phosa_test.npz --json_file /ps/scratch/ps_shared/stripathi/deco/4agniv/hot_phosa_split/imgnames_per_object_dict.json --th 0.05

import numpy as np
import pandas as pd
import torch
import os
import os.path as osp
import json
import trimesh
import argparse
import collections
from tqdm import tqdm

DIST_MATRIX = np.load('/is/cluster/fast/achatterjee/dca_contact/data/smpl/smpl_neutral_geodesic_dist.npy')

def precision_recall_f1score(gt, pred):
    """
    Compute precision, recall, and f1
    """
    precision = np.zeros(gt.shape[0])
    recall = np.zeros(gt.shape[0])
    f1 = np.zeros(gt.shape[0])
    
    for b in range(gt.shape[0]):
        tp_num = gt[b, pred[b, :] >= 0.5].sum()
        precision_denominator = (pred[b, :] >= 0.5).sum()
        recall_denominator = (gt[b, :]).sum()

        precision_ = tp_num / precision_denominator
        recall_ = tp_num / recall_denominator
        if precision_denominator == 0: # if no pred
            precision_ = 1.
            recall_ = 0.
            f1_ = 0.
        elif recall_denominator == 0: # if no GT
            precision_ = 0.
            recall_ = 1.
            f1_ = 0.
        elif (precision_ + recall_) <= 1e-10:  # to avoid precision issues
            precision_= 0.
            recall_= 0.
            f1_ = 0.
        else:
            f1_ = 2 * precision_ * recall_ / (precision_ + recall_)

        precision[b] = precision_
        recall[b] = recall_
        f1[b] = f1_

    return precision, recall, f1

def det_error_metric(gt, pred):
    dist_matrix = torch.tensor(DIST_MATRIX)

    false_positive_dist = torch.zeros(gt.shape[0])
    false_negative_dist = torch.zeros(gt.shape[0])
    
    for b in range(gt.shape[0]):
        gt_columns = dist_matrix[:, gt[b, :]==1] if any(gt[b, :]==1) else dist_matrix
        error_matrix = gt_columns[pred[b, :] >= 0.5, :] if any(pred[b, :] >= 0.5) else gt_columns

        false_positive_dist_ = error_matrix.min(dim=1)[0].mean()
        false_negative_dist_ = error_matrix.min(dim=0)[0].mean()

        false_positive_dist[b] = false_positive_dist_
        false_negative_dist[b] = false_negative_dist_

    return false_positive_dist, false_negative_dist
    
def main(args):    
    with open(args.json_file, 'r') as fp:
        img_list_dict = json.load(fp)
    
    d = np.load(args.npz_file)
    img_count = 0
    
    tot_pre = 0
    tot_rec = 0
    tot_f1 = 0
    tot_fp_err = 0
    
    # for i, img in tqdm(enumerate(d['imgname']), dynamic_ncols=True):
    #     gt = d['contact_label'][i]
    #     gt = np.reshape(gt, (1, -1))
                
    #     imgname = osp.basename(img)
    #     for k, v in img_list_dict.items():
    #         if imgname in v: 
    #             object_class = k
    #             break
    #     # pred_path = osp.join(args.dataset_root, object_class, imgname, 'chore-test', f'k1.smpl_colored_{args.th}.ply')
    #     pred_path = osp.join(args.dataset_root, object_class, imgname, f'k1.smpl_colored_{args.th}.obj')
    #     if not osp.exists(pred_path):
    #         # print(pred_path)
    #         continue
    #     pred_mesh = trimesh.load(pred_path, process=False)
        
    #     pred = np.zeros((1, 6890))
    #     for i in range(pred_mesh.visual.vertex_colors.shape[0]):
    #         c = pred_mesh.visual.vertex_colors[i]
    #         if collections.Counter(c) == collections.Counter([255, 0, 0, 255]):
    #             pred[0, i] = 1
                
    #     pre, rec, f1 = precision_recall_f1score(gt, pred)
    #     fp_err, _ = det_error_metric(pred, gt)  
        
    #     tot_pre += pre.sum()
    #     tot_rec += rec.sum()
    #     tot_f1 += f1.sum()
    #     tot_fp_err += fp_err.numpy().sum()
        
    #     img_count += 1   
        
    # print(f'Dataset size: {img_count}')
    # print(f'Threshold: {args.th}\n')
    
    # print(f'Test Precision: {tot_pre/img_count}')  
    # print(f'Test Recall: {tot_rec/img_count}') 
    # print(f'Test F1 Score: {tot_f1/img_count}') 
    # print(f'Test FP Error: {tot_fp_err/img_count}')  

    # Part of code for checking a random image
    img_search = osp.join('/ps/project/datasets/HOT/Contact_Data/images/training', 'vcoco_000000542163.jpg')
    i = np.where(d['imgname'] == img_search)[0][0]
    print(i)
    # i = 50
    img = d['imgname'][i]
    gt = d['contact_label'][i]
    gt = np.reshape(gt, (1, -1))
                
    imgname = osp.basename(img)
    print(f'Image: {imgname}')
    for k, v in img_list_dict.items():
        if imgname in v: 
            object_class = k
            break
    print(f'Object: {object_class}')
    # pred_path = osp.join(args.dataset_root, object_class, imgname, 'chore-test', f'k1.smpl_colored_{args.th}.ply')
    pred_path = osp.join(args.dataset_root, object_class, imgname, f'k1.smpl_colored_{args.th}.obj')
    if not osp.exists(pred_path):
        print(f'Missing file: {pred_path}')
    pred_mesh = trimesh.load(pred_path, process=False)
        
    pred = np.zeros((1, 6890))
    for i in range(pred_mesh.visual.vertex_colors.shape[0]):
        c = pred_mesh.visual.vertex_colors[i]
        if collections.Counter(c) == collections.Counter([255, 0, 0, 255]):
            pred[0, i] = 1
                
    pre, rec, f1 = precision_recall_f1score(gt, pred)
    fp_err, _ = det_error_metric(pred, gt)  
        
    tot_pre += pre.sum()
    tot_rec += rec.sum()
    tot_f1 += f1.sum()
    tot_fp_err += fp_err.numpy().sum() 
            
    print(f'Test Precision: {tot_pre}')  
    print(f'Test Recall: {tot_rec}') 
    print(f'Test F1 Score: {tot_f1}') 
    print(f'Test FP Error: {tot_fp_err}')  

    # best_pre = 0
    # best_rec = 0
    # best_f1 = 0
    # best_fp_err = 0
    # best_imgname = ''
    # best_obj = ''

    # for i, img in tqdm(enumerate(d['imgname']), dynamic_ncols=True):
    #     gt = d['contact_label'][i]
    #     gt = np.reshape(gt, (1, -1))
                
    #     imgname = osp.basename(img)
    #     for k, v in img_list_dict.items():
    #         if imgname in v: 
    #             object_class = k
    #             break
    #     # pred_path = osp.join(args.dataset_root, object_class, imgname, 'chore-test', f'k1.smpl_colored_{args.th}.ply')
    #     pred_path = osp.join(args.dataset_root, object_class, imgname, f'k1.smpl_colored_{args.th}.obj')
    #     if not osp.exists(pred_path):
    #         # print(pred_path)
    #         continue
    #     pred_mesh = trimesh.load(pred_path, process=False)
        
    #     pred = np.zeros((1, 6890))
    #     for i in range(pred_mesh.visual.vertex_colors.shape[0]):
    #         c = pred_mesh.visual.vertex_colors[i]
    #         if collections.Counter(c) == collections.Counter([255, 0, 0, 255]):
    #             pred[0, i] = 1
                
    #     pre, rec, f1 = precision_recall_f1score(gt, pred)
    #     fp_err, _ = det_error_metric(pred, gt)  
        
    #     tot_pre += pre.sum()
    #     tot_rec += rec.sum()
    #     tot_f1 += f1.sum()
    #     tot_fp_err += fp_err.numpy().sum()

    #     if f1.sum() > best_f1:
    #         best_pre = pre.sum()
    #         best_rec = rec.sum()
    #         best_f1 = f1.sum()
    #         best_fp_err = fp_err.numpy().sum()
    #         best_imgname = imgname
    #         best_obj = object_class
        
    #     img_count += 1   
        
    # print(f'Dataset size: {img_count}')
    # print(f'Threshold: {args.th}\n')
    
    # print(f'Test Precision: {tot_pre/img_count}')  
    # print(f'Test Recall: {tot_rec/img_count}') 
    # print(f'Test F1 Score: {tot_f1/img_count}') 
    # print(f'Test FP Error: {tot_fp_err/img_count}\n')   

    # print(f'Best Precision: {best_pre}')  
    # print(f'Best Recall: {best_rec}') 
    # print(f'Best F1 Score: {best_f1}') 
    # print(f'Best FP Error: {best_fp_err}')  
    # print(f'Best Image: {best_imgname} and Object: {best_obj}')           
       
        
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, default='/is/cluster/fast/achatterjee/CHORE_DECO/hot_test')
    parser.add_argument('--npz_file', type=str, default='/is/cluster/fast/achatterjee/Datasets/hot_behave/hot_behave_test_reduced.npz')
    parser.add_argument('--json_file', type=str, default='/ps/scratch/ps_shared/stripathi/deco/4agniv/hot_behave_split/imgnames_per_object_dict_reduced.json')
    parser.add_argument('--th', type=float)
    args = parser.parse_args()
    
    main(args)
        