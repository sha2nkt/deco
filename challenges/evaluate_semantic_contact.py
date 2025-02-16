import argparse
import torch
import pickle as pkl
import numpy as np
from utils.metrics import metric, precision_recall_f1score, det_error_metric
from common.constants import object_names, categorization_unordered
from tqdm import tqdm

def safe_mean(lst):
    return np.mean(lst) if lst else 0.0

def evaluate(pred_dict, gt_dict):
    # combine keys from pred and gt in the same key order
    pred_keys = list(pred_dict.keys())
    gt_keys = list(gt_dict.keys())
    assert len(pred_keys) == len(gt_keys)
    assert pred_keys == gt_keys

    val_epoch_cont_pre = {obj_name: [] for obj_name in object_names}
    val_epoch_cont_rec = {obj_name: [] for obj_name in object_names}
    val_epoch_cont_f1 = {obj_name: [] for obj_name in object_names}
    val_epoch_fp_geo_err = {obj_name: [] for obj_name in object_names}
    val_epoch_fn_geo_err = {obj_name: [] for obj_name in object_names}
    val_epoch_obj_count = {obj_name: 0 for obj_name in object_names}
    for key in tqdm(pred_keys):
        pred = pred_dict[key]['sem_contact_vids']
        gt = gt_dict[key]['sem_contact_vids']
        for obj_name, gt_vid in gt.items():
            # convert gt_vid to [1,6890] tensor
            gt_contact = torch.zeros(1, 6890)
            gt_contact[0, gt_vid] = 1
            # get the pred_vid
            if obj_name in pred:
                pred_vid = pred[obj_name]
                # convert pred_vid to [1,6890] tensor
                pred_contact = torch.zeros(1, 6890)
                pred_contact[0, pred_vid] = 1
            else:
                pred_contact = torch.zeros(1, 6890)
            cont_pre, cont_rec, cont_f1 = precision_recall_f1score(gt_contact, pred_contact)
            fp_geo_err, fn_geo_err = det_error_metric(pred_contact, gt_contact)
            val_epoch_cont_pre[obj_name].append(cont_pre.cpu().numpy())
            val_epoch_cont_rec[obj_name].append(cont_rec.cpu().numpy())
            val_epoch_cont_f1[obj_name].append(cont_f1.cpu().numpy())
            val_epoch_fp_geo_err[obj_name].append(fp_geo_err.cpu().numpy())
            val_epoch_fn_geo_err[obj_name].append(fn_geo_err.cpu().numpy())
            val_epoch_obj_count[obj_name] += 1


    # get average metrics weighted by the number of objects
    result_dict = {}
    total_obj_count = sum(val_epoch_obj_count.values())

    result_dict['cont_precision'] = sum(safe_mean(val_epoch_cont_pre[obj]) * val_epoch_obj_count[obj]
                        for obj in object_names) / total_obj_count
    result_dict['cont_recall'] = sum(safe_mean(val_epoch_cont_rec[obj]) * val_epoch_obj_count[obj]
                        for obj in object_names) / total_obj_count
    result_dict['cont_f1'] = sum(safe_mean(val_epoch_cont_f1[obj]) * val_epoch_obj_count[obj]
                       for obj in object_names) / total_obj_count
    result_dict['fp_geo_err'] = sum(safe_mean(val_epoch_fp_geo_err[obj]) * val_epoch_obj_count[obj]
                          for obj in object_names) / total_obj_count
    # result_dict['fn_geo_err'] = sum(safe_mean(val_epoch_fn_geo_err[obj]) * val_epoch_obj_count[obj]
    #                               for obj in object_names) / total_obj_count

    # combine all object_wise metrics using categorization_unordered
    for cat_name, cat_list in categorization_unordered.items():
        cat_obj_count = sum(val_epoch_obj_count[obj] for obj in cat_list)
        result_dict[cat_name + '_precision'] = sum(safe_mean(val_epoch_cont_pre[obj]) * val_epoch_obj_count[obj]
                        for obj in cat_list) / cat_obj_count
        result_dict[cat_name + '_recall'] = sum(safe_mean(val_epoch_cont_rec[obj]) * val_epoch_obj_count[obj]
                        for obj in cat_list) / cat_obj_count
        result_dict[cat_name + '_f1'] = sum(safe_mean(val_epoch_cont_f1[obj]) * val_epoch_obj_count[obj]
                          for obj in cat_list) / cat_obj_count
        result_dict[cat_name + '_fp_geo_err'] = sum(safe_mean(val_epoch_fp_geo_err[obj]) * val_epoch_obj_count[obj]
                            for obj in cat_list) / cat_obj_count
        # result_dict[cat_name + '_fn_geo_err'] = sum(safe_mean(val_epoch_fn_geo_err[obj]) * val_epoch_obj_count[obj]
        #                     for obj in cat_obj_list) / cat_obj_count


        # print the results dict
    for key in result_dict.keys():
        print(key, ': ', result_dict[key])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_pkl', type=str, required=True, help='path to pred pkl file')
    parser.add_argument('--gt_pkl', type=str, required=True, help='path to gt pkl file')
    args = parser.parse_args()

    with open(args.pred_pkl, 'rb') as f:
        pred_dict = pkl.load(f)

    with open(args.gt_pkl, 'rb') as f:
        gt_dict = pkl.load(f)

    evaluate(pred_dict, gt_dict)