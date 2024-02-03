import argparse
import torch
import pickle as pkl
import numpy as np
from utils.metrics import metric, precision_recall_f1score, det_error_metric
def evaluate(pred_dict, gt_dict):
    # combine keys from pred and gt in the same key order
    pred_keys = list(pred_dict.keys())
    gt_keys = list(gt_dict.keys())
    assert len(pred_keys) == len(gt_keys)
    assert pred_keys == gt_keys

    # get the contact labels
    contact_labels_3d_pred = []
    contact_labels_3d = []
    for key in pred_keys:
        contact_labels_3d_pred.append(pred_dict[key]['gen_contact_vids'])
        contact_labels_3d.append(gt_dict[key]['gen_contact_vids'])
    contact_labels_3d_pred = torch.FloatTensor(contact_labels_3d_pred)
    contact_labels_3d = torch.FloatTensor(contact_labels_3d)
    # get the semantic contact labels
    pass

    cont_pre, cont_rec, cont_f1 = precision_recall_f1score(contact_labels_3d, contact_labels_3d_pred)
    fp_geo_err, fn_geo_err = det_error_metric(contact_labels_3d_pred, contact_labels_3d)

    # results dict
    result_dict = {}
    result_dict['cont_precision'] = torch.mean(cont_pre).numpy()
    result_dict['cont_recall'] = torch.mean(cont_rec).numpy()
    result_dict['cont_f1'] = torch.mean(cont_f1).numpy()
    result_dict['fp_geo_err'] = torch.mean(fp_geo_err).numpy()
    # result_dict['fn_geo_err'] = torch.mean(fn_geo_err).numpy()

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