import numpy as np
import torch
import monai.metrics as metrics
from common.constants import DIST_MATRIX_PATH

DIST_MATRIX = np.load("../"+DIST_MATRIX_PATH)

def metric(mask, pred, back=True):
  iou = metrics.compute_meaniou(pred, mask, back, False)
  iou = iou.mean()

  return iou

def precision_recall_f1score(gt, pred):
    """
    Compute precision, recall, and f1
    """

    # gt = gt.numpy()
    # pred = pred.numpy()

    precision = torch.zeros(gt.shape[0])
    recall = torch.zeros(gt.shape[0])
    f1 = torch.zeros(gt.shape[0])
    
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

    # return precision, recall, f1
    return precision, recall, f1

def acc_precision_recall_f1score(gt, pred):
    """
    Compute acc, precision, recall, and f1
    """

    # gt = gt.numpy()
    # pred = pred.numpy()

    acc = torch.zeros(gt.shape[0])
    precision = torch.zeros(gt.shape[0])
    recall = torch.zeros(gt.shape[0])
    f1 = torch.zeros(gt.shape[0])

    for b in range(gt.shape[0]):
        tp_num = gt[b, pred[b, :] >= 0.5].sum()
        precision_denominator = (pred[b, :] >= 0.5).sum()
        recall_denominator = (gt[b, :]).sum()
        tn_num = gt.shape[-1] - precision_denominator - recall_denominator + tp_num

        acc_ = (tp_num + tn_num) / gt.shape[-1]
        precision_ = tp_num / (precision_denominator + 1e-10)
        recall_ = tp_num / (recall_denominator + 1e-10)
        f1_ = 2 * precision_ * recall_ / (precision_ + recall_ + 1e-10)

        acc[b] = acc_
        precision[b] = precision_
        recall[b] = recall_

    # return precision, recall, f1
    return acc, precision, recall, f1

def det_error_metric(pred, gt):
    
    gt = gt.detach().cpu()
    pred = pred.detach().cpu()

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