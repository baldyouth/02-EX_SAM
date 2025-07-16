import torch

def calculate_iou(pred_logits, gt_mask, thresh):

    pred_prob = torch.sigmoid(pred_logits)
    pred_binary = (pred_prob > thresh).float()
    
    gt_binary = gt_mask.float()
    
    # 计算TP、TN、FP、FN
    TP = torch.sum((pred_binary == 1) & (gt_binary == 1)).item()
    TN = torch.sum((pred_binary == 0) & (gt_binary == 0)).item()
    FP = torch.sum((pred_binary == 1) & (gt_binary == 0)).item()
    FN = torch.sum((pred_binary == 0) & (gt_binary == 1)).item()
    
    # 计算IoU
    if (FN + FP + TP) <= 0:
        miou = 0
    else:
        iou_1 = TP / (FN + FP + TP)
        iou_0 = TN / (FN + FP + TN)
        miou = (iou_1 + iou_0) / 2
    
    return iou_0, iou_1, miou

import torch
import numpy as np

def calculate_best_iou(pred_logits, gt_mask, thresh_step=0.01):
    pred_prob = torch.sigmoid(pred_logits)
    gt_binary = gt_mask.float()

    best_thresh = 0.0
    best_iou_0 = 0.0
    best_iou_1 = 0.0
    best_miou = 0.0

    for thresh in np.arange(0.0, 1.0, thresh_step):
        pred_binary = (pred_prob > thresh).float()

        TP = torch.sum((pred_binary == 1) & (gt_binary == 1)).item()
        TN = torch.sum((pred_binary == 0) & (gt_binary == 0)).item()
        FP = torch.sum((pred_binary == 1) & (gt_binary == 0)).item()
        FN = torch.sum((pred_binary == 0) & (gt_binary == 1)).item()

        denom_1 = TP + FP + FN
        denom_0 = TN + FP + FN

        if denom_1 > 0:
            iou_1 = TP / denom_1
        else:
            iou_1 = 0.0

        if denom_0 > 0:
            iou_0 = TN / denom_0
        else:
            iou_0 = 0.0

        miou = (iou_0 + iou_1) / 2

        if miou > best_miou:
            best_miou = miou
            best_thresh = thresh
            best_iou_0 = iou_0
            best_iou_1 = iou_1

    return best_iou_0, best_iou_1, best_miou

# P, R, F1
def compute_f_measure(pred_mask, gt_mask):
    """
    pred_mask, gt_mask: torch.Tensor, dtype=bool/int/float, same shape
    """
    pred_mask = pred_mask.bool()
    gt_mask = gt_mask.bool()
    
    tp = torch.sum(pred_mask & gt_mask).item()
    fp = torch.sum(pred_mask & (~gt_mask)).item()
    fn = torch.sum((~pred_mask) & gt_mask).item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f_measure = 2 * precision * recall / (precision + recall + 1e-8)

    return precision, recall, f_measure

# ODS & OIS
def compute_ODS_OIS_single_sample(pred, gt, thresholds=torch.linspace(0.3, 0.8, 51)):
    """
    pred: [B, 1, H, W], float32, 0 or 1
    gt: [B, 1, H, W], float32, 0 or 1

    返回:
        best_f_ods: ODS F-measure
        best_thresh_ods: 对应ODS阈值
        best_f_ois: OIS F-measure
    """
    B = pred.shape[0]
    pred_probs_list = [pred[b, 0] for b in range(B)]  # [H, W]
    gt_list = [gt[b, 0] for b in range(B)]           # [H, W]

    # ODS
    best_f_ods = 0
    best_thresh_ods = 0

    for t in thresholds:
        tp = fp = fn = 0
        for pred_prob, gt_mask in zip(pred_probs_list, gt_list):
            pred_mask = (pred_prob > t)
            tp += torch.sum(pred_mask & (gt_mask == 1)).item()
            fp += torch.sum(pred_mask & (gt_mask == 0)).item()
            fn += torch.sum((~pred_mask) & (gt_mask == 1)).item()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f_measure = 2 * precision * recall / (precision + recall + 1e-8)

        if f_measure > best_f_ods:
            best_f_ods = f_measure
            best_thresh_ods = t.item()

    # OIS
    f_measures_ois = []
    for pred_prob, gt_mask in zip(pred_probs_list, gt_list):
        best_f = 0
        for t in thresholds:
            pred_mask = (pred_prob > t)
            _, _, f_measure = compute_f_measure(pred_mask, gt_mask)
            if f_measure > best_f:
                best_f = f_measure
        f_measures_ois.append(best_f)

    best_f_ois = sum(f_measures_ois) / len(f_measures_ois)

    return best_f_ods, best_thresh_ods, best_f_ois