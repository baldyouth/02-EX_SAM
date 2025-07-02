import torch

def miou_from_binary_preds(preds, target, num_classes=1):
    """
    输入：
        preds: 二值化预测结果，float32，shape [b,1,h,w]，值为0.0或1.0
        target: 二值化真实标签，float32，shape 同上，值为0.0或1.0
        num_classes: 类别数，默认为1（二分类）
    返回：
        mIoU: float 标量，平均交并比
    """
    pred_bin = preds.long()  # 转整型0/1
    target_bin = target.long()

    b = pred_bin.shape[0]
    ious = []

    for batch_i in range(b):
        p = pred_bin[batch_i].view(-1)
        t = target_bin[batch_i].view(-1)

        iou_per_class = []
        for l in range(num_classes + 1):
            pll = torch.sum((p == l) & (t == l)).item()
            plt = torch.sum((p == l) & (t != l)).item()
            ptl = torch.sum((p != l) & (t == l)).item()

            denom = plt + ptl + pll
            iou = 1.0 if denom == 0 else pll / denom
            iou_per_class.append(iou)

        ious.append(sum(iou_per_class) / (num_classes + 1))
        miou = sum(ious) / b

    return miou

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