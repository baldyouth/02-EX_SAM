import os
import torch
import matplotlib.pyplot as plt
from ruamel.yaml import YAML
from torchmetrics.classification import BinaryJaccardIndex

from Lightning.lightning_module import LitModule
from Lightning.datamodule import CrackDataModule


def visualize_and_save(img_tensor, mask_tensor, pred_tensor, save_path=None, show=True):
    """
    可视化输入图像、真实mask和预测mask，并保存到文件（可选）。

    参数:
    - img_tensor: Tensor, 形状[C,H,W]，已归一化，需要反归一化显示
    - mask_tensor: Tensor, 形状[H,W]，真实掩码
    - pred_tensor: Tensor, 形状[H,W]，预测掩码（0/1浮点）
    - save_path: str or None, 如果不为None，则保存图片到该路径
    - show: bool, 是否用 plt.show() 显示图像

    反归一化使用 ImageNet 均值和方差
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    img_np = img_tensor.cpu().permute(1, 2, 0).numpy()
    img_np = (img_np * std) + mean
    img_np = img_np.clip(0, 1)

    mask_np = mask_tensor.cpu().numpy()
    pred_np = pred_tensor.cpu().numpy()

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(img_np)
    plt.title("Input Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(mask_np, cmap='gray')
    plt.title("Ground Truth")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(pred_np, cmap='gray')
    plt.title("Prediction")
    plt.axis('off')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        if show:
            plt.show()
        else:
            plt.close()

# MIOU
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

            print(f'iou({l}):{iou:.4f}')

        ious.append(sum(iou_per_class) / (num_classes + 1))
        miou = sum(ious) / b
        print(f'miou:{miou:.4f}')

    return miou

# precision, recall, f_measure
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

# ODS&OIS for sample
def compute_ODS_OIS_single_sample(pred, gt, thresholds=torch.linspace(0.5, 1, 10)):
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


if __name__ == "__main__":
    yaml = YAML()
    with open('config/config_lightning.yaml', 'r') as f:
        config = yaml.load(f)

    model = LitModule.load_from_checkpoint(
        checkpoint_path="checkpoints/20250702_205215/best-epoch=079.ckpt",
        model_config=config['model'],
        optimizer_config=config['optimizer'],
        scheduler_config=config['scheduler'])
    model.eval()
    model.cuda()

    data_module = CrackDataModule(data_config=config['data'])
    data_module.setup()
    val_dataloader = data_module.val_dataloader()

    save_dir = "./visualizations"
    os.makedirs(save_dir, exist_ok=True)
    save_num = 10
    threshold = 0.6

    for idx, (img, mask) in enumerate(val_dataloader):
        img = img.cuda()
        mask = mask.cuda()

        with torch.no_grad():
            logits = model(img)
            logits_sigmoid = torch.sigmoid(logits)
            preds = (logits_sigmoid > threshold).float()

        visualize_and_save(
            img_tensor=img[0],
            mask_tensor=mask[0].squeeze(0),
            pred_tensor=preds[0].squeeze(0),
            save_path=os.path.join(save_dir, f"visualize_{idx+1}.png"),
            show=False
        )

        miou = miou_from_binary_preds(preds, mask, num_classes=1)
        best_f_ods, best_thresh_ods, best_f_ois = compute_ODS_OIS_single_sample(logits_sigmoid, mask)
        print(f'best_f_ods:{best_f_ods:.4f}, best_thresh_ods:{best_thresh_ods:.4f}, best_f_ois:{best_f_ois:.4f}')
        print('-'*10)

        if (idx+1) >= save_num:
            break

    print('[DONE]')