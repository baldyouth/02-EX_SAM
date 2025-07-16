import torch

from ruamel.yaml import YAML
from Lightning.lightning_module import LitModule
from Lightning.datamodule import CrackDataModule
from Lightning.valid import calculate_iou

def precision_recall_f1(tp, fp, fn):
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return precision, recall, f1

if __name__ == "__main__":
    yaml = YAML()
    with open('config/config_lightning.yaml', 'r') as f:
        config = yaml.load(f)

    model = LitModule.load_from_checkpoint(
        checkpoint_path="checkpoints/20250714_203822/best-epoch=099.ckpt",
        model_config=config['model'],
        optimizer_config=config['optimizer'],
        scheduler_config=config['scheduler'])
    model.eval().cuda()

    data_module = CrackDataModule(data_config=config['data'])
    data_module.setup()
    val_dataloader = data_module.val_dataloader()

    thresholds = torch.arange(0, 1, 0.02).cuda()

    # =================== 缓存 prob 和 mask ===================
    prob_list = []
    mask_list = []
    for img, mask in val_dataloader:
        img = img.cuda()
        mask = mask.cuda()
        with torch.no_grad():
            logits = model(img)
            prob = torch.sigmoid(logits)
        prob_list.append(prob.cpu())
        mask_list.append(mask.cpu())
    print("✅ Prob & mask caching done.")

    # ================ OIS_F1 (每图独立最佳阈值) ================
    total_f1 = 0.0
    num_samples = 0

    for prob, mask in zip(prob_list, mask_list):
        p_flat = prob.view(1, -1).cuda()
        m_flat = mask.view(-1).cuda().bool()

        best_f1 = 0.0
        for thresh in thresholds:
            preds = (p_flat > thresh).view(-1).bool()
            tp = (preds & m_flat).sum().float()
            fp = (preds & (~m_flat)).sum().float()
            fn = ((~preds) & m_flat).sum().float()

            _, _, f1 = precision_recall_f1(tp, fp, fn)
            if f1 > best_f1:
                best_f1 = f1

        total_f1 += best_f1
        num_samples += 1

    ois_f1 = total_f1 / num_samples
    print(f"✅ OIS (每图最佳阈值): {ois_f1:.4f}")

    # ================ ODS (全集固定最佳阈值) ================
    best_ods_f1 = 0.0
    best_thresh = 0.0
    best_miou = 0.0
    best_iou_1 = 0.0
    best_precision = 0.0
    best_recall = 0.0

    for thresh in thresholds:
        total_tp, total_fp, total_fn = 0, 0, 0
        total_miou, total_iou_1 = 0.0, 0.0
        num_images = 0

        for prob, mask in zip(prob_list, mask_list):
            p = prob.cuda()
            m = mask.cuda()
            preds = (p > thresh).bool()
            mask_b = m.bool()

            tp = (preds & mask_b).sum().float()
            fp = (preds & (~mask_b)).sum().float()
            fn = ((~preds) & mask_b).sum().float()

            total_tp += tp
            total_fp += fp
            total_fn += fn

            with torch.no_grad():
                logits = torch.logit(p.clamp(1e-6, 1 - 1e-6))
                _, iou_1, miou = calculate_iou(logits, m, thresh=thresh.item())
                total_miou += miou
                total_iou_1 += iou_1
                num_images += 1

        precision, recall, f1 = precision_recall_f1(total_tp, total_fp, total_fn)
        avg_miou = total_miou / num_images
        avg_iou_1 = total_iou_1 / num_images

        if f1 > best_ods_f1:
            best_ods_f1 = f1
            best_thresh = thresh.item()
            best_miou = avg_miou
            best_iou_1 = avg_iou_1
            best_precision = precision.item()
            best_recall = recall.item()

    print(f"✅ ODS (全集固定最佳阈值):")
    print(f"最佳阈值: {best_thresh:.2f}")
    print(f"ODS: {best_ods_f1:.4f}")
    print(f"Precision: {best_precision:.4f}")
    print(f"Recall: {best_recall:.4f}")
    print(f"mIoU: {best_miou:.4f}")
    print(f"IoU_1: {best_iou_1:.4f}")