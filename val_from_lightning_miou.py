import torch
import numpy as np
from ruamel.yaml import YAML
from Lightning.lightning_module import LitModule
from Lightning.datamodule import CrackDataModule
from Lightning.valid import calculate_iou

# ========== 加载配置和模型 ==========
yaml = YAML()
with open('config/config_lightning.yaml', 'r') as f:
    config = yaml.load(f)

model = LitModule.load_from_checkpoint(
    checkpoint_path="checkpoints/20250710_201652/best-epoch=159.ckpt",
    model_config=config['model'],
    optimizer_config=config['optimizer'],
    scheduler_config=config['scheduler'])
model.eval().cuda()

# ========== 加载验证集 ==========
data_module = CrackDataModule(data_config=config['data'])
data_module.setup()
val_dataloader = data_module.val_dataloader()

# ========== 缓存 prob_list_np, mask_list_np ==========
pred_list_np = []
gt_list_np = []

with torch.no_grad():
    for img, mask in val_dataloader:
        img = img.cuda()
        mask = mask.cuda()

        logits = model(img)
        prob = torch.sigmoid(logits)

        prob_uint8 = (prob.cpu().numpy() * 255).astype('uint8')
        mask_uint8 = (mask.cpu().numpy() * 255).astype('uint8')

        pred_list_np.append(prob_uint8.squeeze())  # [H,W]
        gt_list_np.append(mask_uint8.squeeze())    # [H,W]

print("✅ [缓存完成] 已获取 pred_list_np 和 gt_list_np.")
def cal_mIoU_metrics(pred_list, gt_list, thresh_step=0.01, pred_imgs_names=None, gt_imgs_names=None):
    final_iou = []
    for thresh in np.arange(0.0, 1.0, thresh_step):
        iou_list = []
        for i, (pred, gt) in enumerate(zip(pred_list, gt_list)):
            gt_img = (gt / 255).astype('uint8')
            pred_img = (pred / 255 > thresh).astype('uint8')
            TP = np.sum((pred_img == 1) & (gt_img == 1))
            TN = np.sum((pred_img == 0) & (gt_img == 0))
            FP = np.sum((pred_img == 1) & (gt_img == 0))
            FN = np.sum((pred_img == 0) & (gt_img == 1))
            if (FN + FP + TP) <= 0:
                iou = 0
            else:
                iou_1 = TP / (FN + FP + TP)
                iou_0 = TN / (FN + FP + TN)
                iou = (iou_1 + iou_0)/2
            iou_list.append(iou)
        ave_iou = np.mean(np.array(iou_list))
        final_iou.append(ave_iou)
    mIoU = np.max(np.array(final_iou))
    return mIoU

miou_from_function = cal_mIoU_metrics(pred_list_np, gt_list_np, thresh_step=0.02)
print(f"✅ [cal_mIoU_metrics结果] mIoU (max over thresholds): {miou_from_function:.4f}")
