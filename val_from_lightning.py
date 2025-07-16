import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ruamel.yaml import YAML

from Lightning.lightning_module import LitModule
from Lightning.datamodule import CrackDataModule
from Lightning.valid import calculate_iou

def visualize_and_save(img, mask, preds, prob=None, save_path="vis", idx=0):
    os.makedirs(save_path, exist_ok=True)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    img_np = img.cpu().permute(0, 2, 3, 1).numpy()
    img_np = (img_np * std) + mean
    img_np = img_np.clip(0, 1)

    mask_np = mask.cpu().squeeze(1).numpy()
    preds_np = preds.cpu().squeeze(1).numpy()
    if prob is not None:
        prob_np = prob.cpu().squeeze(1).numpy()

    batch_size = img.shape[0]
    for i in range(batch_size):
        n_cols = 5 if prob is not None else 4
        fig, axs = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

        axs[0].imshow(img_np[i])
        axs[0].set_title("Input Image")
        axs[0].axis("off")

        axs[1].imshow(mask_np[i], cmap='gray')
        axs[1].set_title("Ground Truth")
        axs[1].axis("off")

        axs[2].imshow(preds_np[i], cmap='gray')
        axs[2].set_title("Prediction")
        axs[2].axis("off")

        if prob is not None:
            im = axs[3].imshow(prob_np[i], cmap='jet')
            axs[3].set_title("Prob Heatmap")
            axs[3].axis("off")
            fig.colorbar(im, ax=axs[3], fraction=0.046, pad=0.04)

        # 差异图
        diff_map = np.zeros((*mask_np[i].shape, 3), dtype=np.float32)
        tp = (mask_np[i] == 1) & (preds_np[i] == 1)
        fp = (mask_np[i] == 0) & (preds_np[i] == 1)
        fn = (mask_np[i] == 1) & (preds_np[i] == 0)

        print(f"[{idx+1}] tp: {tp.sum()} | fp:{fp.sum()} | fn:{fn.sum()}")

        diff_map[tp] = [0, 1, 0]      # green: TP
        diff_map[fp] = [1, 0, 0]      # red: FP
        diff_map[fn] = [0, 0, 1]      # blue: FN

        axs[-1].imshow(diff_map)
        axs[-1].set_title("Difference Map\nGreen:TP | Red:FP | Blue:FN")
        axs[-1].axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"vis_batch{idx+1}_{i+1}.png"), dpi=300)
        plt.close()

if __name__ == "__main__":
    yaml = YAML()
    with open('config/config_lightning_56.yaml', 'r') as f:
        config = yaml.load(f)

    model = LitModule.load_from_checkpoint(
        checkpoint_path="checkpoints/20250716_210130/epoch=039.ckpt",
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
    save_num = 20

    threshold_list = np.arange(0.0, 1.0, 0.02)

    for idx, (img, mask) in enumerate(val_dataloader):
        img = img.cuda()
        mask = mask.cuda()

        with torch.no_grad():
            logits = model(img)
            prob = torch.sigmoid(logits)

        best_miou = -1
        best_thresh = None
        best_preds = None
        best_iou_0 = None
        best_iou_1 = None

        for thresh in threshold_list:
            preds = (prob > thresh).float()
            iou_0, iou_1, miou = calculate_iou(logits, mask, thresh=thresh)
            if miou > best_miou:
                best_miou = miou
                best_thresh = thresh
                best_preds = preds.clone()
                best_iou_0 = iou_0
                best_iou_1 = iou_1

        print(f"[{idx+1}] 最佳阈值: {best_thresh:.3f} | 背景:{best_iou_0:.4f}, 缺陷:{best_iou_1:.4f}, 平均:{best_miou:.4f}")

        visualize_and_save(
            img = img,
            mask = mask,
            preds = best_preds,
            prob = prob,
            save_path = save_dir,
            idx = idx
        )

        if (idx+1) >= save_num:
            break

    print('[DONE]')