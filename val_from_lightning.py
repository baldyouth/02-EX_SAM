import os
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ruamel.yaml import YAML

from Lightning.lightning_module import LitModule
from Lightning.datamodule import CrackDataModule
from Lightning.valid import calculate_iou

def visualize_and_save(img, mask, preds, prob=None, save_path="vis", idx=0):

    # img: [B, 3, H, W] normalized
    # mask, preds: [B, 1, H, W]
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
        fig, axs = plt.subplots(1, 4 if prob is not None else 3, figsize=(20, 4))

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

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"vis_batch{idx+1}_{i+1}.png"))
        plt.close()

if __name__ == "__main__":
    yaml = YAML()
    with open('config/config_lightning.yaml', 'r') as f:
        config = yaml.load(f)

    model = LitModule.load_from_checkpoint(
        checkpoint_path="checkpoints/20250709_161406/best-epoch=139.ckpt",
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
    threshold = 0.48

    for idx, (img, mask) in enumerate(val_dataloader):
        img = img.cuda()
        mask = mask.cuda()

        with torch.no_grad():
            logits = model(img)
            prob = torch.sigmoid(logits)
            preds = (prob > threshold).float()

        visualize_and_save(
            img = img,
            mask = mask,
            preds = preds,
            prob = prob,
            save_path = save_dir,
            idx = idx
        )

        iou_0, iou_1, miou  = calculate_iou(logits, mask, thresh=threshold)
        print(f'[{idx+1}] 背景:{iou_0:.4f}, 缺陷:{iou_1:.4f}, 平均:{miou:.4f}')

        if (idx+1) >= save_num:
            break

    print('[DONE]')