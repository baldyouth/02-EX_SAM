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


if __name__ == "__main__":
    yaml = YAML()
    with open('config/config_lightning.yaml', 'r') as f:
        config = yaml.load(f)

    model = LitModule.load_from_checkpoint(
        checkpoint_path="checkpoints/20250626_151041/best-epoch=279.ckpt",
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
    threshold = 0.5

    jaccard_metric = BinaryJaccardIndex().to('cuda')
    total_iou = 0
    iou_count = 0

    for idx, (img, mask) in enumerate(val_dataloader):
        img = img.cuda()
        mask = mask.cuda()

        with torch.no_grad():
            logits = model(img)
            preds = torch.sigmoid(logits)
            preds = (preds > threshold).float()
        
        iou = jaccard_metric(preds, mask)
        print(f'{idx+1} IoU: {iou.item():.4f}')
        total_iou += iou.item()
        iou_count += 1

        visualize_and_save(
            img_tensor=img[0],
            mask_tensor=mask[0].squeeze(0),
            pred_tensor=preds[0].squeeze(0),
            save_path=os.path.join(save_dir, f"visualize_{idx+1}.png"),
            show=False
        )

        if (idx+1) >= save_num:
            break
    
    avg_iou = total_iou / iou_count if iou_count > 0 else 0
    print(f"Average IoU: {avg_iou:.4f}")

    print('[DONE]')