import os
import torch
import cv2
from tqdm import tqdm
from mmpretrain import get_model
from torchvision import transforms
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_tensor
from PIL import Image
import torch.nn.functional as F

# 预处理
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std),
])

# 路径
image_dir = '/home/swjtu/workspace_01/data/crack_segmentation_dataset/train/images'
# feature_save_dir = '/home/swjtu/workspace_01/data/crack_segmentation_dataset/train/features'
# os.makedirs(feature_save_dir, exist_ok=True)

# 模型准备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sam = get_model(
            'vit-large-p16_sam-pre_3rdparty_sa1b-1024px',
            backbone=dict(patch_size=8, out_indices=11, out_channels=256), # out_indices=(2, 5, 8, 11)
            pretrained=True, 
            device=device)

image_paths = sorted(glob(os.path.join(image_dir, '*.jpg')))
for path in tqdm(image_paths):
    path = "/home/swjtu/workspace_01/data/crack_segmentation_dataset/images/DeepCrack_11179-3.jpg"
    image = Image.open(path).convert("RGB")
    image_np = np.array(image)
    tensor_img = torch.as_tensor(image_np).permute(2, 0, 1).float() / 255.0
    tensor_img = tensor_img.unsqueeze(0).to(device)

    with torch.no_grad():
        (feat,) = sam(tensor_img)

    feat_map = feat.squeeze(0).mean(dim=0).cpu().numpy()
    feat_map -= feat_map.min()
    feat_map /= feat_map.max() + 1e-5

    plt.subplot(121)
    plt.imshow(feat_map)
    plt.title("Feature Heatmap")
    plt.axis('off')

    feat = F.interpolate(feat, size=(448, 448), mode='bilinear')
    feat_map = feat.squeeze(0).mean(dim=0).cpu().numpy()
    feat_map -= feat_map.min()
    feat_map /= feat_map.max() + 1e-5

    plt.subplot(122)
    plt.imshow(feat_map)
    plt.title("Feature Heatmap(interpolate)")
    plt.axis('off')

    plt.show()

    basename = os.path.splitext(os.path.basename(path))[0]
    # torch.save(feat.cpu(), os.path.join(feature_save_dir, f'{basename}.pt'))
