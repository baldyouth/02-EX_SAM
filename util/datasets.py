import cv2
import torch
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np
import torch.nn.functional as F

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

# default transform
default_transforms = T.Compose([
    T.ToTensor(), 
    T.Normalize(imagenet_mean, imagenet_std),
])

class CrackDataset(Dataset):
    def __init__(self, dataset, transforms=None, image_size=[448, 448], device='cpu'):
        self.dataset = dataset.reset_index(drop=True)
        self.transforms = transforms
        self.image_size = image_size
        self.device = device
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, ix):
        row = self.dataset.loc[ix].squeeze()
        image_path = row['images']
        mask_path = row['masks']

        # image
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        image_tensor = torch.as_tensor(image_np).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor

        # mask
        mask = Image.open(mask_path).convert("L")
        mask_np = np.array(mask)
        mask_np = (mask_np > 127).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)

        if self.transforms is not None:
            pass
        
        # image = cv2.imread(image_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # mask = cv2.imread(mask_path)
        # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        # if self.transforms:
        #     image_tensor = self.transforms(image)
        #     mask = cv2.resize(mask, self.image_size)
        # else:
        #     image_tensor = default_transforms(image)

        # _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # mask_tensor = torch.as_tensor(mask[None], dtype=torch.float32) / 255.

        # print(mask_tensor)
        # mask_tensor /= 255.
        # print(mask_tensor.min(), mask_tensor.max())

        return image_tensor, mask_tensor
    
    def collate_fn(self, batch):
        images, masks = tuple(zip(*batch))
        images = [img[None] for img in images]
        masks = [msk[None] for msk in masks]
        # images, masks = [torch.cat(i).to(self.device) for i in [images, masks]]
        images, masks = [torch.cat(i) for i in [images, masks]]
        return images, masks

class CrackDatasetwithCache(Dataset):
    def __init__(self, dataset, transforms=None, image_size=[448, 448]):
        self.dataset = dataset.reset_index(drop=True)
        self.transforms = transforms
        self.image_size = image_size

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, ix):
        row = self.dataset.loc[ix].squeeze()
        image_path = row['images']
        mask_path = row['masks']
        feature_path = row['features']

        # image
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        image_tensor = torch.as_tensor(image_np).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor

        # mask
        mask = Image.open(mask_path).convert("L")
        mask_np = np.array(mask)
        mask_np = (mask_np > 127).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)

        # feature
        feature_tensor = torch.load(feature_path).squeeze(0)

        if self.transforms is not None:
            pass

        return image_tensor, mask_tensor, feature_tensor

    def collate_fn(self, batch):
        images, masks, features = tuple(zip(*batch))
        images = [img[None] for img in images]
        masks = [msk[None] for msk in masks]
        features = [f[None] for f in features]
        images, masks, features = [torch.cat(x) for x in [images, masks, features]]
        return images, masks, features

# load_data
def load_data(rootPath = '', 
              transforms = None, 
              image_size = [448, 448],
              batch_size = 1, 
              train = True, 
              shuffle = True, 
              drop_last = True, 
              num_workers=4,
              pin_memory=True,
              is_cache=False):
    if train:
        path_images = glob(os.path.join(rootPath, 'train/images') + '/*.jpg')
        path_masks = glob(os.path.join(rootPath, 'train/masks') + '/*.jpg')

        if is_cache:
            path_features = glob(os.path.join(rootPath, 'train/features') + '/*.pt')
            path_features = sorted([str(p) for p in path_features])
    else:
        path_images = glob(os.path.join(rootPath, 'test/images') + '/*.jpg')
        path_masks = glob(os.path.join(rootPath, 'test/masks') + '/*.jpg')
    
    path_images = sorted([str(p) for p in path_images])
    path_masks = sorted([str(p) for p in path_masks])

    if is_cache:
        data_df = pd.DataFrame({'images': path_images, 'masks': path_masks, 'features': path_features})
        data_set = CrackDatasetwithCache(data_df, transforms = transforms, image_size=image_size)
    else:
        data_df = pd.DataFrame({'images': path_images, 'masks': path_masks})
        data_set = CrackDataset(data_df, transforms = transforms, image_size=image_size)
    
    data_loader = DataLoader(data_set, 
                             batch_size = batch_size, 
                             collate_fn = data_set.collate_fn, 
                             shuffle = shuffle, 
                             drop_last = drop_last, 
                             num_workers=num_workers, 
                             pin_memory=pin_memory)

    return data_loader

def smooth_feature(feature, target_size=(448, 448), sigma=2):
    # 双线性插值
    upsampled = F.interpolate(feature.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False)
    upsampled = upsampled.squeeze(0).cpu().numpy()  # [C, H, W]

    # 高斯滤波（对每个通道）
    smoothed = []
    for c in range(upsampled.shape[0]):
        blurred = cv2.GaussianBlur(upsampled[c], (0, 0), sigmaX=sigma)
        smoothed.append(blurred)

    smoothed = np.stack(smoothed)  # [C, H, W]
    return smoothed

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Currently using "{device}" device.')

    N = 1
    is_cache = True

    if is_cache:
        dataLoader = load_data('/home/swjtu/workspace_01/data/crack_segmentation_dataset', 
                           batch_size = 1, 
                           train = True, 
                           shuffle = False,
                           is_cache=True)
    else:
        dataLoader = load_data('/home/swjtu/workspace_01/data/crack_segmentation_dataset', 
                           batch_size = 1, 
                           train = True, 
                           shuffle = False,
                           is_cache=False)

    if is_cache:
        for i, (images, masks, features) in enumerate(dataLoader):
            img = images[0]
            mask = masks[0]
            feat = features[0]

            plt.subplot(131)
            plt.imshow(img.cpu().detach().numpy().transpose(1,2,0))
            plt.title('img')
            plt.axis('off')

            plt.subplot(132)
            plt.imshow(mask.cpu().detach().numpy().transpose(1,2,0), cmap='gray')
            plt.title('mask')
            plt.axis('off')

            feat = smooth_feature(feat, target_size=(448, 448), sigma=1.2)
            feat_max = feat.max(axis=0)
            # feat_mean = feat.mean(dim=0).cpu().numpy()
            # feat_mean -= feat_mean.min()
            # feat_mean /= feat_mean.max() + 1e-5

            plt.subplot(133)
            plt.imshow(feat_max)
            plt.title("Feature Heatmap (Mean)")
            plt.axis("off")

            plt.show()
            if i + 1 >= N:
                break
    else:
        for i, (images, masks) in enumerate(dataLoader):
            img = images[0]
            mask = masks[0]

            plt.subplot(121)
            plt.imshow(img.cpu().detach().numpy().transpose(1,2,0))
            plt.title('img')
            plt.axis('off')

            plt.subplot(122)
            plt.imshow(mask.cpu().detach().numpy().transpose(1,2,0), cmap='gray')
            plt.title('mask')
            plt.axis('off')

            plt.show()
            if i + 1 >= N:
                break