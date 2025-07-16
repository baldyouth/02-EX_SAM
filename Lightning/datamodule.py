import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pytorch_lightning as pl
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

class CrackDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, filter_keywords=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.filter_keywords = filter_keywords

        all_names = os.listdir(img_dir)
        if self.filter_keywords is not None:
            self.image_names = sorted([
                f for f in all_names if any(k in f for k in filter_keywords)
            ])
        else:
            self.image_names = sorted(all_names)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_names[idx])
        mask_path = os.path.join(self.mask_dir, self.image_names[idx])
        if not os.path.exists(mask_path):
            mask_path = os.path.join(self.mask_dir, self.image_names[idx].split('.')[0] + '.png')

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = (augmented['mask'] > 127).float().unsqueeze(0)
        else:
            pass

        return image, mask

class CrackDataModule(pl.LightningDataModule):
    def __init__(self, data_config):
        super().__init__()
        self.data_root = data_config['data_root']
        self.batch_size = data_config['batch_size']
        self.num_workers = data_config['num_workers']

        self.train_filter_keywords = data_config['train_filter_keywords']
        self.val_filter_keywords = data_config['val_filter_keywords']

        self.sam_transform = A.Compose([
            A.Resize(height=448, width=448),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]), # # Normalize 自动转成 float32 并除以255 → [0,1], 再减去均值，除以标准差
            ToTensorV2()
        ])

    def setup(self, stage=None):
        self.train_dataset = CrackDataset(
            img_dir=os.path.join(self.data_root, 'train/images'),
            mask_dir=os.path.join(self.data_root, 'train/masks'),
            transform=self.sam_transform,
            filter_keywords=self.train_filter_keywords
        )

        self.val_dataset = CrackDataset(
            img_dir=os.path.join(self.data_root, 'test/images'),
            mask_dir=os.path.join(self.data_root, 'test/masks'),
            transform=self.sam_transform,
            filter_keywords=self.val_filter_keywords
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)

if __name__ == '__main__':
    def denormalize(img_tensor, mean, std):
        img = img_tensor.clone().cpu()
        for t, m, s in zip(img, mean, std):
            t.mul_(s).add_(m)
        return img.permute(1, 2, 0).numpy().clip(0, 1)

    import matplotlib.pyplot as plt

    config = {
        'data_root': '/home/swjtu/workspace_01/data/DeepCrack',
        'batch_size': 1,
        'num_workers': 8,
        'train_filter_keywords': None,
        'val_filter_keywords': None
    }

    data_module = CrackDataModule(data_config=config)
    data_module.setup()
    dataset = data_module.train_dataset

    img, mask = dataset[1]
    print(img.shape, mask.shape)
    print(img.shape, img.dtype)
    print(mask.shape, mask.dtype)
    print(img.min(), img.max())
    print(mask.unique())

    img_np = denormalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    mask_np = mask.squeeze(0).numpy()

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img_np)  # [C,H,W] -> [H,W,C]
    plt.title("Augmented Image")

    plt.subplot(1, 2, 2)
    plt.imshow(mask_np, cmap='gray')
    plt.title("Augmented Mask")

    plt.tight_layout()
    plt.show()