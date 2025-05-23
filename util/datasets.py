import cv2
import torch
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import os

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
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        if self.transforms:
            image_tensor = self.transforms(image)
            mask = cv2.resize(mask, self.image_size)
        else:
            image_tensor = default_transforms(image)

        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        mask_tensor = torch.as_tensor(mask[None], dtype=torch.float32) / 255.

        # print(mask_tensor)
        # mask_tensor /= 255.
        # print(mask_tensor.min(), mask_tensor.max())

        return image_tensor, mask_tensor
    
    def collate_fn(self, batch):
        images, masks = tuple(zip(*batch))
        images = [img[None] for img in images]
        masks = [msk[None] for msk in masks]
        images, masks = [torch.cat(i).to(self.device) for i in [images, masks]]
        return images, masks

def load_data(rootPath = '', transforms = None, image_size = [448, 448], device = 'cpu', batch_size = 1, train = True, shuffle = True, drop_last = True):
    if train:
        path_images = glob(os.path.join(rootPath, 'train/images') + '/*.jpg')
        path_masks = glob(os.path.join(rootPath, 'train/masks') + '/*.jpg')
    else:
        path_images = glob(os.path.join(rootPath, 'test/images') + '/*.jpg')
        path_masks = glob(os.path.join(rootPath, 'test/masks') + '/*.jpg')
    
    path_images = sorted([str(p) for p in path_images])
    path_masks = sorted([str(p) for p in path_masks])

    data_df = pd.DataFrame({'images': path_images, 'masks': path_masks})
    data_set = CrackDataset(data_df, transforms = transforms, image_size=image_size, device = device)
    data_loader = DataLoader(data_set, batch_size = batch_size, collate_fn = data_set.collate_fn, shuffle = shuffle, drop_last = drop_last)

    return data_loader

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Currently using "{device}" device.')

    dataLoader = load_data('../../data/crack_segmentation_dataset', 
                           device = device, 
                           batch_size = 2, 
                           train = False, 
                           shuffle = False)
    for images, masks in dataLoader:
        plt.subplot(121)
        plt.imshow(images[1].cpu().detach().numpy().transpose(1,2,0))
        plt.title('/'.join(dataLoader.dataset.dataset.loc[1, 'images'].split('/')[-3:]))
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(masks[1].cpu().detach().numpy().transpose(1,2,0), cmap='gray')
        plt.title(os.path.join(*dataLoader.dataset.dataset.loc[1, 'masks'].split(os.sep)[-3:]))
        plt.axis('off')

        plt.show()
        break

    # import cv2
    # import matplotlib.pyplot as plt

    # path = '/home/swjtu/workspace_01/data/crack_segmentation_dataset/test/masks/CFD_001.jpg'
    # mask_raw = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # plt.imshow(mask_raw, cmap='gray')
    # plt.title("Raw Mask")
    # plt.axis('off')
    # plt.show()