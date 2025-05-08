import torch
import os
import matplotlib.pyplot as plt

from model import *
from util.datasets import load_data

import segmentation_models_pytorch as smp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Currently using "{device}" device.')

# data
dataLoader = load_data('../data/crack_segmentation_dataset', 
                        device = device, 
                        batch_size = 2, 
                        train = False, 
                        shuffle = False,
                        drop_last = True)

# model
SAM_encoder = load_SAM_model('base', device = device)
VisionMamba_encoder = VisionMambaSeg().to(device)

# loss
focal_loss = smp.losses.FocalLoss(mode='binary')
tversky_loss = smp.losses.TverskyLoss(mode='binary')

def Test():
    model = VisionMambaSeg().to(device)
    print(model)
    # x = torch.rand(1, 3, 448, 448, device=device)
    # output = model(x)
    # print(output[0].shape, output[1].shape, output[2].shape, output[3].shape)

    # model = load_SAM_model('base', device = device)
    # inputs = torch.rand(1, 3, 1024, 1024)
    # out = model(inputs)
    # print(type(out))
    # # To extract features.
    # feats = model.extract_feat(inputs)
    # print(type(feats))
    pass

if __name__ == '__main__':
    Test()
    pass
    