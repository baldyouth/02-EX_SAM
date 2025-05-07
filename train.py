import torch
from mmpretrain import get_model
import os
import matplotlib.pyplot as plt

from model import *
from util.datasets import load_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Currently using "{device}" device.')

def load_SAM_model(modeName = 'base', pretrain = True, device = 'cpu'):
    SAM = None
    match modeName.strip().lower():
        case 'base':
            SAM = get_model('vit-base-p16_sam-pre_3rdparty_sa1b-1024px', pretrained = pretrain, device = device)
        case 'large':
            SAM = get_model('vit-large-p16_sam-pre_3rdparty_sa1b-1024px', pretrained = pretrain, device = device)
        case 'huge':
            SAM = get_model('vit-huge-p16_sam-pre_3rdparty_sa1b-1024px', pretrained = pretrain, device = device)
    return SAM

dataLoader = load_data('../data/crack_segmentation_dataset', 
                           device = device, 
                           batch_size = 2, 
                           train = False, 
                           shuffle = False,
                           drop_last = True)

SAM_encoder = load_SAM_model('base', device = device)
VisionMamba_encoder = VisionMambaSeg().to(device)


# inputs = torch.rand(1, 3, 448, 448, device = device)
# out = SAM(inputs)
# feats = SAM.extract_feat(inputs)
# print("out type:", type(out), "feats type:", type(feats), "out device", out[0].device)
# print("值是否一致：", torch.allclose(out[0], feats[0], atol=1e-6), torch.equal(out[0], feats[0]))

if __name__ == '__main__':
    pass
    # model = VisionMambaSeg().to('cuda')
    # # print(model)
    # x = torch.rand(1, 3, 448, 448, device='cuda')
    # output = model(x)
    # print(output[0].shape, output[1].shape, output[2].shape, output[3].shape)

    # model = get_model('vit-base-p16_sam-pre_3rdparty_sa1b-1024px', pretrained=True)
    # inputs = torch.rand(1, 3, 1024, 1024)
    # out = model(inputs)
    # print(type(out))
    # # To extract features.
    # feats = model.extract_feat(inputs)
    # print(type(feats))