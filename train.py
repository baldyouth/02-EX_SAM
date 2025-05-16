import torch
import os
import matplotlib.pyplot as plt
from ruamel.yaml import YAML

from model import *
from util.datasets import load_data
from util.train_fun import train_loop
from util.optimizers import get_optimizer, get_scheduler

yaml = YAML()
with open('config/config_01.yaml', 'r') as f:
    config = yaml.load(f)

device = config['model'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
print(f'Currently using "{device}" device.')

# data
dataLoader = load_data('../data/crack_segmentation_dataset', 
                        device = device, 
                        batch_size = 2, 
                        train = False, 
                        shuffle = False,
                        drop_last = True)

for idex_batch, batch in enumerate(dataLoader):
    (images, masks) = batch
    pass

# model
SAM_encoder = load_SAM_model('base', device = device)
VisionMamba_encoder = VisionMambaSeg().to(device)

# optimizer & lr_scheduler
optimizer = get_optimizer(SAM_encoder, config['training'])
total_steps = len(dataLoader) * config['training'].get('num_epochs', 100)
lr_scheduler = get_scheduler(optimizer, config['training'], total_steps)
pass
def Test01():
    model = VisionMambaSeg().to(device)
    print(model)
def Test02():
    x = torch.rand(1, 3, 448, 448, device=device)
    output = model(x)
    print(output[0].shape, output[1].shape, output[2].shape, output[3].shape)

    model = load_SAM_model('base', device = device)
    inputs = torch.rand(1, 3, 1024, 1024)
    out = model(inputs)
    print(type(out))
    # To extract features.
    feats = model.extract_feat(inputs)
    print(type(feats))

if __name__ == '__main__':
    Test01()
    pass
    