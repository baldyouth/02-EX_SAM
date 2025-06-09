import torch
from ruamel.yaml import YAML
import random
import numpy as np
import os
import math

from model import *
from util.datasets import load_data
from util.train_fun import train_loop
from util.optimizers import get_optimizer, get_scheduler

from model.img_model import ImgModel

def set_seed(seed: int = 1):
    random.seed(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

yaml = YAML()
with open('config/config_01.yaml', 'r') as f:
    config = yaml.load(f)

device = config['model'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
print(f'Currently using "{device}" device.')

# !!! ====== data ======
train_loader = load_data(config["dataset"]["root_path"], 
                        transforms = None,
                        image_size=config["dataset"]["size"], 
                        batch_size = config["dataset"]["batch_size"], 
                        train = True, 
                        shuffle = True,
                        drop_last = True,
                        num_workers=config["dataset"]["num_workers"],
                        pin_memory=config["dataset"]["pin_memory"])

test_loader = load_data(config["dataset"]["root_path"], 
                        transforms = None,
                        image_size=config["dataset"]["size"], 
                        batch_size = config["dataset"]["batch_size"], 
                        train = False, 
                        shuffle = False,
                        drop_last = True,
                        num_workers=1)

# total_samples = len(dataLoader.dataset)
# print(f"Total samples: {total_samples}")

# !!! ====== model ======
set_seed(config["training"]["seed"])
model = ImgModel(device=device)

# print(model)

# !!! ====== optimizer & lr_scheduler ======
optimizer = get_optimizer(model, config['training'])
total_steps = math.ceil(len(train_loader) * config['training']['num_epochs'] / config['training']['gradient_accumulation_steps'])
lr_scheduler = get_scheduler(optimizer, config['training'], total_steps)

# !!! ====== strat train ======
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"[MODEL PARAMS] Total: {total_params:,}, Trainable: {trainable_params:,}")

train_loop(config, 
           model, 
           optimizer = optimizer, 
           train_dataloader = train_loader, 
           val_dataloader = test_loader, 
           lr_scheduler = lr_scheduler, 
           resume = False)

# pass
# def Test01():
#     model = VisionMambaSeg().to(device)
#     print(model)
# def Test02():
#     x = torch.rand(1, 3, 448, 448, device=device)
#     output = model(x)
#     print(output[0].shape, output[1].shape, output[2].shape, output[3].shape)

#     model = load_SAM_model('base', device = device)
#     inputs = torch.rand(1, 3, 1024, 1024)
#     out = model(inputs)
#     print(type(out))
#     # To extract features.
#     feats = model.extract_feat(inputs)
#     print(type(feats))

# if __name__ == '__main__':
#     Test01()
#     pass
    