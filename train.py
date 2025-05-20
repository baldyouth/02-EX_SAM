import torch
import os
import matplotlib.pyplot as plt
from ruamel.yaml import YAML

from model import *
from util.datasets import load_data
from util.train_fun import train_loop
from util.optimizers import get_optimizer, get_scheduler

from model.segModel import segModel

yaml = YAML()
with open('config/config_01.yaml', 'r') as f:
    config = yaml.load(f)

device = config['model'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
print(f'Currently using "{device}" device.')

# !!! ====== data ======
train_loader = load_data(config["dataset"]["root_path"], 
                        transforms = None,
                        device = device, 
                        batch_size = config["dataset"]["batch_size"], 
                        train = True, 
                        shuffle = True,
                        drop_last = True)
test_loader = load_data(config["dataset"]["root_path"], 
                        transforms = None,
                        device = device, 
                        batch_size = config["dataset"]["batch_size"], 
                        train = False, 
                        shuffle = False,
                        drop_last = True)

# total_samples = len(dataLoader.dataset)
# print(f"Total samples: {total_samples}")

# !!! ====== model ======
model = segModel(modeName='base', 
                device='cuda', 
                dims=config["model"]["dims"], 
                num_heads=config["model"]["num_heads"], 
                num_classes=config["model"]["num_classes"], 
                out_size=config["dataset"]["size"]).to(device)

# print(model)

# !!! ====== optimizer & lr_scheduler ======
optimizer = get_optimizer(model, config['training'])
total_steps = len(train_loader) * config['training'].get('num_epochs', 100)
lr_scheduler = get_scheduler(optimizer, config['training'], total_steps)

# !!! ====== strat train ======
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
    