
from model.segModel import segModel

import torch

seg = segModel().to('cuda')
input = torch.rand((1, 3, 448, 448), device='cuda')
logits, mask = seg(input)
print(logits.shape, mask.shape)