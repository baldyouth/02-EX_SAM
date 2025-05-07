import torch
import numpy as np
from PIL import Image as PILImage

def tensor_2_PIL(data: torch.Tensor) -> PILImage.Image:
    data = data.cpu().detach().numpy().transpose((1, 2, 0)) * 255.0
    data = data.astype(np.uint8)
    return PILImage.fromarray(data)