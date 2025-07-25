from model.img_encoder import ImgEncoder
from model.img_decoder import ImgDecoder

import torch
import torch.nn as nn
import torch.nn.functional as F

class ImgModel(nn.Module):
    def __init__(self, model_config, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.encoder = ImgEncoder(model_config)
        self.decoder = ImgDecoder()

    def forward(self, x):
        feature = self.encoder(x)
        out = self.decoder(feature)

        return out

if __name__ == '__main__':
    input = torch.rand((1, 3, 448, 448), device='cuda')

    model = ImgModel().to('cuda')
    output = model(input)

    print(output.shape)