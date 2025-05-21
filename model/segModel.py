import torch.nn as nn

from model.loadSAM import load_SAM_model
from model.encoder import encoder
from model.decoder import concatEncoder, fusionModule, segHead

class segModel(nn.Module):
    def __init__(self, modeName='base', device='cuda', dims=[32, 64, 128, 256], num_heads=[4, 4, 8, 8], num_classes=1, out_size=(448, 448)):
        super().__init__()
        self.sam = load_SAM_model(modeName=modeName, device=device)
        self.encoder = encoder(dims=dims, num_heads=num_heads)
        self.decoder = concatEncoder(in_c=[x * 2 for x in dims[1:]], out_c=dims[-1])
        self.fusion = fusionModule(dim=dims[-1], d_state=16, d_conv=4, expand=2, mamba_depth=2)
        self.seghead = segHead(fusion_channels = dims[-1], skip_channels=dims[-1], mid_channels=dims[-2], num_classes=num_classes, out_size=out_size)

        for param in self.sam.parameters():
            param.requires_grad = False
        self.sam.eval()

    def forward(self, img):
        (sam_f,) = self.sam(img)
        mynet_f = self.encoder(img) # return (x(0), x(1), x(3))

        concat_f = self.decoder(mynet_f)
        fusion_f = self.fusion(sam_f, concat_f)

        logits, mask = self.seghead(fusion_f, sam_f, concat_f)

        return logits, mask
