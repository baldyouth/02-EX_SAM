# from mmpretrain import get_model
# from peft import get_peft_model, LoraConfig, TaskType

# def load_SAM_model(modeName='base', pretrain=True, device='cpu', use_lora=True, lora_r=4, lora_alpha=1.0, lora_dropout=0.0):
#     match modeName.strip().lower():
#         case 'base':
#             SAM = get_model('vit-base-p16_sam-pre_3rdparty_sa1b-1024px', pretrained=pretrain, device=device)
#         case 'large':
#             SAM = get_model('vit-large-p16_sam-pre_3rdparty_sa1b-1024px', pretrained=pretrain, device=device)
#         case 'huge':
#             SAM = get_model('vit-huge-p16_sam-pre_3rdparty_sa1b-1024px', pretrained=pretrain, device=device)
#         case _:
#             raise ValueError(f'Unsupported modeName {modeName}')
        
#     lora_config = LoraConfig(
#         task_type=TaskType.FEATURE_EXTRACTION,
#         r=lora_r,
#         lora_alpha=lora_alpha,
#         target_modules = ["attn.qkv", "attn.proj", "ffn.layers.0.0", "ffn.layers.1"], 
#         lora_dropout=lora_dropout,
#         bias="none",
#         modules_to_save=None,
#     )

#     SAM = get_peft_model(SAM, lora_config)

#     for name, param in SAM.named_parameters():
#         if "lora_" not in name:
#             param.requires_grad = False
#         else:
#             param.requires_grad = True

#     return SAM

import torch
import torch.nn as nn
import math
from sam import ImageEncoderViT
from mmpretrain import get_model

class LoRALinear(nn.Linear):
    def __init__(self, in_features, out_features, r=4, lora_alpha=1.0, lora_dropout=0.0, bias=True):
        super().__init__(in_features, out_features, bias=bias)
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r if r > 0 else 1
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()

        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros((r, in_features)))
            self.lora_B = nn.Parameter(torch.zeros((out_features, r)))
            
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.lora_A = None
            self.lora_B = None

    def forward(self, x):
        result = super().forward(x)
        if self.r > 0:
            lora_out = self.lora_dropout(x) @ self.lora_A.T
            lora_out = lora_out @ self.lora_B.T
            result = result + lora_out * self.scaling
        return result


def inject_lora_to_vitsam(model, target_keywords=('qkv', 'proj', 'ffn.layers.0.0', 'ffn.layers.1'), r=4, lora_alpha=1.0, lora_dropout=0.0):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(k in name for k in target_keywords):
            parent = model
            parts = name.split('.')
            for p in parts[:-1]:
                parent = getattr(parent, p)
            last_part = parts[-1]
            old_linear = getattr(parent, last_part)

            lora_linear = LoRALinear(
                in_features=old_linear.in_features,
                out_features=old_linear.out_features,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias=old_linear.bias is not None
            )
            
            lora_linear.weight.data = old_linear.weight.data.clone()
            if old_linear.bias is not None:
                lora_linear.bias.data = old_linear.bias.data.clone()

            setattr(parent, last_part, lora_linear)


def load_SAM_model(modeName='base', pretrain=True, device='cuda', use_lora=True, lora_r=4, lora_alpha=1.0, lora_dropout=0.0):
    
    match modeName.strip().lower():
        case 'base':
            SAM = ImageEncoderViT(use_rel_pos=True, window_size=14, global_attn_indexes=[2, 5, 8, 11])
            state_dict = torch.load("/home/swjtu/workspace_01/02-EX_SAM/checkpoints_sam/sam_vit_b_01ec64.pth")
        case 'large':
            SAM = ImageEncoderViT(use_rel_pos=True, window_size=14, global_attn_indexes=[2, 5, 8, 11]) #TODO
        case 'huge':
            SAM = ImageEncoderViT(use_rel_pos=True, window_size=14, global_attn_indexes=[2, 5, 8, 11]) #TODO
        case _:
            raise ValueError(f'Unsupported modeName {modeName}')

    filtered_image_dict = {
        k.replace('image_encoder.', '', 1): v
        for k, v in state_dict.items()
        if k.startswith('image_encoder.')
    }
    SAM.load_state_dict(filtered_image_dict, strict=True)

    # if use_lora:
    #     inject_lora_to_vitsam(
    #         SAM.backbone,
    #         target_keywords=('qkv', 'proj', 'ffn.layers.0.0', 'ffn.layers.1'),
    #         r=lora_r,
    #         lora_alpha=lora_alpha,
    #         lora_dropout=lora_dropout
    #     )
    
    # for name, param in SAM.backbone.named_parameters():
    #     if 'lora_' in name:
    #         param.requires_grad = True
    #     else:
    #         param.requires_grad = False
    return SAM

def count_trainable_params(model):
    train_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    sum_parameters = sum(p.numel() for p in model.parameters())
    print(f'train_parameters: {train_parameters}, sum_parameters: {sum_parameters}') 

if __name__ == '__main__':

    from sam import ImageEncoderViT, sam_model_registry
    sam_checkpoint = "/home/swjtu/workspace_01/02-EX_SAM/checkpoints_sam/sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    
    # sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    # sam.to(device='cuda')

    # img_encoder = sam.image_encoder

    ################
    state_dict = torch.load(sam_checkpoint, map_location='cuda')
    filtered_image_dict = {
        k.replace('image_encoder.', '', 1): v
        for k, v in state_dict.items()
        if k.startswith('image_encoder.')
    }
    img_encoder = ImageEncoderViT(use_rel_pos=True, window_size=14, global_attn_indexes=[2, 5, 8, 11])
    img_encoder.load_state_dict(filtered_image_dict, strict=True)
    input = torch.rand((1, 3, 1024, 1024)).to('cuda')
    img_encoder.to('cuda')
    ouputs = img_encoder(input)
    for output in ouputs:
        print(output.shape)