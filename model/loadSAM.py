from mmpretrain import get_model

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

if __name__ == '__main__':
    import torch
    input = torch.rand((1, 3, 448, 448), device = 'cuda')
    model = load_SAM_model(modeName = 'large', device = 'cuda')
    (output,) = model(input) #!!!
    print(f'output shape: {output.shape}')
    print(model)