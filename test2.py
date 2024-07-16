import torch
import torch.nn as nn
import timm
from functools import partial
from models import ModelWithIntermediateLayers, ModelWithIntermediateLayersMD

def load_model(name, device):
    autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=torch.float)
    if name == 'dinov2':
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14', skip_validation=True).to(device)
        n_last_blocks = 1
        return ModelWithIntermediateLayers(model, n_last_blocks, autocast_ctx).to(device)
    if name == 'dinov2_reg':
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg', skip_validation=True).to(device)
        n_last_blocks = 1
        return ModelWithIntermediateLayers(model, n_last_blocks, autocast_ctx).to(device)
    if name == 'megadescriptor':
        model = timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384", pretrained=True)
        autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=torch.float)
        return ModelWithIntermediateLayersMD(model, autocast_ctx).to(device)
    else:
        raise ValueError(f"Unsupported feature extractor: {name}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
feature_model = load_model('dinov2_reg', device)

# Dummy input image
images = torch.randn(1, 3, 224, 224).to(device)  # Batch size 1, 3 color channels, 384x384 image

# Get features
((patch_tokens, class_token),) = feature_model(images)

print("Patch Tokens Shape:", patch_tokens.shape)
print("Class Token Shape:", class_token.shape)
# print("Pooled patches:", torch.mean(patch_tokens.view(patch_tokens.shape[0], -1, patch_tokens.shape[-1]), dim=1, keepdim=True))
