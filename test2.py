import torch
import torch.nn as nn
import timm
from functools import partial

class ModelWithIntermediateLayersMegaDescriptor(nn.Module):
    def __init__(self, feature_model, autocast_ctx):
        super().__init__()
        self.feature_model = feature_model
        self.feature_model.eval()  # Set the model to evaluation mode
        self.autocast_ctx = autocast_ctx

    def forward(self, images):
        with torch.inference_mode():  # Disable gradient computation
            with self.autocast_ctx():  # Use mixed precision if applicable
                patch_tokens = self.feature_model.forward_features(images)
                patch_tokens = patch_tokens.view(patch_tokens.shape[0], -1, patch_tokens.shape[-1])
                class_token = self.feature_model(images) 
        return ((patch_tokens, class_token),)

def load_model(name, device):
    if name == 'dinov2':
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14', skip_validation=True).to(device)
        n_last_blocks = 1
        autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=torch.float)
        return ModelWithIntermediateLayers(model, n_last_blocks, autocast_ctx).to(device)
    elif name == 'megadescriptor':
        model = timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384", pretrained=True).to(device)
        autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=torch.float)
        return ModelWithIntermediateLayersMegaDescriptor(model, autocast_ctx).to(device)
    else:
        raise ValueError(f"Unsupported feature extractor: {name}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
feature_model = load_model('megadescriptor', device)

# Dummy input image
images = torch.randn(1, 3, 384, 384).to(device)  # Batch size 1, 3 color channels, 384x384 image

# Get features
((patch_tokens, class_token),) = feature_model(images)

print("Patch Tokens Shape:", patch_tokens.shape)
print("Class Token Shape:", class_token)
print("Pooled patches:", torch.mean(patch_tokens.view(patch_tokens.shape[0], -1, patch_tokens.shape[-1]), dim=1, keepdim=True))
