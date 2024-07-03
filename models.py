import torch
import timm
from torch import nn

class ModelWithIntermediateLayers(nn.Module):
    def __init__(self, feature_model, n_last_blocks, autocast_ctx):
        super().__init__()
        self.feature_model = feature_model
        self.feature_model.eval()
        self.n_last_blocks = n_last_blocks
        self.autocast_ctx = autocast_ctx

    def forward(self, images):
        with torch.inference_mode():
            with self.autocast_ctx():
                features = self.feature_model.get_intermediate_layers(
                    images, self.n_last_blocks, return_class_token=True
                )
        return features

def create_linear_input(x_tokens_list, use_avgpool, use_class):
    intermediate_output = x_tokens_list
    _, class_token = intermediate_output
    class_output = torch.cat([class_token], dim=-1)
    patch_output = torch.mean((intermediate_output[0]).float(), dim=1)

    if use_avgpool and use_class:
      output = torch.cat((class_output, patch_output), dim=-1,)
      output = output.reshape(output.shape[0], -1)
      return output.float()

    if use_avgpool and not use_class:
      return patch_output.float()

    if not use_avgpool and use_class:
      return class_output.float()
    return None