import torch
from torch import nn
import torch.nn.functional as F
import math

from jepa_utils import Block, CrossAttention, CrossAttentionBlock, trunc_normal_

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
    
class ModelWithIntermediateLayersMD(nn.Module):
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

class AttentivePooler(nn.Module):
    """ Attentive Pooler """
    def __init__(
        self,
        num_queries=1,
        embed_dim=1536,
        num_heads=12,
        mlp_ratio=4.0,
        depth=1,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        qkv_bias=True,
        complete_block=True
    ):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.zeros(1, num_queries, embed_dim))

        self.complete_block = complete_block
        if complete_block:
            self.cross_attention_block = CrossAttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer)
        else:
            self.cross_attention_block = CrossAttention(
                dim=embed_dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias)

        self.blocks = None
        if depth > 1:
            self.blocks = nn.ModuleList([
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=False,
                    norm_layer=norm_layer)
                for i in range(depth-1)])

        self.init_std = init_std
        trunc_normal_(self.query_tokens, std=self.init_std)
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        if self.complete_block:
            rescale(self.cross_attention_block.xattn.proj.weight.data, 1)
            rescale(self.cross_attention_block.mlp.fc2.weight.data, 1)
        else:
            rescale(self.cross_attention_block.proj.weight.data, 1)
        if self.blocks is not None:
            for layer_id, layer in enumerate(self.blocks, 1):
                rescale(layer.attn.proj.weight.data, layer_id + 1)
                rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        q = self.query_tokens.repeat(len(x), 1, 1)
        q = self.cross_attention_block(q, x)
        if self.blocks is not None:
            for blk in self.blocks:
                q = blk(q)
        return q

class AttentiveClassifier(nn.Module):
    """ Attentive Classifier """
    def __init__(
        self,
        embed_dim=1536,
        num_heads=12,
        mlp_ratio=4.0,
        depth=1,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        qkv_bias=True,
        num_classes=1000,
        complete_block=True,
        use_class=True
    ):
        super().__init__()
        self.pooler = AttentivePooler(
            num_queries=1,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            depth=depth,
            norm_layer=norm_layer,
            init_std=init_std,
            qkv_bias=qkv_bias,
            complete_block=complete_block,
        )
        self.use_class = use_class
        if use_class:
            self.linear = nn.Linear(2 * embed_dim, num_classes, bias=True)
        else:
            self.linear = nn.Linear(embed_dim, num_classes, bias=True)

    def forward(self, x):
        pooled_output = self.get_pooled_output(x)
        x = self.linear(pooled_output)
        return x

    def get_pooled_output(self, x):
        (patch_tokens, class_token) = x
        pooled_output = self.pooler(patch_tokens).squeeze(1)
        if self.use_class:
            pooled_output = torch.cat([pooled_output, class_token], dim=-1)
        return pooled_output
