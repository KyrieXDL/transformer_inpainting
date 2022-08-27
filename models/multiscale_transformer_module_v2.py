import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
from models.multiscale_attention_v2 import MultiScaleBlock
from models.fpn2d import FPN2D
from models.fpn1d import FPN1D
from models.fpn_crossattn_1d import FPNCrossAttn1D


def round_width(width, multiplier, min_width=1, divisor=1, verbose=False):
    if not multiplier:
        return width
    width *= multiplier
    min_width = min_width or divisor

    width_out = max(min_width, int(width + divisor / 2) // divisor * divisor)
    if width_out < 0.9 * width:
        width_out += divisor
    return int(width_out)


class MultiscaleTransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, dim_out=768, depth=12, num_heads=12, dropout=0.2, fpn_type='2d',
                 patch_size=16, downsample_layers=[], use_pyramid=False, pool_type='interpolate',
                 use_norm_pred=False, pool_pos='last'):
        super().__init__()
        dpr = [dropout] * depth
        downsample_ratio_list = [1] * depth

        self.fpn_layers = []
        for i in downsample_layers:
            downsample_ratio_list[i] = 2
            self.fpn_layers.append(i-1)
        self.fpn_layers.append(depth-1)

        self.blocks = nn.ModuleList()
        fpn_embed_dim = []
        for i in range(depth):
            if use_pyramid:
                dim_out = dim_out * downsample_ratio_list[i]
            attention_block = MultiScaleBlock(
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=dim_out // 64,
                mlp_ratio=4.0,
                drop_path=dpr[i],
                downsample_ratio=downsample_ratio_list[i],
                pool_type=pool_type,
                pool_pos=pool_pos
            )

            self.blocks.append(attention_block)
            embed_dim = dim_out

            if i in self.fpn_layers:
                fpn_embed_dim.append(embed_dim)

        if fpn_type == '2d':
            self.fpn = FPN2D(fpn_embed_dim, patch_size ** 2 * 3, patch_size=patch_size, use_norm_pred=use_norm_pred)
        elif fpn_type == '1d':
            self.fpn = FPN1D(fpn_embed_dim, patch_size ** 2 * 3, patch_size=patch_size, use_norm_pred=use_norm_pred)
        elif fpn_type == 'crossattn_1d':
            self.fpn = FPNCrossAttn1D(fpn_embed_dim, patch_size ** 2 * 3, patch_size=patch_size, use_norm_pred=use_norm_pred)
        else:
            raise ValueError
            
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        thw = [int(x.shape[1] ** 0.5), int(x.shape[1] ** 0.5)]
        fpn_feats, fpn_shapes = [], []
        for i, blk in enumerate(self.blocks):
            x, thw = blk(x, thw)
#             print(x.shape)
            if i in self.fpn_layers:
                fpn_feats.append(x)
                fpn_shapes.append(thw)

        fpn_outputs = self.fpn(fpn_feats, fpn_shapes)

        return fpn_outputs
