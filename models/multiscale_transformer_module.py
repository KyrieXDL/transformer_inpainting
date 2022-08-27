import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
from models.multiscale_attention import MultiScaleBlock
from models.fpn2d import FPN2D
from models.fpn1d import FPN1D


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
                 q_downsample_layers=[], kv_downsample_layers=[]):
        super().__init__()
        dpr = [dropout] * depth

        pool_q, pool_kv = [(1, 1)] * depth, [(1, 1)] * depth
        stride_q, stride_kv = [(1, 1)] * depth, [(1, 1)] * depth

        self.fpn_layers = []
        for i in q_downsample_layers:
            pool_q[i] = (3, 3)
            stride_q[i] = (2, 2)
            self.fpn_layers.append(i-1)
        self.fpn_layers.append(depth-1)

        for i in kv_downsample_layers:
            pool_kv[i] = (3, 3)
            stride_kv[i] = (2, 2)

        self.blocks = nn.ModuleList()
        for i in range(depth):
            attention_block = MultiScaleBlock(
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                mlp_ratio=4.0,
                drop_path=dpr[i],
                kernel_q=pool_q[i] if len(pool_q) > i else [],
                kernel_kv=pool_kv[i] if len(pool_kv) > i else [],
                stride_q=stride_q[i] if len(stride_q) > i else [],
                stride_kv=stride_kv[i] if len(stride_kv) > i else [],
            )

            self.blocks.append(attention_block)
            embed_dim = dim_out

        if fpn_type == '2d':
            self.fpn = FPN2D([dim_out]*len(self.fpn_layers), dim_out)
        else:
            self.fpn = FPN1D([dim_out]*len(self.fpn_layers), dim_out)
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
            if i in self.fpn_layers:
                fpn_feats.append(x)
                fpn_shapes.append(thw)

        fpn_outputs = self.fpn(fpn_feats, fpn_shapes)

        return fpn_outputs