import numpy
import torch
import torch.nn as nn
from timm.models.vision_transformer import DropPath, Mlp
from torch.nn.init import trunc_normal_
import torch.nn.functional as F


def attention_pool(x, hw_shape, pool=None, norm=None, downsample_ratio=1):
    if downsample_ratio == 1:
        return x, hw_shape

    B, N, C = x.shape
    H, W = hw_shape
    x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

    if pool is None:
        x = F.interpolate(x, scale_factor=1 / downsample_ratio, mode='bicubic')
    else:
        x = pool(x)
    x = x.permute(0, 2, 3, 1).view(B, (H // downsample_ratio) * (W // downsample_ratio), x.shape[1])

    hw_shape_new = (H // downsample_ratio, W // downsample_ratio)

    if norm is not None:
        x = norm(x)

    return x, hw_shape_new


class MultiScaleAttention(nn.Module):
    def __init__(self, dim, dim_out, num_heads=12):
        super().__init__()
        self.num_heads = num_heads
        self.dim_out = dim_out
        head_dim = dim_out // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim_out * 3)
        self.proj = nn.Linear(dim_out, dim_out)

    def forward(self, x, hw_shape, pool_pos='last', pool=None, pool_norm=None, downsample_ratio=1):
        B, N, C = x.shape
        hw_shape_new = hw_shape

        qkv = self.qkv(x).reshape(B, N, 3, -1).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if pool_pos == 'first':
            q, hw_shape_new = attention_pool(q, hw_shape, pool, norm=pool_norm,
                                             downsample_ratio=downsample_ratio)
            # k, hw_shape_new = attention_pool(k, hw_shape, pool, norm=pool_norm,
            #                                  downsample_ratio=downsample_ratio)
            # v, hw_shape_new = attention_pool(v, hw_shape, pool, norm=pool_norm,
            #                                  downsample_ratio=downsample_ratio)
        q = q.reshape(B, hw_shape_new[0] * hw_shape_new[1], self.num_heads, -1).permute(0, 2, 1, 3)
        k = k.reshape(B, hw_shape[0] * hw_shape[1], self.num_heads, -1).permute(0, 2, 1, 3)
        v = v.reshape(B, hw_shape[0] * hw_shape[1], self.num_heads, -1).permute(0, 2, 1, 3)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        x = attn @ v

        x = x + q
        x = x.transpose(1, 2).reshape(B, -1, self.dim_out)
        x = self.proj(x)

        return x, hw_shape_new


class MultiScaleBlock(nn.Module):
    def __init__(self, dim, dim_out, num_heads, mlp_ratio=4.0, drop_path=0.0, downsample_ratio=1,
                 pool_type='interpolate', pool_pos='last'):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.downsample_ratio = downsample_ratio
        self.pool_pos = pool_pos
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)

        self.attn = MultiScaleAttention(dim, dim_out, num_heads)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = nn.LayerNorm(dim_out, eps=1e-6)
        mlp_hidden_dim = int(dim_out * mlp_ratio)
        self.mlp = Mlp(in_features=dim_out, hidden_features=mlp_hidden_dim, out_features=dim_out)

        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

        self.pool = None
        self.pool_norm = None
        if pool_type == 'conv':
            self.pool = nn.Conv2d(dim_out, dim_out, (3, 3), stride=(2, 2), padding=1, bias=False)
            self.pool_norm = nn.LayerNorm(dim_out, eps=1e-6)
        elif pool_type == 'avg':
            self.pool = nn.AvgPool2d((3, 3), stride=(2, 2), padding=1)
        elif pool_type == 'max':
            self.pool = nn.MaxPool2d((3, 3), stride=(2, 2), padding=1)

    def forward(self, x, hw_shape):
        x_norm = self.norm1(x)
        x_block, hw_shape_new = self.attn(x_norm, hw_shape, self.pool_pos, self.pool, self.pool_norm,
                                          self.downsample_ratio)
        if self.dim != self.dim_out:
            x = self.proj(x_norm)
        if self.pool_pos == 'first':
            x, _ = attention_pool(x, hw_shape, downsample_ratio=self.downsample_ratio)
        x = x + self.drop_path(x_block)

        if self.pool_pos == 'mid':
            x, hw_shape_new = attention_pool(x, hw_shape, self.pool, norm=self.pool_norm,
                                             downsample_ratio=self.downsample_ratio)

        x_norm = self.norm2(x)
        x_mlp = self.mlp(x_norm)
        x = x + self.drop_path(x_mlp)

        if self.pool_pos == 'last':
            x, hw_shape_new = attention_pool(x, hw_shape, self.pool, norm=self.pool_norm,
                                             downsample_ratio=self.downsample_ratio)

        return x, hw_shape_new


if __name__ == '__main__':
    model = MultiScaleBlock(768, 768*2, 12, pool_type='avg', downsample_ratio=2, pool_pos='last')
    x = torch.rand((2, 196, 768))
    output, new_shape = model(x, (14, 14))
    print(output.size())
    print(new_shape)
