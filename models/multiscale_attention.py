import numpy
import torch
import torch.nn as nn
from timm.models.vision_transformer import DropPath, Mlp
from torch.nn.init import trunc_normal_


def attention_pool(tensor, pool, hw_shape, norm=None):
    if pool is None:
        return tensor, hw_shape
    tensor_dim = tensor.ndim
    if tensor_dim == 4:
        pass
    elif tensor_dim == 3:
        tensor = tensor.unsqueeze(1)
    else:
        raise NotImplementedError(f"Unsupported input dimension {tensor.shape}")

    B, N, L, C = tensor.shape
    H, W = hw_shape
    tensor = tensor.reshape(B * N, H, W, C).permute(0, 3, 1, 2).contiguous()
    tensor = pool(tensor)

    hw_shape = [tensor.shape[2], tensor.shape[3]]
    L_pooled = tensor.shape[2] * tensor.shape[3]
    tensor = tensor.reshape(B, N, C, L_pooled).transpose(2, 3)

    if norm is not None:
        tensor = norm(tensor)

    if tensor_dim == 3:
        tensor = tensor.squeeze(1)
    return tensor, hw_shape


class MultiScaleAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        num_heads=8,
        kernel_q=(1, 1),
        kernel_kv=(1, 1),
        stride_q=(1, 1),
        stride_kv=(1, 1),
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dim_out = dim_out
        head_dim = dim_out // num_heads
        self.scale = head_dim**-0.5
        padding_q = [int(q // 2) for q in kernel_q]
        padding_kv = [int(kv // 2) for kv in kernel_kv]

        self.qkv = nn.Linear(dim, dim_out * 3)
        self.proj = nn.Linear(dim_out, dim_out)

        # Skip pooling with kernel and stride size of (1, 1, 1).
        if numpy.prod(kernel_q) == 1 and numpy.prod(stride_q) == 1:
            kernel_q = ()
        if numpy.prod(kernel_kv) == 1 and numpy.prod(stride_kv) == 1:
            kernel_kv = ()

        dim_conv = dim_out // num_heads
        self.pool_q, self.norm_q, self.pool_k, self.norm_k, self.pool_v, self.norm_v = (None,) * 6
        if len(kernel_q) > 0:
            self.pool_q = nn.Conv2d(dim_conv, dim_conv, kernel_q, stride=stride_q, padding=padding_q, groups=dim_conv, bias=False)
            self.norm_q = nn.LayerNorm(dim_conv, eps=1e-6)
        if len(kernel_kv) > 0:
            self.pool_k = nn.Conv2d(dim_conv, dim_conv, kernel_kv, stride=stride_kv, padding=padding_kv, groups=dim_conv, bias=False)
            self.norm_k = nn.LayerNorm(dim_conv, eps=1e-6)
            self.pool_v = nn.Conv2d(dim_conv, dim_conv, kernel_kv, stride=stride_kv, padding=padding_kv, groups=dim_conv, bias=False)
            self.norm_v = nn.LayerNorm(dim_conv, eps=1e-6)

    def forward(self, x, hw_shape):
        B, N, _ = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q, q_shape = attention_pool(q, self.pool_q, hw_shape, norm=self.norm_q)
        k, k_shape = attention_pool(k, self.pool_k, hw_shape, norm=self.norm_k)
        v, v_shape = attention_pool(v, self.pool_v, hw_shape, norm=self.norm_v)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        x = attn @ v

        x = x + q
        x = x.transpose(1, 2).reshape(B, -1, self.dim_out)
        x = self.proj(x)

        return x, q_shape


class MultiScaleBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        num_heads,
        mlp_ratio=4.0,
        drop_path=0.0,
        kernel_q=(1, 1),
        kernel_kv=(1, 1),
        stride_q=(1, 1),
        stride_kv=(1, 1),
    ):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)

        self.attn = MultiScaleAttention(dim, dim_out, num_heads, kernel_q, kernel_kv, stride_q, stride_kv)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = nn.LayerNorm(dim_out, eps=1e-6)
        mlp_hidden_dim = int(dim_out * mlp_ratio)
        self.mlp = Mlp(in_features=dim_out, hidden_features=mlp_hidden_dim, out_features=dim_out)

        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

        self.pool_skip = None
        if len(stride_q) > 0 and numpy.prod(stride_q) > 1:
            kernel_skip = [s + 1 if s > 1 else s for s in stride_q]
            stride_skip = stride_q
            padding_skip = [int(skip // 2) for skip in kernel_skip]
            self.pool_skip = nn.MaxPool2d(
                kernel_skip, stride_skip, padding_skip, ceil_mode=False
            )

    def forward(self, x, hw_shape):
        x_norm = self.norm1(x)
        x_block, hw_shape_new = self.attn(x_norm, hw_shape)

        if self.dim != self.dim_out:
            x = self.proj(x_norm)
        x_res, _ = attention_pool(x, self.pool_skip, hw_shape)
        x = x_res + self.drop_path(x_block)

        x_norm = self.norm2(x)
        x_mlp = self.mlp(x_norm)
        x = x + self.drop_path(x_mlp)
        return x, hw_shape_new


if __name__ == '__main__':
    model = MultiScaleBlock(768, 768, 12, stride_q=(2, 2), kernel_q=(3, 3),
                            stride_kv=(2, 2), kernel_kv=(3, 3))
    x = torch.rand((2, 196, 768))
    output, new_shape = model(x, (14, 14))
    print(output.size())
    print(new_shape)