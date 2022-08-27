import torch
from torch import nn
from torch.nn import functional as F


class FPN1D(nn.Module):
    def __init__(self, in_channels, out_channel, patch_size=16, use_norm_pred=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channel = out_channel
        self.patch_size = patch_size
        self.use_norm_pred = use_norm_pred

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.fpn_norms = nn.ModuleList()
        self.fpn_preds = nn.ModuleList()
        for i in range(len(in_channels)):
            l_conv = nn.Conv1d(in_channels[i], out_channel, 1)
            fpn_conv = nn.Conv1d(out_channel, out_channel, 1, groups=out_channel)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

            if self.use_norm_pred:
                fpn_norm = nn.LayerNorm(out_channel, eps=1e-6)
                fpn_pred = nn.Linear(out_channel, patch_size ** 2 * 3, bias=True)
                self.fpn_norms.append(fpn_norm)
                self.fpn_preds.append(fpn_pred)

    def forward(self, inputs, shapes):
        # build laterals, fpn_masks will remain the same with 1x1 convs
        laterals = []
        batchsize = inputs[0].shape[0]
        for i in range(len(self.lateral_convs)):
            x = self.lateral_convs[i](inputs[i].permute(0, 2, 1))
            laterals.append(x)

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            x = laterals[i].view(batchsize, self.out_channel, shapes[i][0], shapes[i][1])
            x = F.interpolate(x, scale_factor=shapes[i - 1][0] / shapes[i][0], mode='bicubic')
            x = x.view(batchsize, self.out_channel, shapes[i - 1][0] * shapes[i - 1][1])
            laterals[i - 1] += x

        # fpn conv / norm -> outputs
        fpn_feats = tuple()
        for i in range(used_backbone_levels):
            x = self.fpn_convs[i](laterals[i]).permute(0, 2, 1)
            if self.use_norm_pred:
                x = self.fpn_norms[i](x)
                x = self.fpn_preds[i](x)

            x = x.reshape(x.shape[0], shapes[i][0], shapes[i][1], self.patch_size, self.patch_size, 3)
            x = torch.einsum('nhwpqc->nchpwq', x)
            x = x.reshape(x.shape[0], 3, shapes[i][0] * self.patch_size, shapes[i][1] * self.patch_size)
            fpn_feats += (x,)

        return fpn_feats


if __name__ == '__main__':
    shapes = [(14, 14), (7, 7), (4, 4)]
    feats = [torch.rand((2, 196, 768)), torch.rand((2, 49, 768)), torch.rand((2, 16, 768))]
    model = FPN1D([768, 768, 768], 768)
    model(feats, shapes)