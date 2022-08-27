import torch
import os
import torch.nn as nn
import torch.nn.functional as F


class L2Loss(nn.Module):
    def __init__(self, hole_weight=1, valid_weight=0, norm_pix_loss=False):
        super(L2Loss, self).__init__()
        self.norm_pix_loss = norm_pix_loss
        self.hole_weight = hole_weight
        self.valid_weight = valid_weight

    def __call__(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, 3, H, W]
        mask: [N, 3, H, W], 0 is keep, 1 is remove,
        """
        if self.norm_pix_loss:
            mean = pred.mean(dim=(1, 2, 3), keepdim=True)
            var = pred.var(dim=(1, 2, 3), keepdim=True)
            pred = (pred - mean) / (var + 1.e-6) ** .5

        loss = (pred - imgs) ** 2
        loss_hole = torch.mean((loss*mask).sum(dim=(1, 2, 3)) / mask.sum(dim=(1, 2, 3)))
        loss_valid = torch.mean((loss*(1-mask)).sum(dim=(1, 2, 3)) / (1-mask).sum(dim=(1, 2, 3)))
        loss = loss_hole * self.hole_weight + loss_valid * self.valid_weight
        return loss

class AdversarialLoss(nn.Module):
    r"""
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """

    def __init__(self, type='nsgan', target_real_label=1.0, target_fake_label=0.0):
        r"""
        type = nsgan | lsgan | hinge
        """
        super(AdversarialLoss, self).__init__()

        self.type = type
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if type == 'nsgan':
            self.criterion = nn.BCELoss()

        elif type == 'lsgan':
            self.criterion = nn.MSELoss()

        elif type == 'hinge':
            self.criterion = nn.ReLU()

    def __call__(self, outputs, is_real, is_disc=None):
        if self.type == 'hinge':
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()

        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
            loss = self.criterion(outputs, labels)
            return loss



class PerceptualLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(PerceptualLoss, self).__init__()
        self.criterion = torch.nn.L1Loss()
        self.weights = weights

    def __call__(self, x_vgg, y_vgg):
        # Compute features
        content_loss = 0.0
        prefix = [1, 2, 3, 4, 5]
        for i in range(5):
            content_loss += self.weights[i] * self.criterion(
                x_vgg['relu{}_1'.format(prefix[i])], y_vgg['relu{}_1'.format(prefix[i])])
        return content_loss


class StyleLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self):
        super(StyleLoss, self).__init__()
        self.criterion = torch.nn.L1Loss()

    def compute_gram(self, x):
        b, c, h, w = x.size()
        f = x.view(b, c, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * c)
        return G

    def __call__(self, x_vgg, y_vgg):
        # Compute loss
        style_loss = 0.0
        prefix = [2, 3, 4, 5]
        posfix = [2, 4, 4, 2]
        for pre, pos in list(zip(prefix, posfix)):
            style_loss += self.criterion(self.compute_gram(x_vgg['relu{}_{}'.format(pre, pos)]),
                                         self.compute_gram(y_vgg['relu{}_{}'.format(pre, pos)]))
        return style_loss