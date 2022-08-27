from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
# from timm.models.vision_transformer import PatchEmbed, Block, to_2tuple
from models.vision_transformer import Block
from models.pos_embed import get_2d_sincos_pos_embed
from models.model_modules import PatialPatchEmbed


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, config):
        super().__init__()
        img_size = config['img_size']
        patch_size = config['patch_size']
        in_chans = config['in_chans']
        embed_dim = config['embed_dim']
        depth = config['depth']
        num_heads = config['num_heads']
        decoder_embed_dim = config['decoder_embed_dim']
        decoder_depth = config['decoder_depth']
        decoder_num_heads = config['decoder_num_heads']
        mlp_ratio = config['mlp_ratio']
        norm_pix_loss = config['norm_pix_loss']
        self.use_pconv = config['use_pconv']
        self.use_random_mask = config['use_random_mask']

        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        #         self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.patch_embed = PatialPatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding
        self.mask_enc_token = nn.Parameter(torch.zeros(1, embed_dim))

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        #         self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)  # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    int(self.patch_embed.num_patches ** .5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        attn_mask = torch.ones(x_masked.size()[:2], device=x.device)

        return x_masked, mask, ids_restore, attn_mask

    def masking(self, x, mask):
        """
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        # mask: 0 is keep, 1 is remove
        patch = self.patch_embed.patch_size[0]
        kernels = torch.ones((1, 1, patch, patch), device=x.device)
        stride = (patch, patch)
        mask = F.conv2d(mask, weight=kernels, stride=stride, bias=torch.zeros(1, device=x.device))
        mask = mask.flatten(2).squeeze(1)

        mask[mask < patch ** 2] = 0
        mask[mask >= patch ** 2] = 1

        #         mask[mask >= 0] = 1

        b = mask.size()[0]
        max_len = 0
        for i in range(b):
            index = torch.where(mask[i] == 0)[0]
            max_len = max(max_len, len(index))

        restore_index_list = []
        x_masked_list = []
        attn_mask_list = []
        for i in range(b):
            mask_index = torch.where(mask[i] == 1)[0]
            keep_index = torch.where(mask[i] == 0)[0]
            # print(x[i].shape, keep_index.shape)
            x_masked = torch.gather(x[i], dim=0, index=keep_index.unsqueeze(-1).repeat(1, D))
            seq_len = x_masked.size()[0]
            padding_len = max_len - seq_len
            if padding_len > 0:
                x_masked = torch.cat([x_masked, self.mask_enc_token.repeat(padding_len, 1)])
            x_masked_list.append(x_masked)

            attn_mask = torch.zeros(max_len, device=x.device)
            attn_mask[:seq_len] = 1
            attn_mask_list.append(attn_mask)

            restore_index = torch.argsort(torch.cat([keep_index, mask_index], dim=0))
            restore_index_list.append(restore_index)

        ids_restore = torch.stack(restore_index_list, dim=0)
        x_masked = torch.stack(x_masked_list, dim=0)
        attn_mask = torch.stack(attn_mask_list, dim=0)

        return x_masked, mask, ids_restore, attn_mask

    def forward_encoder(self, x, mask_ratio, mask=None):
        # embed patches
        if self.use_pconv:
            x = self.patch_embed(x, 1-mask)
        else:
            x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        if self.use_random_mask and self.training:
            x, mask, ids_restore, attn_mask = self.random_masking(x, mask_ratio)
        else:
            x, mask, ids_restore, attn_mask = self.masking(x, mask)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        cls_mask = torch.ones((x.shape[0], 1), device=x.device)
        x = torch.cat((cls_tokens, x), dim=1)
        attn_mask = torch.cat((cls_mask, attn_mask), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = self.norm(x)

        return x, mask, ids_restore, attn_mask

    def forward_decoder(self, x, ids_restore, attn_mask):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        B, L, D = x.shape
        x_list = []
        for i in range(B):
            index = torch.where(attn_mask[0] == 1)[0]
            x_masked = torch.gather(x[i], dim=0, index=index.unsqueeze(-1).repeat(1, D))
            #             mask_token = self.mask_token.squeeze(0)
            mask_tokens = self.mask_token.repeat(ids_restore.shape[1] + 1 - x_masked.shape[0], 1)
            x_ = torch.cat([x_masked[1:, :], mask_tokens], dim=0)  # no cls token
            x_list.append(x_)
        x_ = torch.stack(x_list, dim=0)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward(self, imgs, mask_ratio=0.75, mask=None):
        latent, mask, ids_restore, attn_mask = self.forward_encoder(imgs, mask_ratio, mask)
        pred = self.forward_decoder(latent, ids_restore, attn_mask)  # [N, L, p*p*3]

        image_pred = self.unpatchify(pred)
        mask = mask.unsqueeze(-1).repeat(1, 1, self.patch_embed.patch_size[0] ** 2 * 3)  # (N, H*W, p*p*3)
        mask = self.unpatchify(mask)  # 1 is removing, 0 is keeping
        return image_pred, mask


def mae_vit_base(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model