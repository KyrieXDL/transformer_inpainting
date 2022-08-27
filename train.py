import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW, Adam
from mydataset.inpaint_dataset import InpaintDataset
from models.inpaint_model_mae import MaskedAutoencoderViT
from models.discrime_model import Discriminator
import torch.nn.functional as F
import utils
import os
from datetime import datetime
import time
import argparse
from PIL import Image
import numpy as np
from tqdm import tqdm
from metrics import ssim, psnr, masked_mae, mae
from loss import AdversarialLoss, L2Loss


def train_epoch(model, dis_model, dataloader, optimizer, dis_optimizer, loss_criterion, adversarial_loss,
                epoch, logger, args):
    start_time = time.time()
    num_total_steps = len(dataloader)
    model.train()
    dis_model.train()
    for step, batch in enumerate(dataloader):
        image = batch[0].cuda()
        irregular_mask = batch[1].cuda()

        output = model(image, mask_ratio=0.5, mask=irregular_mask)
        image_pred, patch_mask = output[0], output[1]

        if not args.use_random_mask:
            mask = irregular_mask
        else:
            mask = patch_mask

        image_comp = image * (1 - mask) + image_pred * mask

        gen_loss, adv_loss, mae_loss, dis_loss = 0, 0, 0, 0
        # 判别器loss
        if args.use_adv:
            dis_real_feat = dis_model(image)
            dis_fake_feat = dis_model(image_comp.detach())
            dis_real_loss = adversarial_loss(dis_real_feat, True, True)
            dis_fake_loss = adversarial_loss(dis_fake_feat, False, True)
            dis_loss += (dis_real_loss + dis_fake_loss) / 2
            dis_loss.backward()
            dis_optimizer.step()
            dis_optimizer.zero_grad()

        # 生成器loss
        gen_loss += loss_criterion(image, image_pred, mask)

        # 生成器多尺度loss
        if args.use_fpn_loss:
            fpn_feats = output[3]
            fpn_loss_weight = 0.5
            for i in range(len(fpn_feats)):
                cur_mask = F.interpolate(mask, scale_factor=fpn_feats[i].shape[2] / image.shape[2], mode='nearest')
                image_gt = F.interpolate(image, scale_factor=fpn_feats[i].shape[2] / image.shape[2], mode='bilinear')
                gen_loss += loss_criterion(image_gt, fpn_feats[i], cur_mask) * fpn_loss_weight

        # 生成器valid patch loss
        if args.use_mae_loss:
            image_mae_pred = output[2]
            mae_mask = (1 - patch_mask) * irregular_mask
            hole_mae_loss = torch.mean(((image_mae_pred * mae_mask - image * mae_mask) ** 2)
                                       .sum(dim=(1, 2, 3)) / mae_mask.sum(dim=(1, 2, 3)))
            valid_mae_loss = torch.mean(((image_mae_pred * (1 - irregular_mask) - image * (1 - irregular_mask)) ** 2).
                                        sum(dim=(1, 2, 3)) / (1 - irregular_mask).sum(dim=(1, 2, 3)))
            mae_loss = hole_mae_loss * args.mae_hole_weight + valid_mae_loss * args.mae_valid_weight
            gen_loss += mae_loss

        # 生成器对抗loss
        if args.use_adv:
            gen_fake_feat = dis_model(image_comp)
            adv_loss = adversarial_loss(gen_fake_feat, True, False)
            gen_loss += adv_loss

        gen_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 100 == 0:
            time_per_step = (time.time() - start_time) / max(1, step)
            remaining_time = time_per_step * (num_total_steps - step)
            remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
            logger.info(
                'epoch: {}, step: {}, eta: {}, gen loss: {}, mae loss: {}, adv_loss: {}, dis loss: {}'.format(epoch,
                                                                                                              step,
                                                                                                              remaining_time,
                                                                                                              gen_loss,
                                                                                                              mae_loss,
                                                                                                              adv_loss,
                                                                                                              dis_loss))


def val_epoch(model, dataloader, loss_criterion, epoch, logger, args):
    model.eval()
    mae_average_meters, ssim_average_meters, psnr_average_meters = utils.AverageMeter(), utils.AverageMeter(), utils.AverageMeter()
    all_image_gt, all_image_pred, all_image_comp = [], [], []
    start_time = time.time()
    with torch.no_grad():
        for step, batch in tqdm(enumerate(dataloader)):
            image = batch[0].cuda(non_blocking=True)
            irregular_mask = batch[1].cuda(non_blocking=True)
            output = model(image, mask=irregular_mask)
            image_pred, patch_mask = output[0], output[1]

            loss = loss_criterion(image, image_pred, irregular_mask)

            image_pred = torch.einsum('nchw->nhwc', image_pred)
            image_gt = torch.einsum('nchw->nhwc', image)
            mask = torch.einsum('nchw->nhwc', irregular_mask.repeat(1, 3, 1, 1))

            image_comp = image_gt * (1 - mask) + image_pred * mask
            image_mask = image_gt * (1 - mask)
            image_comp_np = utils.postprocess(image_comp.cpu())
            image_gt_np = utils.postprocess(image_gt.cpu())
            image_pred_np = utils.postprocess(image_pred.cpu())
            image_mask_np = utils.postprocess(image_mask.cpu())

            if step == 0:
                idx = 5
                Image.fromarray(image_gt_np[idx]).save('./output_imgs/{}_{}_img_orig.jpg'.format(idx, args.flag))
                Image.fromarray(image_comp_np[idx]).save('./output_imgs/{}_{}_img_comp.jpg'.format(idx, args.flag))
                Image.fromarray(image_pred_np[idx]).save('./output_imgs/{}_{}_img_pred.jpg'.format(idx, args.flag))
                Image.fromarray(image_mask_np[idx]).save('./output_imgs/{}_{}_img_mask.jpg'.format(idx, args.flag))

                if args.use_fpn_loss:
                    fpn_feats = output[3]
                    for i in range(len(fpn_feats)):
                        cur_mask = F.interpolate(irregular_mask, scale_factor=fpn_feats[i].shape[2] / image.shape[2],
                                                 mode='nearest')
                        cur_image_gt = F.interpolate(image, scale_factor=fpn_feats[i].shape[2] / image.shape[2],
                                                     mode='bilinear')

                        cur_image_gt = torch.einsum('nchw->nhwc', cur_image_gt).detach().cpu()
                        cur_mask = torch.einsum('nchw->nhwc', cur_mask.repeat(1, 3, 1, 1)).detach().cpu()
                        cur_image_pred = torch.einsum('nchw->nhwc', fpn_feats[i]).detach().cpu()

                        cur_image_comp = cur_image_gt * (1 - cur_mask) + cur_image_pred * cur_mask
                        cur_image_mask = cur_image_gt * (1 - cur_mask)

                        cur_image_gt = utils.postprocess(cur_image_gt)
                        cur_image_comp = utils.postprocess(cur_image_comp)
                        cur_image_mask = utils.postprocess(cur_image_mask)
                        cur_image_pred = utils.postprocess(cur_image_pred)

                        Image.fromarray(cur_image_gt[idx]).save('./output_imgs/{}_{}_img_orig_{}.jpg'.format(idx, args.flag, i+1))
                        Image.fromarray(cur_image_comp[idx]).save('./output_imgs/{}_{}_img_comp_{}.jpg'.format(idx,args.flag, i+1))
                        Image.fromarray(cur_image_pred[idx]).save('./output_imgs/{}_{}_img_pred_{}.jpg'.format(idx,args.flag, i+1))
                        Image.fromarray(cur_image_mask[idx]).save('./output_imgs/{}_{}_img_mask_{}.jpg'.format(idx,args.flag, i+1))

            all_image_gt.append(image_gt_np)
            all_image_pred.append(image_pred_np)
            all_image_comp.append(image_comp_np)
            
            break
    
    all_image_gt = np.concatenate(all_image_gt, axis=0)
    all_image_pred = np.concatenate(all_image_pred, axis=0)
    all_image_comp = np.concatenate(all_image_comp, axis=0)

    val_mae = mae(all_image_gt, all_image_pred, mean=True)
    val_ssim = ssim(all_image_gt, all_image_comp, mean=True)
    val_psnr = psnr(all_image_gt, all_image_comp, mean=True)
    
    remaining_time = (time.time() - start_time)
    remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
    print('cost_time: ', remaining_time)

    logger.info('epoch: {}, loss: {}, mae: {}, ssim: {}, psnr: {}'.format(epoch, loss,
                                                                          val_mae,
                                                                          val_ssim,
                                                                          val_psnr))


def main(args):
    ### init
    utils.fix_random_seeds(args.seed)
    if not os.path.exists(args.model_save_path):
        os.makedirs(args.model_save_path)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    year, month, day = datetime.now().year, datetime.now().month, datetime.now().day
    logger = utils.create_logger(
        os.path.join(args.output_dir, 'log_{}_{}_{}_{}.txt'.format(args.flag, year, month, day)))
    logger.info(args)

    ### create dataset and dataloader
    train_dataset = InpaintDataset(args.train_data_path, args.train_mask_path, use_external_mask=args.use_external_mask)
    val_dataset = InpaintDataset(args.val_data_path, args.val_mask_path, use_external_mask=True)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, pin_memory=True,
                                  num_workers=args.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batchsize, shuffle=False, pin_memory=True,
                                num_workers=args.num_workers)

    print(len(train_dataset), len(val_dataset))

    ### create model
    config = {'img_size': 224, 'patch_size': args.patch_size, 'in_chans': 3,
              'embed_dim': args.embed_dim, 'depth': args.depth, 'num_heads': args.num_heads,
              'decoder_embed_dim': args.decoder_embed_dim, 'decoder_depth': args.decoder_depth,
              'decoder_num_heads': args.decoder_num_heads, 'mlp_ratio': 4, 'use_pconv': args.use_pconv,
              'use_random_mask': args.use_random_mask, 'decoder_arch': args.decoder_arch, 'fpn_type': args.fpn_type,
              'q_downsample_layers': args.q_downsample_layers, 'kv_downsample_layers': args.kv_downsample_layers,
              'use_pyramid': args.use_pyramid, 'pool_type': args.pool_type, 'pool_pos': args.pool_pos, 
              'use_norm_pred': args.use_norm_pred, 'use_mae_loss': args.use_mae_loss}
    model = MaskedAutoencoderViT(config)
    model = model.cuda()
    dis_model = Discriminator(in_channels=3, use_sigmoid=True, use_spectral_norm=True)
    dis_model = dis_model.cuda()

    if args.checkpoint != '':
        state_dict = torch.load(args.checkpoint, map_location='cpu')
        msg = model.load_state_dict(state_dict, strict=False)
        logger.info('load ckpt from {}'.format(args.checkpoint))
        logger.info('msg: {}'.format(msg))

    ### optimizer
    optimizer, scheduler = utils.create_optimizer(model, args.lr, no_decay_names=['bias', 'LayerNorm.weight'],
                                                  warmup_steps=args.warmup_steps,
                                                  max_steps=args.epochs * len(train_dataloader),
                                                  schedule_type=args.schedule_type)

    dis_optimizer = Adam(dis_model.parameters(), lr=5e-5, betas=(0.5, 0.999))

    loss_criterion = L2Loss(hole_weight=args.hole_weight, valid_weight=args.valid_weight)
    dis_loss_criterion = AdversarialLoss(type='nsgan').cuda()

    ### train
    if args.phase == 'train':
        for epoch in range(args.epochs):
            train_epoch(model, dis_model, train_dataloader, optimizer, dis_optimizer, loss_criterion,
                        dis_loss_criterion,
                        epoch, logger=logger, args=args)
            val_epoch(model, val_dataloader, loss_criterion, epoch, logger, args)

            optimizer_state_dict = {
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }
            torch.save(model.state_dict(), os.path.join(args.model_save_path, 'pytorch_model.bin'))
            torch.save(optimizer_state_dict, os.path.join(args.model_save_path, 'optimizer.bin'))
    elif args.phase == 'val':
        val_epoch(model, val_dataloader, loss_criterion, 0, logger, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--flag', type=str, default='base')
    parser.add_argument('--output_dir', type=str, default='./output/logs')
    parser.add_argument('--model_save_path', type=str, default='./saved_models/base')
    parser.add_argument('--train_data_path', type=str, default='./saved_models/base')
    parser.add_argument('--val_data_path', type=str, default='./saved_models/base')
    parser.add_argument('--train_mask_path', type=str, default='../data/train_masks')
    parser.add_argument('--val_mask_path', type=str, default='../data/val_masks')

    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--depth', type=int, default=12)
    parser.add_argument('--embed_dim', type=int, default=768)
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--decoder_depth', type=int, default=8)
    parser.add_argument('--decoder_embed_dim', type=int, default=512)
    parser.add_argument('--decoder_num_heads', type=int, default=8)
    parser.add_argument('--decoder_arch', type=str, default='transformer')
    parser.add_argument('--pool_type', type=str, default='interpolate')
    parser.add_argument('--pool_pos', type=str, default='last')
    parser.add_argument('--fpn_type', type=str, default='2d')
    parser.add_argument('--q_downsample_layers', type=str, default='[4, 8]')
    parser.add_argument('--kv_downsample_layers', type=str, default='[4, 8]')
    parser.add_argument('--use_pyramid', action='store_true')
    parser.add_argument('--use_norm_pred', action='store_true')
    parser.add_argument('--use_mae_loss', action='store_true')

    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batchsize', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--device_ids', default='0')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--prefetch', type=int, default=6)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--schedule_type', type=str, default='')
    parser.add_argument('--use_adv', action='store_true')
    parser.add_argument('--use_random_mask', action='store_true')
    parser.add_argument('--use_external_mask', action='store_true')
    parser.add_argument('--use_pconv', action='store_true')
    parser.add_argument('--use_fpn_loss', action='store_true')
    parser.add_argument('--hole_weight', type=float, default=1)
    parser.add_argument('--valid_weight', type=float, default=0)
    parser.add_argument('--mae_hole_weight', type=float, default=1)
    parser.add_argument('--mae_valid_weight', type=float, default=0)

    args = parser.parse_args()
    print(torch.cuda.current_device())
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_ids
    torch.cuda.set_device(int(args.device_ids))
    print(torch.cuda.current_device())

    main(args)
