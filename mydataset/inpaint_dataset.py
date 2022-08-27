import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, ToTensor
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import PIL
import numpy as np
from utils import generate_stroke_mask
import os
import math


class InpaintDataset(Dataset):
    def __init__(self, data_path, mask_dir, use_external_mask=False):
        super(InpaintDataset, self).__init__()
        self.use_external_mask = use_external_mask

        with open(data_path, 'r') as fr:
            lines = fr.readlines()
        self.data = lines
        self.transform = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        mask_files = os.listdir(mask_dir)
        mask_files = [os.path.join(mask_dir, f) for f in mask_files]
        ratio = math.ceil(len(self.data) / len(mask_files))
        mask_files = (mask_files * ratio)[:len(self.data)]
        self.mask_data = mask_files
        # self.transform = self.build_transform()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path = self.data[index]
        image = Image.open(image_path.strip())
        image = image.resize((224, 224), Image.BICUBIC)
        x = self.transform(image)

        # 生成不规则的mask
        if self.use_external_mask:
            mask = Image.open(self.mask_data[index]).convert('L')
            mask = np.array(mask) / 255
            mask = torch.tensor(mask, dtype=torch.float)[None, :, :]
        else:
            mask = generate_stroke_mask([224, 224], max_parts=10)[:, :, 0]
            mask = (mask > 0).astype(np.uint8)
            mask = torch.tensor(mask, dtype=torch.float)[None, :, :]

        return x, mask

    def build_transform(self, is_train, args):
        mean = IMAGENET_DEFAULT_MEAN
        std = IMAGENET_DEFAULT_STD
        # train transform
        if is_train:
            # this should always dispatch to transforms_imagenet_train
            transform = create_transform(
                input_size=args.input_size,
                is_training=True,
                color_jitter=args.color_jitter,
                auto_augment=args.aa,
                interpolation='bicubic',
                re_prob=args.reprob,
                re_mode=args.remode,
                re_count=args.recount,
                mean=mean,
                std=std,
            )
            return transform

        # eval transform
        t = []
        if args.input_size <= 224:
            crop_pct = 224 / 256
        else:
            crop_pct = 1.0
        size = int(args.input_size / crop_pct)
        t.append(
            transforms.Resize(size, interpolation=Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(mean, std))
        return transforms.Compose(t)


if __name__ == '__main__':
    data_path = '../data/train_data.txt'
    dataset = InpaintDataset(data_path)
    x = dataset[0]
    print(x.shape)