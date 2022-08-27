import numpy as np
from utils import generate_stroke_mask
import os
from PIL import Image
import matplotlib.pyplot as plt
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=10)
    parser.add_argument('--save_dir', type=str, default='./masks')
    parser.add_argument('--mask_ratio_low', type=float, default=0.4)
    parser.add_argument('--mask_ratio_high', type=float, default=0.7)
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    cnt = 0
    while cnt < args.N:
        mask = generate_stroke_mask([224, 224], max_parts=10)[:, :, 0]
        mask = (mask > 0).astype(np.uint8) * 255
        ratio = np.sum(mask/255)/(224*224)

        if ratio >= args.mask_ratio_high or ratio <= args.mask_ratio_low:
            continue

        mask_img = Image.fromarray(np.array(mask).astype(np.uint8))
        mask_img.save(os.path.join(args.save_dir, 'mask_{}.jpg'.format(cnt)))
        cnt += 1