python train.py \
    --flag 'mae_pconv'\
    --phase 'train'\
    --output_dir './output/logs'\
    --model_save_path './saved_models/mae_pconv'\
    --train_data_path '../data/celeba_train_data.txt'\
    --val_data_path '../data/celeba_val_data.txt'\
    --train_mask_path '../data/train_masks'\
    --val_mask_path '../data/val_masks'\
    --device_ids '2'\
    --batchsize 64\
    --epochs 30\
    --lr 5e-5\
    --valid_weight 1\
    --hole_weight 6\
    --use_pconv\
    