python train.py \
    --flag 'mae_multiscale'\
    --phase 'train'\
    --output_dir './output/logs'\
    --model_save_path './saved_models/mae_multiscale'\
    --train_data_path '../data/celeba_train_data.txt'\
    --val_data_path '../data/celeba_val_data.txt'\
    --train_mask_path '../data/train_masks'\
    --val_mask_path '../data/val_masks'\
    --device_ids '1'\
    --batchsize 64\
    --epochs 30\
    --lr 5e-5\
    --valid_weight 1\
    --hole_weight 6\
    --use_random_mask\
    --depth 6\
    --decoder_depth 12\
    --decoder_arch 'multiscale_transformer'\
    