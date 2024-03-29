python train.py \
    --flag 'mae_multiscale_external_fpn1d'\
    --phase 'train'\
    --output_dir './output/logs'\
    --model_save_path './saved_models/mae_multiscale_external_fpn1d'\
    --train_data_path '../data/celeba_train_data.txt'\
    --val_data_path '../data/celeba_val_data.txt'\
    --train_mask_path '../data/train_masks'\
    --val_mask_path '../data/val_masks'\
    --device_ids '6'\
    --batchsize 64\
    --epochs 30\
    --lr 5e-5\
    --valid_weight 1\
    --hole_weight 6\
    --use_external_mask\
    --depth 6\
    --decoder_depth 12\
    --decoder_arch 'multiscale_transformer'\
    --use_fpn_loss\
    --fpn_type '1d'\
    --decoder_embed_dim 768\
    --q_downsample_layers '[4, 8]'\
    --kv_downsample_layers '[]'\
    