python train.py \
    --flag 'mae_multiscale_external_fpncrossattn1d_interpolate_normpred_poollast'\
    --phase 'train'\
    --output_dir './output/logs'\
    --model_save_path './saved_models/mae_multiscale_external_fpncrossattn1d_interpolate_normpred_poollast'\
    --train_data_path '../data/celeba_train_data.txt'\
    --val_data_path '../data/celeba_val_data.txt'\
    --train_mask_path '../data/train_masks'\
    --val_mask_path '../data/val_masks_7_8'\
    --device_ids '2'\
    --batchsize 64\
    --epochs 30\
    --lr 5e-5\
    --valid_weight 1\
    --hole_weight 6\
    --use_external_mask\
    --depth 6\
    --embed_dim 768\
    --decoder_depth 12\
    --decoder_arch 'multiscale_transformer'\
    --decoder_embed_dim 768\
    --patch_size 14\
    --q_downsample_layers '[4, 8]'\
    --kv_downsample_layers '[]'\
    --pool_type 'interpolate'\
    --use_norm_pred\
    --use_fpn_loss\
    --fpn_type 'crossattn_1d'\



#     --checkpoint './saved_models/mae_multiscale_external_fpn1d_interpolate_normpred/pytorch_model.bin'\
    