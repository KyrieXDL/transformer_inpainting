python train.py \
    --flag 'mae'\
    --phase 'val'\
    --output_dir './output/logs'\
    --model_save_path './saved_models/mae'\
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
    --use_random_mask\
    --depth 6\
    --embed_dim 768\
    --decoder_depth 12\
    --decoder_arch 'transformer'\
    --decoder_embed_dim 512\
    --decoder_num_heads 16\
    --checkpoint './saved_models/mae/pytorch_model.bin'\
    