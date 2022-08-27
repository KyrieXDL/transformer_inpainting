python train.py \
    --flag 'mae_adv'\
    --output_dir './output/logs'\
    --model_save_path './saved_models/mae_adv'\
    --train_data_path '../data/celeba_train_data.txt'\
    --val_data_path '../data/celeba_val_data.txt'\
    --device_ids '1'\
    --batchsize 128\
    --epochs 30\
    --lr 5e-5\
    --valid_weight 1\
    --hole_weight 6\
    --use_adv\
    