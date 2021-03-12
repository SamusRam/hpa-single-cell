#!/bin/bash
nohup bash orchestration_scripts/data_prep.sh &

nohup bash orchestration_scripts/train_init.sh &

nohup bash orchestration_scripts/bbox_generation.sh &

bash orchestration_scripts/graph_denoising_init.sh
------------------
bash train_accum_grad_lr_reduce_on_pl.sh 0 3e-4 0
#bash train_accum_grad_lr_reduce_on_pl.sh 0 1e-2 1 BAN
#nohup python -m src.train.train_bestfitting --lr-reduce-patience 4 --init-lr 3e-4 --fold 0 --gpu-id 1  --img_size 512 --batch_size 32 --gradient-accumulation-steps 3 --out_dir densenet121_1024_subset_all_data_obvious_neg_grad_accum_start_lr_3e4_noclipping > no_clipping.log & # --load-state-dict-path "../output/models/densenet121_512_subset_all_data_obvious_neg_grad_accum_start_lr_${2}/fold${1}/final.pth" > "1024_densenet_fold${1}_lr_${2}.log"


#nohup python -m src.train.train_bestfitting --lr-reduce-patience 4 --init-lr 3e-4 --fold 0 --img_size 512 --batch_size 32 --gradient-accumulation-steps 3 --target-class-count-for-balancing 3000 --out_dir densenet121_512_subset_all_data_no_neg_grad_accum_start_lr_3e4_noclipping_lrpatience_2_3000_balancing --ignore-negs --lr-reduce-patience 1 > no_negs_no_clipping.log &


# checking public data effect
## densenet
nohup python -m src.train.train_bestfitting --gpu-id 1 --epochs 55 --scheduler Adam55 --workers 24 --fold 0 --img_size 1024 --batch_size 8 --gradient-accumulation-steps 120 --target-class-count-for-balancing 7000 --out_dir densenet121_1024_subset_all_data_grad_accum_clipping_7000_balancing_Adam55 > densenet_1024_adam55.log &
#nohup python -m src.train.train_bestfitting --load-state-dict-path ../output/models/densenet121_512_subset_all_data_grad_accum_clipping_4000_balancing_Adam45/fold0/final.pth --epochs 45 --scheduler Adam45 --workers 28 --fold 0 --img_size 512 --batch_size 32 --gradient-accumulation-steps 3 --target-class-count-for-balancing 8000 --out_dir densenet121_512_subset_all_data_grad_accum_clipping_8000_balancing_Adam45 > densenet_adam45_2.log &
#
#
### effnet
#nohup python -m src.train.train_bestfitting --epochs 45 --scheduler Adam45 --workers 14 --arch class_efficientnet_dropout --fold 0 --img_size 512 --batch_size 8 --gradient-accumulation-steps 12 --target-class-count-for-balancing 4000 --out_dir efficientnetb1_512_subset_all_data_grad_accum_clipping_00_balancing > effnetb1_adam45.log &
#nohup python -m src.train.train_bestfitting --epochs 45 --gpu-id 1 --effnet-encoder efficientnet-b2 --scheduler Adam45 --workers 14 --arch class_efficientnet_dropout --fold 0 --img_size 512 --batch_size 8 --gradient-accumulation-steps 12 --target-class-count-for-balancing 4000 --out_dir efficientnetb2_512_subset_all_data_grad_accum_clipping_00_balancing > effnetb1_adam45.log &
#
##nohup python -m src.train.train_bestfitting --arch class_efficientnet_dropout --epochs 20 --fold 0 --img_size 512 --batch_size 8  --scheduler-lr-multiplier 0.25 --out_dir efficientnetb1_512 &
#
##### orig scheduler
#python -m src.train.train_bestfitting --epochs 45 --workers 28 --init-lr 2e-3 --fold 0 --img_size 512 --batch_size 32 --gradient-accumulation-steps 3 --target-class-count-for-balancing 1500 --out_dir densenet121_512_subset_all_data_no_pubs_grad_accum_start_lr_2e3_noclipping_lrpatience_2_1500_balancing_warmup --without-public-data
