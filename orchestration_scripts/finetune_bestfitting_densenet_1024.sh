#!/bin/bash

fold_i=$1

if [ "$#" -eq 1 ]; then
  gpu_id=$1
else
  gpu_id=$2
fi

gpu_id=$(($gpu_id%2))

#gpu_id=0

echo $fold_i
echo $gpu_id

python -m src.train.train_bestfitting --fold $fold_i --gpu-id $gpu_id --img_size 1024 --batch_size 8 --workers 14 --scheduler Adam20 --scheduler-lr-multiplier 4 --out_dir densenet121_1024_subset_all_data_obvious_neg_start_lr_4e5  --target-class-count-for-balancing 7000 > "bestfitting_tuning_1024_fast_densenet_fold${fold_i}.log"

python -m orchestration_scripts.active_waiting_for_empty_gpu --gpu-i $gpu_id

python -m src.train.train_bestfitting --fold $fold_i --gpu-id $gpu_id --img_size 1024 --batch_size 8 --workers 14 --gradient-accumulation-steps 50 --scheduler Adam20 --scheduler-lr-multiplier 0.5 --out_dir densenet121_1024_subset_all_data_obvious_neg_grad_accum_start_lr_5e6 --load-state-dict-path "../output/models/densenet121_1024_subset_all_data_obvious_neg_start_lr_4e5/fold${fold_i}/final.pth"  --target-class-count-for-balancing 1500 > "bestfitting_finetuning_1024_densenet_fold${fold_i}_final.log"
