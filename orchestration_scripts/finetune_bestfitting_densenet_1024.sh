#!/bin/bash

fold_i=$1

if [ "$#" -eq 1 ]; then
  gpu_id=$1
else
  gpu_id=$2
fi

gpu_id=$(($gpu_id%2))

#gpu_id=0

echo "Launching finetuning for fold ${fold_i} on GPU ${gpu_id}"

init_training_file="output/models/densenet121_1024_all_data__obvious_neg__gradaccum_200__start_lr_4e5/fold${fold_i}/9.pth"
# check, cause I had to resume after the first training
if [ ! -f "$init_training_file" ]; then
    python -m src.train.train_bestfitting --fold $fold_i --gpu-id $gpu_id --img_size 1024 --batch_size 8 --workers 22 --gradient-accumulation-steps 200 --scheduler Adam10 --epochs 10 --scheduler-lr-multiplier 4 --out_dir densenet121_1024_all_data__obvious_neg__gradaccum_200__start_lr_4e5 --eval-at-start --resume "output/models/densenet121_1024_all_data__obvious_neg__gradaccum_200__start_lr_4e5/fold${fold_i}/final.pth" > "bestfitting_tuning_1024_fast_densenet_fold${fold_i}.log"
fi
#python -m orchestration_scripts.active_waiting_for_empty_gpu --gpu-i $gpu_id

python -m src.train.train_bestfitting --fold $fold_i --gpu-id $gpu_id --img_size 1024 --batch_size 8 --workers 22 --gradient-accumulation-steps 20 --scheduler Adam10 --epochs 10 --scheduler-lr-multiplier 0.3 --out_dir densenet121_1024_all_data__obvious_neg__gradaccum_20__start_lr_3e6  --load-state-dict-path "output/models/densenet121_1024_all_data__obvious_neg__gradaccum_200__start_lr_4e5/fold${fold_i}/final.pth" --eval-at-start > "bestfitting_finetuning_1024_densenet_fold${fold_i}_final.log"
#######
#python -m src.train.train_bestfitting --fold 4 --gpu-id 0 --img_size 1024 --batch_size 8 --workers 22 --gradient-accumulation-steps 400 --scheduler Adam10 --epochs 10 --scheduler-lr-multiplier 0.3 --out_dir densenet121_1024_all_data__obvious_neg__gradaccum_20__start_lr_3e6  --load-state-dict-path "output/models/densenet121_1024_all_data__obvious_neg__gradaccum_200__start_lr_4e5/fold4/final.pth" --eval-at-start > "bestfitting_finetuning_1024_densenet_fold4_final.log"
#python -m src.train.train_bestfitting --fold 1 --gpu-id 1 --img_size 1024 --batch_size 8 --workers 22 --gradient-accumulation-steps 400 --scheduler Adam10 --epochs 10 --scheduler-lr-multiplier 0.05 --out_dir densenet121_1024_all_data__obvious_neg__gradaccum_20__start_lr_3e6  --load-state-dict-path "output/models/densenet121_1024_all_data__obvious_neg__gradaccum_200__start_lr_4e5/fold1/010.pth" --eval-at-start > "bestfitting_finetuning_1024_densenet_fold1_final.log"