#!/bin/bash

#fold_i=$1
#
#if [ "$#" -eq 1 ]; then
#  gpu_id=$1
#else
#  gpu_id=$2
#fi
#
#gpu_id=$(($gpu_id%2))
#
##gpu_id=0
#
#echo "Launching finetuning for fold ${fold_i} on GPU ${gpu_id}"
for fold_i in {0..1}; do
#fold_i=4
  python -m src.train.train_bestfitting --fold $fold_i --gpu-id 0 --clip-and-replace-grad-explosures --clean-duplicates --clean-mitotic-samples --clean-aggresome --copy-paste-augment-mitotic-aggresome --img_size 1024 --loss FocalSymmetricLovaszHardLogLoss --batch_size 8 --workers 15 --gradient-accumulation-steps 4 --scheduler Adam10 --epochs 3 --scheduler-lr-multiplier 0.03 --out_dir densenet121_1024_mitotic_aggresome_copypaste_all_data__obvious_neg__gradaccum_4__start_lr_3e7  --load-state-dict-path "output/models/densenet121_1024_all_data__obvious_neg__gradaccum_20__start_lr_3e6/fold${fold_i}/final.pth" --eval-at-start > "bestfitting_mitotic_aggresome_finetuning_copypaste_1024_densenet_fold${fold_i}.log"
done
#######
#python -m src.train.train_bestfitting --fold 4 --gpu-id 0 --img_size 1024 --batch_size 8 --workers 22 --gradient-accumulation-steps 400 --scheduler Adam10 --epochs 10 --scheduler-lr-multiplier 0.3 --out_dir densenet121_1024_all_data__obvious_neg__gradaccum_20__start_lr_3e6  --load-state-dict-path "output/models/densenet121_1024_all_data__obvious_neg__gradaccum_200__start_lr_4e5/fold4/final.pth" --eval-at-start > "bestfitting_finetuning_1024_densenet_fold4_final.log"
#python -m src.train.train_bestfitting --fold 1 --gpu-id 1 --img_size 1024 --batch_size 8 --workers 22 --gradient-accumulation-steps 400 --scheduler Adam10 --epochs 10 --scheduler-lr-multiplier 0.05 --out_dir densenet121_1024_all_data__obvious_neg__gradaccum_20__start_lr_3e6  --load-state-dict-path "output/models/densenet121_1024_all_data__obvious_neg__gradaccum_200__start_lr_4e5/fold1/010.pth" --eval-at-start > "bestfitting_finetuning_1024_densenet_fold1_final.log"