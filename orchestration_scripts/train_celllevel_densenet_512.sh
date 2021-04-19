#!/bin/bash

fold_i=$1

if [ "$#" -eq 1 ]; then
  gpu_id=$1
else
  gpu_id=$2
fi

gpu_id=$(($gpu_id%2))
# aug and grad accum after the 1st epoch
nohup python -m src.train.train_cellwise --img_size 512 --batch_size 32 --workers 20 --scheduler Adam10WarmUp --epochs 10 --out_dir "img_level_cellwise_densenet121_512_all_data__obvious_neg__adam10warmup"  --load-state-dict-path "use-img-level-densenet-ckpt" --gradient-accumulation-steps 50 --fold $fold_i --gpu-id $gpu_id --image-level-labels > img_level_cellwise_fold_0.log &#######

nohup python -m src.train.train_cellwise --img_size 512 --batch_size 8 --workers 20 --scheduler Adam10WarmUp --epochs 10 --out_dir "img_level_cellwise_effnetb7_512_all_data__obvious_neg__adam10warmup" --gradient-accumulation-steps 200 --fold 0 --image-level-labels --normalize --arch class_efficientnet_dropout --effnet-encoder efficientnet-b4 > img_level_cellwise_fold_0_effnetb4.log &
#python -m src.train.train_bestfitting --fold 1 --gpu-id 1 --img_size 1024 --batch_size 8 --workers 22 --gradient-accumulation-steps 400 --scheduler Adam10 --epochs 10 --scheduler-lr-multiplier 0.05 --out_dir densenet121_1024_all_data__obvious_neg__gradaccum_20__start_lr_3e6  --load-state-dict-path "../output/models/densenet121_1024_all_data__obvious_neg__gradaccum_200__start_lr_4e5/fold1/010.pth" --eval-at-start > "bestfitting_finetuning_1024_densenet_fold1_final.log"
#python -m src.train.train_bestfitting --fold 1 --gpu-id 1 --img_size 1024 --batch_size 8 --workers 22 --gradient-accumulation-steps 400 --scheduler Adam10 --epochs 10 --scheduler-lr-multiplier 0.05 --out_dir densenet121_1024_all_data__obvious_neg__gradaccum_20__start_lr_3e6  --load-state-dict-path "../output/models/densenet121_1024_all_data__obvious_neg__gradaccum_200__start_lr_4e5/fold1/010.pth" --eval-at-start > "bestfitting_finetuning_1024_densenet_fold1_final.log"