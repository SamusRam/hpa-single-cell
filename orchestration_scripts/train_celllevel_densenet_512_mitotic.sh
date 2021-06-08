#!/bin/bash

fold_i=$1

if [ "$#" -eq 1 ]; then
  gpu_id=$1
else
  gpu_id=$2
fi

gpu_id=$(($gpu_id%2))
# aug and grad accum after the 1st epoch


python -m src.train.train_cellwise_mitotic_bin --fold $fold_i --gpu-id $gpu_id --img_size 224 --batch_size 8 --workers 10 --gradient-accumulation-steps 1 --cell-level-labels-path output/densenet121_pred.h5 --scheduler Adam10 --epochs 10 --scheduler-lr-multiplier 100 --out_dir densenet121_mitotic_224_cellwise__start_lr_1e3  --load-state-dict-path "output/models/densenet121_512_cellwise__gradaccum_4__start_lr_4e5/fold${fold_i}/final.pth" > "cellwise_mitotic_224_densenet_fold${fold_i}.log"

python -m src.train.train_cellwise_mitotic_bin --fold $fold_i --gpu-id $gpu_id --img_size 224 --batch_size 8 --workers 10 --gradient-accumulation-steps 1 --cell-level-labels-path output/densenet121_pred.h5 --scheduler Adam10 --epochs 10 --scheduler-lr-multiplier 4 --out_dir densenet121_mitotic_224_cellwise__start_lr_4e5  --load-state-dict-path "output/models/densenet121_mitotic_224_cellwise__start_lr_1e3/fold${fold_i}/final.pth" --load-as-is > "cellwise_mitotic_224_densenet_fold${fold_i}_2nd_run.log"


