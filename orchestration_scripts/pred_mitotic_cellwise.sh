#!/bin/bash

fold_i=$1

if [ "$#" -eq 1 ]; then
  gpu_id=$1
else
  gpu_id=$2
fi

gpu_id=$(($gpu_id%2))
# aug and grad accum after the 1st epoch




python -m src.predict.predict_mitotic_cellwise --fold $fold_i --gpu-id $gpu_id --img_size 512 --batch_size 32 --workers 10 --cell-level-labels-path output/densenet121_pred.h5 --load-state-dict-path "output/models/densenet121_512_cellwise__gradaccum_4__start_lr_4e5/fold${fold_i}/final.pth"  > "cellwise_512_mitotoic_pred_fold${fold_i}.log"

