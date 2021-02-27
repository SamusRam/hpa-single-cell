#!/bin/bash

for fold_i in {0..1}; do
  echo "submitting to gpu $fold_i, fold $fold_i"
  nohup python -m src.train.train_bestfitting --fold $fold_i --gpu-id $fold_i &
done
python -m orchestration_scripts.active_waiting_for_empty_gpu
python -m src.train.train_bestfitting --fold 2 --gpu-id 1