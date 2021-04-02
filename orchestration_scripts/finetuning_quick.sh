#!/bin/bash

for fold_i in 1 0; do
  echo "$fold_i"

  gpu_id=$(($fold_i%2))
#    gpu_id=0
  echo "Long waiting for gpu ${gpu_id} to start with fold ${fold_i}"
  python -m orchestration_scripts.active_waiting_for_empty_gpu --gpu-i $gpu_id --sleep-sec $((500+100*$fold_i))
  echo "Putting fold $fold_i into bg.."
  nohup bash orchestration_scripts/finetune_bestfitting_densenet_1024_quick.sh $fold_i &
done