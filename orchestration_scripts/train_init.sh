#!/bin/bash
#nohup python -m src.train.train_bestfitting --fold 0 --img_size 512 --batch_size 32 &
#nohup python -m src.train.train_bestfitting --fold 0 --img_size 512 --batch_size 32 --epochs 25 --scheduler-lr-fraction 0.75 --scheduler-epoch-offset 19 --resume 019.pth &
#nohup python -m src.train.train_bestfitting --fold 0 --img_size 1024 --batch_size 8 --load-state-dict-path ../output/models/densenet121_512_all_data_obvious_neg/fold0/final.pth &
#
#nohup python -m src.train.train_bestfitting --arch class_efficientnet_dropout --fold 0 --img_size 512 --batch_size 16 &
#
#python -m src.train.train_bestfitting --fold 0 --img_size 512 --batch_size 32 --img_size 1024 --batch_size 8
for fold_i in 1 0; do
  echo "$fold_i"
#  if [ $fold_i -gt 1 ]; then
#    gpu_id=$(($fold_i%2))
##    gpu_id=0
#    echo "Long waiting for gpu ${gpu_id} to start with fold ${fold_i}"
#    python -m orchestration_scripts.active_waiting_for_empty_gpu --gpu-i $gpu_id --sleep-sec $((500+100*$fold_i))
#  fi
  gpu_id=$(($fold_i%2))
#    gpu_id=0
  echo "Long waiting for gpu ${gpu_id} to start with fold ${fold_i}"
  python -m orchestration_scripts.active_waiting_for_empty_gpu --gpu-i $gpu_id --sleep-sec $((500+100*$fold_i))
  echo "Putting fold $fold_i into bg.."
  nohup bash orchestration_scripts/finetune_bestfitting_densenet_1024.sh $fold_i &
done