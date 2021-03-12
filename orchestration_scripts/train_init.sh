#!/bin/bash
for fold_i in {0..2}; do
  if [ $fold_i -gt 1 ]; then
    gpu_id=$(($fold_i%2))
#    gpu_id=0
    echo "Long waiting for gpu ${gpu_id} to start with fold ${fold_i}"
    python -m orchestration_scripts.active_waiting_for_empty_gpu --gpu-i $gpu_id --sleep-sec 500
  fi
  echo "Putting fold $fold_i into bg.."
  nohup bash orchestration_scripts/finetune_bestfitting_densenet_1024.sh $fold_i &
done

#python -m src.train.train_bestfitting --fold 0 --img_size 512 --batch_size 32 --scheduler-lr-multiplier 100 --gradient-accumulation-steps 3 --out_dir loss_check --load-state-dict-path "../output/models/densenet121_512_all_data_obvious_neg/fold0/final.pth"


#nohup python -m src.train.train_bestfitting --arch class_efficientnet_dropout --epochs 20 --fold 0 --img_size 512 --batch_size 8  --scheduler-lr-multiplier 0.25 --out_dir efficientnetb3_512 &

#nohup python -m src.train.train_bestfitting --arch class_efficientnet_dropout --epochs 20 --fold 0 --img_size 512 --batch_size 8  --scheduler-lr-multiplier 30 --out_dir efficientnetb1_512 &
#
#nohup python -m src.train.train_bestfitting --arch class_efficientnet_dropout --epochs 30 --fold 0 --img_size 1024 --batch_size 6 --load-state-dict-path ../output/models/efficientnetb1_512/fold0/final.pth --out_dir efficientnetb1_1024 &

