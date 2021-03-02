#!/bin/bash

nohup python -m src.train.train_bestfitting --fold 0 --img_size 512 --batch_size 32 &
nohup python -m src.train.train_bestfitting --fold 0 --img_size 512 --batch_size 32 --epochs 25 --scheduler-lr-fraction 0.75 --scheduler-epoch-offset 19 --resume 019.pth &
nohup python -m src.train.train_bestfitting --fold 0 --img_size 1024 --batch_size 8 --load-state-dict-path ../output/models/densenet121_512_all_data_obvious_neg/fold0/final.pth &

nohup python -m src.train.train_bestfitting --arch class_efficientnet_dropout --fold 0 --img_size 512 --batch_size 16 &

