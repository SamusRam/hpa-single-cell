#!/bin/bash

nohup python -m src.train.train_bestfitting --fold 0  &
nohup python -m src.train.train_bestfitting --fold 0 --scheduler-lr-fraction 0.5 --resume ../output/models/densenet121_512_all_data_obvious_neg/fold0/final.pth &